import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

'''
python onboarding/recommender_methylation.py \
  --profile onboarding/ACC_METH_profile.json \
  --lookup onboarding/column_name_lookup.json \
  --schema onboarding/configs/schema_registry.json \
  --out onboarding/ACC_METH_mapping.json

'''


def load_json(p: str) -> Dict[str, Any]:
    return json.loads(Path(p).read_text())


def iter_files_from_profile(profile_json: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    if "files" in profile_json and isinstance(profile_json["files"], dict):
        return list(profile_json["files"].items())
    fp = profile_json.get("file_path", "input_file")
    return [(fp, profile_json)]


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def token_set(s: str) -> set:
    return set(re.split(r"[\s_\-\.]+", norm_text(s)))


def to_snake_case(name: str) -> str:
    if name.islower() and "_" in name:
        return name
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return s.replace("__", "_").lower()


def load_lookup_normalized(lookup_path: str) -> Dict[str, List[str]]:
    raw = load_json(lookup_path)
    merged: Dict[str, set] = {}
    for k, vals in raw.items():
        kk = to_snake_case(k)
        merged.setdefault(kk, set()).add(str(k).strip())
        for v in vals:
            merged[kk].add(str(v).strip())
    return {k: sorted(vs) for k, vs in merged.items()}


def load_methylation_schema(schema_path: str) -> Dict[str, Any]:
    schema = load_json(schema_path)
    return schema["modules"]["methylation"]["fields"]


# -------------------------
# scoring (simple + safe)
# -------------------------

def synonym_score(col_name: str, col_tokens: List[str], synonyms: List[str]) -> float:
    c_norm = norm_text(col_name)
    syn_norms = [norm_text(x) for x in synonyms]
    if c_norm in syn_norms:
        return 1.0
    cset = set(col_tokens)
    best = 0.0
    for s in synonyms:
        stoks = token_set(s)
        if not stoks:
            continue
        overlap = len(cset & stoks) / len(stoks)
        best = max(best, overlap)
    return 0.9 * best


def dtype_bonus(target_dtype: str, inferred_dtype: str) -> float:
    if target_dtype == inferred_dtype:
        return 0.15
    if target_dtype in ("int", "float") and inferred_dtype in ("int", "float"):
        return 0.10
    return 0.0


def score_column_for_target(target_key: str, target_dtype: str, synonyms: List[str], col_name: str, col_prof: Dict[str, Any]) -> Tuple[float, List[str]]:
    tokens = col_prof.get("header", {}).get("tokens", [])
    inferred = col_prof.get("inferred_dtype", "string")
    miss = float(col_prof.get("missing_rate", 0.0))

    s = synonym_score(col_name, tokens, synonyms)
    score = float(s)
    reasons: List[str] = []

    if s >= 1.0:
        reasons.append("exact synonym match")
    elif s > 0:
        reasons.append(f"synonym overlap≈{s/0.9:.2f}")

    b = dtype_bonus(target_dtype, inferred)
    if b:
        score += b
        reasons.append(f"dtype match ({target_dtype}~{inferred})")

    if miss > 0.5:
        score -= 0.10
        reasons.append(f"high missingness={miss:.2f}")

    return score, reasons


def best_column_in_file(file_prof: Dict[str, Any], lookup: Dict[str, List[str]], schema_fields: Dict[str, Any], target_key: str) -> Dict[str, Any]:
    cols = file_prof.get("columns", {})
    synonyms = lookup.get(target_key, [target_key])
    target_dtype = schema_fields.get(target_key, {}).get("dtype", "string")

    best = {"file": file_prof.get("file_path", "input_file"), "column": None, "score": -1e9, "reasons": ["not found"]}
    for col_name, col_prof in cols.items():
        score, reasons = score_column_for_target(target_key, target_dtype, synonyms, col_name, col_prof)
        if score > best["score"]:
            best = {"file": file_prof.get("file_path", "input_file"), "column": col_name, "score": score, "reasons": reasons}
    return best


def pick_matrix_and_manifest(files: List[Tuple[str, Dict[str, Any]]]) -> Tuple[Tuple[str, Dict[str, Any]], Optional[Tuple[str, Dict[str, Any]]]]:
    """
    Choose matrix file as the one detected wide matrix (or the one with the most columns).
    Manifest file is the other one (if present).
    """
    if len(files) == 1:
        return files[0], None

    # prefer one flagged as wide matrix
    for fp, prof in files:
        wide = prof.get("wide_matrix_detection", {}).get("is_wide_matrix", False)
        if wide:
            # choose the other as manifest
            other = next(((fpp, pp) for (fpp, pp) in files if fpp != fp), None)
            return (fp, prof), other

    # fallback: choose max columns as matrix
    files_sorted = sorted(files, key=lambda x: int(x[1].get("shape", {}).get("n_cols", 0)), reverse=True)
    matrix = files_sorted[0]
    manifest = files_sorted[1] if len(files_sorted) > 1 else None
    return matrix, manifest


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    profile_json = load_json(args.profile)
    files = iter_files_from_profile(profile_json)
    lookup = load_lookup_normalized(args.lookup)
    schema_fields = load_methylation_schema(args.schema)

    (matrix_fp, matrix_prof), manifest = pick_matrix_and_manifest(files)
    manifest_fp, manifest_prof = (manifest if manifest else (None, None))

    # Matrix columns
    matrix_cols = list(matrix_prof.get("columns", {}).keys())

    # Feature column for CpG ID: choose best match for cpg_id in matrix
    best_cpg = best_column_in_file(matrix_prof, lookup, schema_fields, "cpg_id")
    feature_col = best_cpg["column"] or (matrix_cols[0] if matrix_cols else None)
    sample_cols = [c for c in matrix_cols if c != feature_col]

    mappings: Dict[str, Any] = {
        "feature_id_col": {"mode": "map", "source": {"file": matrix_fp, "column": feature_col}},
        "sample_cols": {"mode": "compute", "recipe": "all_except_feature", "inputs": {"feature_col": feature_col}},
        "cpg_id": {"mode": "map", "source": {"file": matrix_fp, "column": feature_col}},
        "beta_value": {"mode": "compute", "recipe": "from_matrix_values"},
        "external_sample_id": {"mode": "compute", "recipe": "from_column_headers"},
        "sample_id": {"mode": "compute", "recipe": "lookup_sample_id_from_sample_table_by_external_sample_id"},
    }

    # Add manifest-based optional fields automatically (if manifest exists)
    # Only include if score passes a reasonable threshold.
    if manifest_prof:
        for target in ["chromosome", "position_start", "position_end", "gene_symbol", "region_context"]:
            best = best_column_in_file(manifest_prof, lookup, schema_fields, target)
            threshold = 0.8 if target in {"chromosome", "position_start", "position_end"} else 0.6
            if best["column"] is not None and best["score"] >= threshold:
                mappings[target] = {"mode": "map", "source": {"file": manifest_fp, "column": best["column"]}}
            else:
                mappings[target] = {"mode": "default", "value": ""}

    transforms = {
        "beta_value": {"template": "to_float", "inputs": "Beta_Value", "params": {"invalid_to": ""}},
        "beta_value__clamp": {"template": "clamp_0_1", "inputs": "beta_value", "params": {"invalid_to": ""}},
        "chromosome": {"template": "strip_chr_prefix", "inputs": "Chromosome", "params": {"case_insensitive": True}},
        "position_start": {"template": "to_int", "inputs": "Position_Start", "params": {"invalid_to": ""}},
        "position_end": {"template": "to_int", "inputs": "Position_End", "params": {"invalid_to": ""}},
    }

    out = {
        "module": "methylation",
        "inputs_profile": args.profile,
        "matrix_file": matrix_fp,
        "manifest_file": manifest_fp,
        "mappings": mappings,
        "transforms": transforms,
        "wide_matrix": {"feature_col": feature_col, "sample_cols": sample_cols},
    }

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote methylation mapping to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()