# onboarding/recommender_sample.py
#
# Step 3 (Sample): mapping recommender using:
# - profiler output JSON (single or bundle)
# - column_name_lookup.json (global synonyms)
# - schema_registry.json (sample module)
#
# Output:
#   sample_mapping.json with map/compute/default/json_subfields modes.
#
# Usage:
'''
   python onboarding/recommender_sample.py \
     --profile onboarding/ACC_Entity_Sample_profile.json \
     --lookup onboarding/column_name_lookup.json \
     --schema onboarding/configs/schema_registry.json \
     --out onboarding/ACC_Sample_mapping.json
'''

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def to_snake_case(name: str) -> str:
    if name.islower() and "_" in name:
        return name
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = s.replace("__", "_")
    return s.lower()


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def token_set(s: str) -> set:
    return set(re.split(r"[\s_\-\.]+", norm_text(s)))


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def iter_files_from_profile(profile_json: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    if "files" in profile_json and isinstance(profile_json["files"], dict):
        return list(profile_json["files"].items())
    fp = profile_json.get("file_path", "input_file")
    return [(fp, profile_json)]


def load_lookup_normalized(lookup_path: str) -> Dict[str, List[str]]:
    raw = load_json(lookup_path)
    merged: Dict[str, set] = {}
    for k, vals in raw.items():
        kk = to_snake_case(k)
        merged.setdefault(kk, set()).add(str(k).strip())
        for v in vals:
            merged[kk].add(str(v).strip())
    return {k: sorted(vs) for k, vs in merged.items()}


def load_sample_schema(schema_path: str) -> Dict[str, Any]:
    schema = load_json(schema_path)
    return schema["modules"]["sample"]["fields"]


# -------------------------
# scoring
# -------------------------

def synonym_score(col_name: str, col_tokens: List[str], synonyms: List[str]) -> float:
    c_norm = norm_text(col_name)
    syn_norms = [norm_text(x) for x in synonyms]
    if c_norm in syn_norms:
        return 1.0

    cset = set(col_tokens)
    best = 0.0
    for s in synonyms:
        stoks = set(re.split(r"[\s_\-\.]+", norm_text(s)))
        stoks = {t for t in stoks if t}
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
    if target_dtype in ("date", "datetime") and inferred_dtype in ("date", "datetime"):
        return 0.10
    return 0.0


def id_bonus(uniq_ratio: float) -> float:
    return 0.15 if uniq_ratio >= 0.90 else 0.0


def is_id_field(target_key: str) -> bool:
    return target_key.endswith("_id") or target_key in {
        "entity_id", "external_entity_id", "sample_id", "external_sample_id"
    }


STOPWORDS = {"of","to","and","or","in","at","the","a","an","for","on","by","from"}

def distinctive_tokens_for_field(field: str) -> set:
    toks = {t for t in token_set(field) if t and t not in STOPWORDS}
    return toks

def column_tokens(col_prof: Dict[str, Any]) -> set:
    raw = col_prof.get("header", {}).get("raw", "")
    return set(re.split(r"[\s_\-\.]+", norm_text(raw)))


def score_column_for_target(
    target_key: str,
    target_dtype: str,
    synonyms: List[str],
    col_name: str,
    col_prof: Dict[str, Any],
    file_key: str
) -> Tuple[float, List[str]]:
    tokens = col_prof.get("header", {}).get("tokens", [])
    inferred = col_prof.get("inferred_dtype", "string")
    uniq = float(col_prof.get("uniqueness_ratio", 0.0))
    miss = float(col_prof.get("missing_rate", 0.0))
    dt = col_prof.get("date_time", {})

    s = synonym_score(col_name, tokens, synonyms)
    score = float(s)
    reasons: List[str] = []

    if s >= 1.0:
        reasons.append("exact synonym match")
    elif s > 0:
        reasons.append(f"synonym overlap={s/0.9:.2f}")

    # soft penalty if target tokens don't appear (only if not exact match)
    if s < 1.0:
        toks_target = distinctive_tokens_for_field(target_key)
        toks_col = column_tokens(col_prof)
        if toks_target and toks_col and len(toks_target & toks_col) == 0:
            score -= 0.5
            reasons.append("penalized: no distinctive target tokens in column name")

    # prefer clinical-like files for sample fields (small, safe bias)
    if "clinical" in file_key.lower():
        score += 0.05
        reasons.append("small bonus: clinical file")

    # dtype bonus
    b = dtype_bonus(target_dtype, inferred)
    if b:
        score += b
        reasons.append(f"dtype match ({target_dtype}~{inferred})")

    # ID bonus only for ID fields
    if is_id_field(target_key):
        b = id_bonus(uniq)
        if b:
            score += b
            reasons.append(f"id-like uniqueness={uniq:.2f}")

    # missingness penalty
    if miss > 0.5:
        score -= 0.10
        reasons.append(f"high missingness={miss:.2f}")

    return float(score), reasons


def best_column_across_files(
    files: List[Tuple[str, Dict[str, Any]]],
    lookup: Dict[str, List[str]],
    schema_fields: Dict[str, Any],
    target_key: str,
    default_dtype: str = "string",
) -> Dict[str, Any]:
    synonyms = lookup.get(target_key, [target_key])
    target_dtype = schema_fields.get(target_key, {}).get("dtype", default_dtype)

    best = {"file": None, "column": None, "score": -1e9, "reasons": ["not found"]}
    for fp, prof in files:
        cols = prof.get("columns", {})
        for col_name, col_prof in cols.items():
            score, reasons = score_column_for_target(target_key, target_dtype, synonyms, col_name, col_prof, fp)
            if score > best["score"]:
                best = {"file": fp, "column": col_name, "score": score, "reasons": reasons}
    return best


def min_score_for_field(field: str, meta: Dict[str, Any]) -> float:
    # sample module: be strict about IDs, looser for optional metadata
    if field in {"external_sample_id", "external_entity_id"}:
        return 0.8
    if field.endswith("_id"):
        return 0.8
    dtype = meta.get("dtype", "string")
    if dtype in ("date", "datetime"):
        return 0.9
    return 0.6


# -------------------------
# Main recommender (full coverage)
# -------------------------

def recommend_sample(profile_path: str, lookup_path: str, schema_path: str) -> Dict[str, Any]:
    profile_json = load_json(profile_path)
    files = iter_files_from_profile(profile_json)
    lookup = load_lookup_normalized(lookup_path)
    schema_fields = load_sample_schema(schema_path)

    plan: Dict[str, Any] = {}

    # Required IDs
    plan["external_sample_id"] = {"mode": "map", "source": best_column_across_files(files, lookup, schema_fields, "external_sample_id")}
    plan["external_entity_id"] = {"mode": "map", "source": best_column_across_files(files, lookup, schema_fields, "external_entity_id")}

    # Derived / linked fields
    plan["sample_id"] = {"mode": "compute", "recipe": "uuid4_per_external_sample_id"}
    plan["entity_id"] = {"mode": "compute", "recipe": "lookup_entity_id_from_entity_table_by_external_entity_id"}

    # All other fields from schema
    for field, meta in schema_fields.items():
        if field in plan:
            continue

        # JSON blobs: explicit subfield mapping
        if meta.get("dtype") == "json" and isinstance(meta.get("subfields"), dict):
            subplan = {}
            for subfield, submeta in meta["subfields"].items():
                best = best_column_across_files(files, lookup, schema_fields, subfield, default_dtype=submeta.get("dtype", "string"))
                threshold = min_score_for_field(subfield, submeta)
                if best["column"] is None or best["score"] < threshold:
                    subplan[subfield] = {"mode": "default", "value": ""}
                else:
                    subplan[subfield] = {"mode": "map", "source": best}
            plan[field] = {"mode": "json_subfields", "subfields": subplan}
            continue

        # Normal fields: map or default
        best = best_column_across_files(files, lookup, schema_fields, field, default_dtype=meta.get("dtype", "string"))
        threshold = min_score_for_field(field, meta)
        if best["column"] is None or best["score"] < threshold:
            plan[field] = {"mode": "default", "value": ""}
        else:
            plan[field] = {"mode": "map", "source": best}

    return {
        "module": "sample",
        "inputs_profile": profile_path,
        "mappings": plan
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = recommend_sample(args.profile, args.lookup, args.schema)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote sample mapping to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()