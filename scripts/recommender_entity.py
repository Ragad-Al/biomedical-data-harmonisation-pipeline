# onboarding/recommender_entity.py
#
# Step 3 (Entity): mapping recommender using ONLY:
# - profiler output JSON (single or bundle)
# - column_name_lookup.json (global synonyms)
# - schema_registry.json (for roles/dtypes, optional but recommended)
#
# It outputs a mapping config with:
# - map vs compute decisions
# - per-file chosen columns for recipe inputs (DOB/DOD logic)
# - join-consistent external_entity_id selection across clinical + survival files
#
# Usage:
'''
   python onboarding/recommender_entity.py \
     --profile onboarding/ACC_Entity_profile.json \
     --lookup onboarding/column_name_lookup.json \
     --schema onboarding/configs/schema_registry.json \
     --out onboarding/ACC_Entity_mapping.json

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


def load_entity_schema(schema_path: str) -> Dict[str, Any]:
    schema = load_json(schema_path)
    return schema["modules"]["entity"]["fields"]


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


def date_bonus(target_dtype: str, dt_flags: Dict[str, Any]) -> float:
    if target_dtype == "date" and dt_flags.get("date_like"):
        return 0.15
    if target_dtype == "datetime" and (dt_flags.get("datetime_like") or dt_flags.get("epoch_seconds_like") or dt_flags.get("epoch_millis_like")):
        return 0.15
    return 0.0

STOPWORDS = {"of","to","and","or","in","at","the","a","an","for","on","by","from"}

def distinctive_tokens_for_field(field: str) -> set:
    toks = {t for t in token_set(field) if t and t not in STOPWORDS}
    return toks

def column_tokens(col_prof: Dict[str, Any]) -> set:
    # Use the raw header string and split on dots too
    raw = col_prof.get("header", {}).get("raw", "")
    return set(re.split(r"[\s_\-\.]+", norm_text(raw)))

def is_id_field(target_key: str) -> bool:
    return target_key.endswith("_id") or target_key in {
        "entity_id", "external_entity_id", "sample_id", "external_sample_id"
    }

def score_column_for_target(
    target_key: str,
    target_dtype: str,
    synonyms: List[str],
    col_name: str,
    col_prof: Dict[str, Any]
) -> Tuple[float, List[str]]:
    tokens = col_prof.get("header", {}).get("tokens", [])
    inferred = col_prof.get("inferred_dtype", "string")
    uniq = float(col_prof.get("uniqueness_ratio", 0.0))
    miss = float(col_prof.get("missing_rate", 0.0))
    dt = col_prof.get("date_time", {})

    # Start from synonym score
    s = synonym_score(col_name, tokens, synonyms)
    score = float(s)
    reasons: List[str] = []

    if s >= 1.0:
        reasons.append("exact synonym match")
    elif s > 0:
        reasons.append(f"synonym overlap={s/0.9:.2f}")

    # Distinctive-token penalty ONLY if not exact synonym match
    if s < 1.0:
        toks_target = distinctive_tokens_for_field(target_key)
        toks_col = column_tokens(col_prof)
        if toks_target and toks_col and len(toks_target & toks_col) == 0:
            score -= 0.5
            reasons.append("penalized: no distinctive target tokens in column name")

    # Date/datetime gating: only apply if target expects date/datetime
    if target_dtype in ("date", "datetime"):
        is_dt = (
            dt.get("date_like")
            or dt.get("datetime_like")
            or dt.get("epoch_seconds_like")
            or dt.get("epoch_millis_like")
        )
        if not is_dt:
            score -= 1.0
            reasons.append("penalized: target is date/datetime but column not date-like")

    # dtype bonus
    b = dtype_bonus(target_dtype, inferred)
    if b:
        score += b
        reasons.append(f"dtype match ({target_dtype}~{inferred})")

    # ID bonus only for true ID fields
    if is_id_field(target_key):
        b = id_bonus(uniq)
        if b:
            score += b
            reasons.append(f"id-like uniqueness={uniq:.2f}")

    # extra date bonus (only helps if date-like)
    b = date_bonus(target_dtype, dt)
    if b:
        score += b
        reasons.append("date/datetime pattern detected")

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
    topk: int = 3,
) -> Dict[str, Any]:
    synonyms = lookup.get(target_key, [target_key])
    target_dtype = schema_fields.get(target_key, {}).get("dtype", default_dtype)

    best = {"file": None, "column": None, "score": -1e9, "reasons": ["not found"]}
    for fp, prof in files:
        cols = prof.get("columns", {})
        for col_name, col_prof in cols.items():
            score, reasons = score_column_for_target(target_key, target_dtype, synonyms, col_name, col_prof)
            if score > best["score"]:
                best = {"file": fp, "column": col_name, "score": score, "reasons": reasons}
    return best


def best_columns_per_file(
    files: List[Tuple[str, Dict[str, Any]]],
    lookup: Dict[str, List[str]],
    schema_fields: Dict[str, Any],
    target_key: str,
    default_dtype: str = "string",
    topk: int = 3,
) -> Dict[str, Any]:
    """
    For join consistency needs: choose best candidate per file.
    """
    synonyms = lookup.get(target_key, [target_key])
    target_dtype = schema_fields.get(target_key, {}).get("dtype", default_dtype)

    out = {}
    for fp, prof in files:
        cols = prof.get("columns", {})
        scored = []
        for col_name, col_prof in cols.items():
            score, reasons = score_column_for_target(target_key, target_dtype, synonyms, col_name, col_prof)
            scored.append((col_name, score, reasons))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            col, score, reasons = scored[0]
            out[fp] = {"source_column": col, "score": score, "reasons": reasons}
        else:
            out[fp] = {"source_column": None, "score": 0.0, "reasons": ["no columns"]}
    return out


def join_consistency_bonus(col_a: str, col_b: str, prof_a: Dict[str, Any], prof_b: Dict[str, Any]) -> Tuple[float, str]:
    """
    Light join-consistency: token similarity + TCGA-like agreement.
    """
    if col_a is None or col_b is None:
        return 0.0, "missing candidate"

    cols_a = prof_a.get("columns", {})
    cols_b = prof_b.get("columns", {})
    a = cols_a.get(col_a, {})
    b = cols_b.get(col_b, {})

    bonus = 0.0
    expl = []

    # TCGA-like agreement
    if a.get("patterns", {}).get("tcga_like") and b.get("patterns", {}).get("tcga_like"):
        bonus += 0.20
        expl.append("both TCGA-like")

    # token similarity
    ta = token_set(col_a)
    tb = token_set(col_b)
    if ta and tb:
        j = len(ta & tb) / len(ta | tb)
        if j >= 0.25:
            bonus += 0.05
            expl.append(f"name token similarity={j:.2f}")

    return bonus, "; ".join(expl) if expl else "no extra evidence"


# -------------------------
# Recipes (DOB/DOD inputs only)
# -------------------------

def is_date_like_column(col_prof: Dict[str, Any]) -> bool:
    dt = col_prof.get("date_time", {})
    return bool(
        dt.get("date_like")
        or dt.get("datetime_like")
        or dt.get("epoch_seconds_like")
        or dt.get("epoch_millis_like")
    )

def recipe_inputs(files, lookup, schema_fields, keys: List[str]) -> Dict[str, Any]:
    """
    Select best input column for each recipe key.
    Special rule:
      - diagnosis_date must be date/datetime-like; otherwise output None so pipeline falls back to year_of_diagnosis.
    """
    out = {}
    for k in keys:
        best = best_column_across_files(files, lookup, schema_fields, k)

        if k == "diagnosis_date":
            # Validate that chosen column is truly date/datetime-like
            if best["file"] is not None and best["column"] is not None:
                # find the column profile to check date_time flags
                file_prof = dict(files).get(best["file"], {})
                col_prof = file_prof.get("columns", {}).get(best["column"], {})
                if not is_date_like_column(col_prof):
                    # Force to None so DOB/DOD logic uses year_of_diagnosis fallback
                    best = {"file": None, "column": None, "score": best["score"], "reasons": ["rejected: not date-like"]}
            else:
                best = {"file": None, "column": None, "score": best.get("score", 0.0), "reasons": ["not found"]}

        out[k] = best
    return out

def min_score_for_field(field: str, meta: Dict[str, Any]) -> float:
    high_risk = {
        "cause_of_death", "comorbidities", "family_history",
        "cardiovascular_disease", "prior_treatments", "performance_status",
        "date_of_death", "date_of_birth", "diagnosis_date",
    }

    if field in high_risk:
        return 1.0

    dtype = meta.get("dtype", "string")
    if dtype in ("date", "datetime"):
        return 0.9

    if field.endswith("_id") or field in {"external_entity_id"}:
        return 0.8

    return 0.6
# -------------------------
# Main recommender (full coverage)
# -------------------------

def recommend_entity_full(profile_path: str, lookup_path: str, schema_path: str) -> Dict[str, Any]:
    profile_json = load_json(profile_path)
    files = iter_files_from_profile(profile_json)
    lookup = load_lookup_normalized(lookup_path)
    schema_fields = load_entity_schema(schema_path)

    # Build mapping plan for every field in schema
    plan: Dict[str, Any] = {}

    # Special: external_entity_id join-consistent across files
    per_file = best_columns_per_file(files, lookup, schema_fields, "external_entity_id")

    join_info = {"bonus": 0.0, "explanation": "single file"}
    if len(files) >= 2:
        (f1, p1), (f2, p2) = files[0], files[1]
        c1 = per_file.get(f1, {}).get("source_column")
        c2 = per_file.get(f2, {}).get("source_column")
        bonus, expl = join_consistency_bonus(c1, c2, p1, p2)
        join_info = {"bonus": bonus, "explanation": expl}

    plan["external_entity_id"] = {
        "mode": "map",
        "per_file": per_file,
        "join_consistency": join_info
    }

    # Special: entity_id always computed (uuid4)
    plan["entity_id"] = {"mode": "compute", "recipe": "uuid4"}

    # Special: date_of_birth and date_of_death computed using your recipes
    plan["date_of_birth"] = {
        "mode": "compute",
        "recipe": "dob_ref_diag_then_days_to_birth_else_year_of_birth",
        "inputs": recipe_inputs(files, lookup, schema_fields, ["diagnosis_date", "year_of_diagnosis", "days_to_birth", "year_of_birth"])
    }
    plan["date_of_death"] = {
        "mode": "compute",
        "recipe": "dod_ref_diag_then_days_to_death_else_year_of_death",
        "inputs": recipe_inputs(files, lookup, schema_fields, ["diagnosis_date", "year_of_diagnosis", "days_to_death", "year_of_death"])
    }

    # All other schema fields:
    # - if found via lookup+profile => map
    # - else default to ""
    for field, meta in schema_fields.items():
        if field in plan:
            continue  # already handled
        if meta.get("source_role") == "derived":
            # If schema marks derived but we don't have a specific recipe, default to ""
            plan[field] = {"mode": "default", "value": ""}
            continue

        # JSON blobs: explicit subfield mapping (Option B)
        if meta.get("dtype") == "json" and isinstance(meta.get("subfields"), dict):
            subplan = {}
            for subfield, submeta in meta["subfields"].items():
                best = best_column_across_files(
                    files, lookup, schema_fields,
                    subfield,
                    default_dtype=submeta.get("dtype", "string")
                )

                threshold = min_score_for_field(subfield, submeta)
                if best["column"] is None or best["score"] < threshold:
                    subplan[subfield] = {"mode": "default", "value": ""}
                else:
                    subplan[subfield] = {"mode": "map", "source": best}
            plan[field] = {"mode": "json_subfields", "subfields": subplan}
            continue

        # Normal field mapping
        best = best_column_across_files(
            files, lookup, schema_fields,
            field,
            default_dtype=meta.get("dtype", "string")
        )

        threshold = min_score_for_field(field, meta)
        if best["column"] is None or best["score"] < threshold:
            plan[field] = {"mode": "default", "value": ""}
        else:
            plan[field] = {"mode": "map", "source": best}

    return {
        "module": "entity",
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

    out = recommend_entity_full(args.profile, args.lookup, args.schema)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote full entity mapping to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()