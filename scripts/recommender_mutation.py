# onboarding/recommender_mutation.py
#
# Step 3 (Mutation): produce mapping + transform suggestions
#
# Inputs:
#   --profile  ACC_Mutation_profile.json   (from profiler.py)
#   --lookup   column_name_lookup.json     (synonyms)
#   --out      ACC_Mutation_mapping.json
#
# Optional:
#   --schema   schema_registry.json        (if provided, uses mutation module fields)
#
# Usage:

'''
python onboarding/recommender_mutation.py \
  --profile onboarding/ACC_Mutation_profile.json \
  --lookup  onboarding/column_name_lookup.json \
  --out     onboarding/ACC_Mutation_mapping.json \
  --schema  onboarding/configs/schema_registry.json
'''

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


# -------------------------
# helpers
# -------------------------

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
        kk = norm_text(k).replace(" ", "_")
        merged.setdefault(kk, set()).add(str(k).strip())
        for v in vals:
            merged[kk].add(str(v).strip())
    return {k: sorted(vs) for k, vs in merged.items()}


# -------------------------
# schema fields (fallback if schema_registry.json not provided)
# -------------------------

FALLBACK_MUTATION_FIELDS = {
    # required
    "external_sample_id": {"dtype": "string", "required_input": True},   # for linking to sample_id
    "sample_id": {"dtype": "string", "required_output": True},          # produced by mutation.py via sample table join
    "chromosome": {"dtype": "string", "required_output": True},
    "position_start": {"dtype": "int", "required_output": True},
    "position_end": {"dtype": "int", "required_output": False},
    "reference_allele": {"dtype": "string", "required_output": True},
    "alternate_allele": {"dtype": "string", "required_output": True},

    # common optional
    "gene_symbol": {"dtype": "string"},
    "consequence": {"dtype": "string"},
    "amino_acid_change": {"dtype": "string"},
    "nucleotide_change": {"dtype": "string"},
    "dna_variant_allele_frequency": {"dtype": "float"},
    "variant_caller_software": {"dtype": "string"},
}


def load_mutation_schema(schema_path: Optional[str]) -> Dict[str, Any]:
    if not schema_path:
        return FALLBACK_MUTATION_FIELDS
    schema = load_json(schema_path)
    # schema_registry.json format: modules -> mutation -> fields
    # We still add external_sample_id for linking if not present.
    fields = schema["modules"]["mutation"]["fields"]
    out = {k: v for k, v in fields.items()}
    if "external_sample_id" not in out:
        out["external_sample_id"] = {"dtype": "string", "required_input": True}
    return out


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
        stoks = token_set(s)
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
    return 0.0


def id_bonus(uniq_ratio: float) -> float:
    return 0.15 if uniq_ratio >= 0.90 else 0.0


def is_id_field(key: str) -> bool:
    return key.endswith("_id") or key in {"external_sample_id", "sample_id"}


def score_column_for_target(
    target_key: str,
    target_dtype: str,
    synonyms: List[str],
    col_name: str,
    col_prof: Dict[str, Any],
) -> Tuple[float, List[str]]:
    tokens = col_prof.get("header", {}).get("tokens", [])
    inferred = col_prof.get("inferred_dtype", "string")
    uniq = float(col_prof.get("uniqueness_ratio", 0.0))
    miss = float(col_prof.get("missing_rate", 0.0))
    patterns = col_prof.get("patterns", {})

    s = synonym_score(col_name, tokens, synonyms)
    score = float(s)
    reasons: List[str] = []

    if s >= 1.0:
        reasons.append("exact synonym match")
    elif s > 0:
        reasons.append(f"synonym overlap≈{s/0.9:.2f}")

    # dtype evidence
    b = dtype_bonus(target_dtype, inferred)
    if b:
        score += b
        reasons.append(f"dtype match ({target_dtype}~{inferred})")

    # ID evidence
    if is_id_field(target_key):
        b = id_bonus(uniq)
        if b:
            score += b
            reasons.append(f"id-like uniqueness={uniq:.2f}")

    # pattern hint for chromosome
    if target_key == "chromosome" and patterns.get("chromosome_like"):
        score += 0.10
        reasons.append("chromosome-like pattern")

    # missingness
    if miss > 0.5:
        score -= 0.10
        reasons.append(f"high missingness={miss:.2f}")

    return float(score), reasons


def best_column(
    prof: Dict[str, Any],
    lookup: Dict[str, List[str]],
    schema_fields: Dict[str, Any],
    target_key: str,
) -> Dict[str, Any]:
    cols = prof.get("columns", {})
    synonyms = lookup.get(target_key, [target_key])
    target_dtype = schema_fields.get(target_key, {}).get("dtype", "string")

    best = {"file": prof.get("file_path", "input_file"), "column": None, "score": -1e9, "reasons": ["not found"]}
    for col_name, col_prof in cols.items():
        score, reasons = score_column_for_target(target_key, target_dtype, synonyms, col_name, col_prof)
        if score > best["score"]:
            best = {"file": prof.get("file_path", "input_file"), "column": col_name, "score": score, "reasons": reasons}
    return best


# -------------------------
# transform suggestions
# -------------------------

def has_col(prof: Dict[str, Any], col: str) -> bool:
    return col in prof.get("columns", {})


def suggest_transforms(prof: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return field -> transform spec (only when needed).
    """
    t: Dict[str, Any] = {}

    # Normalize chromosome (strip "chr" prefix) – safe even if not present.
    t["chromosome"] = {
        "template": "strip_chr_prefix",
        "params": {"case_insensitive": True}
    }

    # Uppercase alleles
    t["reference_allele"] = {"template": "uppercase", "params": {}}
    t["alternate_allele"] = {"template": "uppercase", "params": {}}

    # VAF: prefer existing VAF column; else compute if counts exist
    # (this profile file does NOT include counts, but we keep logic generic)
    if mapping.get("dna_variant_allele_frequency", {}).get("mode") == "map":
        # still keep a clamp to [0,1] as a suggestion
        t["dna_variant_allele_frequency"] = {"template": "clamp_0_1", "params": {"invalid_to": ""}}
    else:
        if has_col(prof, "t_alt_count") and has_col(prof, "t_ref_count"):
            t["dna_variant_allele_frequency"] = {
                "template": "vaf_from_counts",
                "inputs": {"alt": "t_alt_count", "ref": "t_ref_count"},
                "params": {"zero_div_to": ""}
            }

    # Alternate allele: if Tumor_Seq_Allele2 exists, suggest using it
    # (not present in this profile, but useful generically)
    if has_col(prof, "Tumor_Seq_Allele2"):
        t["alternate_allele"] = {
            "template": "choose_first_non_null",
            "inputs": ["Tumor_Seq_Allele2", "Tumor_Seq_Allele1", mapping.get("alternate_allele", {}).get("source", {}).get("column")],
            "params": {"uppercase": True}
        }

    # Nucleotide / amino acid change
    if has_col(prof, "HGVSc"):
        t["nucleotide_change"] = {"template": "passthrough", "inputs": ["HGVSc"], "params": {}}
    if has_col(prof, "HGVSp") or has_col(prof, "HGVSp_Short"):
        t["amino_acid_change"] = {
            "template": "choose_first_non_null",
            "inputs": [c for c in ["HGVSp", "HGVSp_Short", "Amino_Acid_Change"] if has_col(prof, c)],
            "params": {}
        }

    # Consequence fallback
    if has_col(prof, "Consequence") or has_col(prof, "Variant_Classification"):
        t["consequence"] = {
            "template": "choose_first_non_null",
            "inputs": [c for c in ["Consequence", "Variant_Classification", "effect"] if has_col(prof, c)],
            "params": {}
        }

    return t


# -------------------------
# main recommend
# -------------------------

def recommend_mutation(profile_path: str, lookup_path: str, out_path: str, schema_path: Optional[str] = None) -> Dict[str, Any]:
    profile_json = load_json(profile_path)
    files = iter_files_from_profile(profile_json)
    # mutation file profile is usually single file; take first
    fp, prof = files[0]

    lookup = load_lookup_normalized(lookup_path)
    schema_fields = load_mutation_schema(schema_path)

    plan: Dict[str, Any] = {}

    # Build mapping plan for each schema field
    for field, meta in schema_fields.items():
        # sample_id is not in the input file (it’s linked via sample table)
        if field == "sample_id":
            plan[field] = {"mode": "compute", "recipe": "lookup_sample_id_from_sample_table_by_external_sample_id"}
            continue

        best = best_column(prof, lookup, schema_fields, field)
        # thresholding
        threshold = 0.8 if field in {"external_sample_id", "chromosome", "position_start", "reference_allele", "alternate_allele"} else 0.6
        if best["column"] is None or best["score"] < threshold:
            plan[field] = {"mode": "default", "value": ""}
        else:
            plan[field] = {"mode": "map", "source": best}

    # Add transform suggestions
    transforms = suggest_transforms(prof, plan)

    return {
        "module": "mutation",
        "inputs_profile": profile_path,
        "mappings": plan,
        "transforms": transforms,
        "notes": [
            "sample_id is expected to be resolved via sample table join using external_sample_id.",
            "If schema_registry.json is not provided, a minimal mutation schema is used."
        ]
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema", default=None, help="Optional schema_registry.json")
    args = ap.parse_args()

    out = recommend_mutation(args.profile, args.lookup, args.out, schema_path=args.schema)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote mutation mapping to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()