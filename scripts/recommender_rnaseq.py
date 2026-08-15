# onboarding/recommender_rnaseq.py
'''
python onboarding/recommender_rnaseq.py \
  --profile onboarding/ACC_RNASEQ_profile.json \
  --out onboarding/ACC_RNASEQ_mapping.json

'''

import json
import re
from pathlib import Path
from typing import Dict, Any, List


def load_json(p: str) -> Dict[str, Any]:
    return json.loads(Path(p).read_text())

def recommend_rnaseq(profile_path: str, out_path: str) -> Dict[str, Any]:
    prof = load_json(profile_path)
    file_key = prof.get("file_path", "input_file")
    cols = list(prof.get("columns", {}).keys())

    # Identify feature column (preferred: Ensembl_ID)
    feature_col = None
    for c in cols:
        if c.lower() in {"ensembl_id", "ensembl"}:
            feature_col = c
            break
    if feature_col is None:
        # fallback: first column
        feature_col = cols[0] if cols else None

    # Sample columns = everything except feature col
    sample_cols = [c for c in cols if c != feature_col]

    mapping = {
        "module": "rnaseq",
        "inputs_profile": profile_path,
        "mappings": {
            # Wide-matrix identifiers
            "feature_id_col": {"mode": "map", "source": {"file": file_key, "column": feature_col}},
            "sample_cols": {"mode": "compute", "recipe": "all_except_feature", "inputs": {"feature_col": feature_col}},

            # Output schema fields (long format)
            "external_sample_id": {"mode": "compute", "recipe": "from_column_headers"},
            "sample_id": {"mode": "compute", "recipe": "lookup_sample_id_from_sample_table_by_external_sample_id"},
            "ensembl_id": {"mode": "map", "source": {"file": file_key, "column": feature_col}},
            "expression_value": {"mode": "compute", "recipe": "from_matrix_values"},

            # Optional fields (default unless present in long-format inputs)
            "gene_symbol": {"mode": "default", "value": ""},
            "entrez_gene_id": {"mode": "default", "value": ""},
            "transcript_id": {"mode": "default", "value": ""},
            "quantification_method": {"mode": "default", "value": ""},
            "normalization_method": {"mode": "default", "value": ""},
            "total_reads_input_for_quantification": {"mode": "default", "value": ""},
            "total_reads_mapped": {"mode": "default", "value": ""},
            "percent_reads_mapped": {"mode": "default", "value": ""},
            "expression_matrix_path_url": {"mode": "default", "value": ""},
            "log2_fold_change": {"mode": "default", "value": ""},
            "p_value": {"mode": "default", "value": ""},
            "adjusted_p_value": {"mode": "default", "value": ""},
            "expression_status": {"mode": "default", "value": ""},
            "differential_expression_method": {"mode": "default", "value": ""},
            "gene_annotation_version": {"mode": "default", "value": ""},
        },
        "transforms": {
            # Strip Ensembl version suffix: ENSG... .15 -> ENSG...
            "ensembl_id": {"template": "strip_suffix_regex", "inputs": "Ensembl_ID", "params": {"pattern": r"\\.[0-9]+$"}},

            # Ensure numeric
            "expression_value": {"template": "to_float", "inputs": "Expression_Value", "params": {"invalid_to": ""}},
        },
        "wide_matrix": {
            "feature_col": feature_col,
            "sample_cols": sample_cols
        }
    }

    Path(out_path).write_text(json.dumps(mapping, indent=2))
    print(f"Wrote RNA-seq mapping to {Path(out_path).resolve()}")
    return mapping

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recommend_rnaseq(args.profile, args.out)

if __name__ == "__main__":
    main()