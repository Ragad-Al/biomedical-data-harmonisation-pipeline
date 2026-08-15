# onboarding/recommender_sv.py

'''
python onboarding/recommender_sv.py \
  --profile onboarding/pcawg_donor_SV_profile.json \
  --out onboarding/PCAWG_SV_mapping.json

'''


import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

def load_json(p: str) -> Dict[str, Any]:
    return json.loads(Path(p).read_text())

def iter_files_from_profile(profile_json: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    if "files" in profile_json and isinstance(profile_json["files"], dict):
        return list(profile_json["files"].items())
    return [(profile_json.get("file_path", "input_file"), profile_json)]

def recommend_sv(profile_path: str, out_path: str) -> Dict[str, Any]:
    prof = load_json(profile_path)
    files = iter_files_from_profile(prof)
    fp, p = files[0]

    # We know exact headers from your profile: sample, chr, start, end, gene, altGene, effect, reference, alt
    # (confirmed in pcawg_donor_SV_profile.json)
    mapping = {
        "module": "sv",
        "inputs_profile": profile_path,
        "mappings": {
            # linkage key (external sample id in SV input)
            "external_sample_id": {"mode": "map", "source": {"file": fp, "column": "sample"}},

            # canonical SV fields (snake_case)
            "chromosome": {"mode": "map", "source": {"file": fp, "column": "chr"}},
            "position_start": {"mode": "map", "source": {"file": fp, "column": "start"}},
            "position_end": {"mode": "map", "source": {"file": fp, "column": "end"}},

            # optional
            "gene_symbol": {"mode": "map", "source": {"file": fp, "column": "gene"}},
            "sv_type": {"mode": "map", "source": {"file": fp, "column": "effect"}},
            "alternate_gene_symbol": {"mode": "map", "source": {"file": fp, "column": "altGene"}},
        },
        "transforms": {
            # normalize chr prefix
            "chromosome": {"template": "strip_chr_prefix", "inputs": "Chromosome", "params": {"case_insensitive": True}},

            # ensure coordinates are ints
            "position_start": {"template": "to_int", "inputs": "Position_Start", "params": {"invalid_to": ""}},
            "position_end": {"template": "to_int", "inputs": "Position_End", "params": {"invalid_to": ""}},

            # keep SV type as-is (optional uppercase)
            # "sv_type": {"template": "uppercase", "inputs": "SV_Type", "params": {}},
        }
    }

    Path(out_path).write_text(json.dumps(mapping, indent=2))
    return mapping

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recommend_sv(args.profile, args.out)
    print(f"Wrote SV mapping to {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()