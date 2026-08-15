# onboarding/profiler.py

# 1 file 
# python onboarding/profiler.py --file data_mutations.tsv --rows 500 --out mutation_profile.json

# 2 files
# python onboarding/profiler.py --file TCGA-ACC.clinical.tsv TCGA-ACC.survival.tsv --rows 200 --out onboarding/ACC_Entity_profile.json

'''
python onboarding/profiler.py \
  --file TCGA-ACC.somaticmutation_wxs.tsv \
  --rows 200 \
  --out onboarding/ACC_Mutation_profile.json


python onboarding/profiler.py \
  --file pcawg_consensus_1.6.161116.somatic_svs.xena.donor.tsv \
  --rows 200 \
  --out onboarding/pcawg_donor_SV_profile.json


  python onboarding/profiler.py \
  --file TCGA-ACC.star_tpm.tsv \
  --rows 200 \
  --out onboarding/ACC_RNASEQ_profile.json
  

python onboarding/profiler.py \
  --file TCGA-ACC.methylation450.tsv HM450.hg38.manifest.gencode.v36.tsv \
  --rows 200 \
  --out onboarding/ACC_METH_profile.json
  
'''


from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd


# -------------------------
# Delimiter detection + header normalization
# -------------------------

def sniff_delimiter(file_path: str, default: str = "\t") -> str:
    """
    Robust delimiter detection:
    - Read first non-empty line and pick delimiter with highest count.
    """
    candidates = ["\t", ",", ";", "|"]
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # Skip metadata lines starting with '#', BUT keep real headers like '#id\tgene\tchrom...'
            if (line.lstrip().startswith("#")
                    and not line.lower().startswith("#id\t")
                    and not line.lower().startswith("#id,")):
                continue
            counts = {d: line.count(d) for d in candidates}
            best = max(counts, key=counts.get)
            return best if counts[best] > 0 else default
    return default


def normalize_header(col: str) -> Dict[str, Any]:
    raw = str(col)
    s = raw.strip()
    s_lower = s.lower()
    tokens = re.split(r"[\s_\-]+", s_lower)
    tokens = [t for t in tokens if t]
    return {"raw": raw, "lower": s_lower, "tokens": tokens}


# -------------------------
# Pattern detectors
# -------------------------

RE_TCGA = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}.*$", re.IGNORECASE)
RE_CHR = re.compile(r"^(chr)?([1-9]|1[0-9]|2[0-2]|x|y|m|mt)$", re.IGNORECASE)
RE_ENSG = re.compile(r"^ENSG\d{6,}$", re.IGNORECASE)
RE_ENST = re.compile(r"^ENST\d{6,}$", re.IGNORECASE)
RE_UNIPROT = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
RE_ALLELE = re.compile(r"^([ACGTN]|-)$", re.IGNORECASE)

RE_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RE_ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$")
RE_SLASH_DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
RE_TIME = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")
RE_EPOCH_SEC = re.compile(r"^\d{10}$")
RE_EPOCH_MS = re.compile(r"^\d{13}$")


def _frac_match(vals: List[str], regex: re.Pattern) -> float:
    if not vals:
        return 0.0
    return sum(bool(regex.match(v)) for v in vals) / len(vals)


def date_time_flags(series: pd.Series, n_sample: int = 200) -> Dict[str, Any]:
    s = series.dropna()
    if len(s) == 0:
        return {"date_like": False, "datetime_like": False, "time_like": False,
                "epoch_seconds_like": False, "epoch_millis_like": False, "date_support": {}}

    vals = s.astype(str).head(n_sample).tolist()

    f_iso_date = _frac_match(vals, RE_ISO_DATE)
    f_iso_dt = _frac_match(vals, RE_ISO_DATETIME)
    f_slash = _frac_match(vals, RE_SLASH_DATE)
    f_time = _frac_match(vals, RE_TIME)
    f_epoch_s = _frac_match(vals, RE_EPOCH_SEC)
    f_epoch_ms = _frac_match(vals, RE_EPOCH_MS)

    return {
        "date_like": (f_iso_date > 0.5) or (f_slash > 0.5),
        "datetime_like": (f_iso_dt > 0.5),
        "time_like": (f_time > 0.5),
        "epoch_seconds_like": (f_epoch_s > 0.5),
        "epoch_millis_like": (f_epoch_ms > 0.5),
        "date_support": {
            "iso_date_frac": f_iso_date,
            "iso_datetime_frac": f_iso_dt,
            "slash_date_frac": f_slash,
            "time_frac": f_time,
            "epoch_seconds_frac": f_epoch_s,
            "epoch_millis_frac": f_epoch_ms,
            "n_checked": len(vals),
        }
    }


def pattern_flags(series: pd.Series, n_sample: int = 200) -> Dict[str, Any]:
    s = series.dropna()
    if len(s) == 0:
        return {"tcga_like": False, "chromosome_like": False, "ensembl_gene_like": False,
                "ensembl_tx_like": False, "uniprot_like": False, "allele_like": False,
                "pattern_support": {}}

    vals = s.astype(str).head(n_sample).tolist()

    tcga = sum(bool(RE_TCGA.match(v)) for v in vals)
    chr_like = sum(bool(RE_CHR.match(v)) for v in vals)
    ensg = sum(bool(RE_ENSG.match(v)) for v in vals)
    enst = sum(bool(RE_ENST.match(v)) for v in vals)
    unip = sum(bool(RE_UNIPROT.match(v)) for v in vals)
    allele = sum(bool(RE_ALLELE.match(v)) for v in vals)

    denom = max(len(vals), 1)
    return {
        "tcga_like": tcga / denom > 0.5,
        "chromosome_like": chr_like / denom > 0.5,
        "ensembl_gene_like": ensg / denom > 0.5,
        "ensembl_tx_like": enst / denom > 0.5,
        "uniprot_like": unip / denom > 0.5,
        "allele_like": allele / denom > 0.5,
        "pattern_support": {
            "tcga_frac": tcga / denom,
            "chromosome_frac": chr_like / denom,
            "ensembl_gene_frac": ensg / denom,
            "ensembl_tx_frac": enst / denom,
            "uniprot_frac": unip / denom,
            "allele_frac": allele / denom,
            "n_checked": len(vals),
        }
    }


# -------------------------
# Numeric stats + dtype inference
# -------------------------

def numeric_profile(series: pd.Series) -> Dict[str, Any]:
    """
    Compute numeric stats if values are numeric-like.
    Robust to boolean columns (TRUE/FALSE) by forcing float dtype.
    """
    num = pd.to_numeric(series, errors="coerce")

    n = len(series)
    n_num = int(num.notna().sum())
    frac_num = n_num / n if n else 0.0

    if n_num == 0:
        return {
            "numeric_fraction": frac_num,
            "min": None, "max": None, "q05": None, "q50": None, "q95": None,
            "fraction_0_1": None,
            "integer_like_fraction": None,
        }

    # IMPORTANT: ensure numeric operations (quantile) work even if dtype is bool/int
    numf = num.astype("float64")

    q05 = float(numf.quantile(0.05))
    q50 = float(numf.quantile(0.50))
    q95 = float(numf.quantile(0.95))
    minv = float(numf.min())
    maxv = float(numf.max())

    in_0_1 = ((numf >= 0) & (numf <= 1)).mean()
    int_like = ((numf - numf.round()).abs() < 1e-9).mean()

    return {
        "numeric_fraction": frac_num,
        "min": minv,
        "max": maxv,
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "fraction_0_1": float(in_0_1),
        "integer_like_fraction": float(int_like),
    }


def infer_dtype(num_prof: Dict[str, Any], dt_flags: Dict[str, Any]) -> str:
    if dt_flags.get("datetime_like") or dt_flags.get("epoch_seconds_like") or dt_flags.get("epoch_millis_like"):
        return "datetime"
    if dt_flags.get("date_like"):
        return "date"
    if dt_flags.get("time_like"):
        return "time"

    if num_prof["numeric_fraction"] >= 0.95:
        if num_prof["integer_like_fraction"] is not None and num_prof["integer_like_fraction"] >= 0.98:
            return "int"
        return "float"
    return "string"


def string_profile(series: pd.Series, n_sample: int = 200) -> Dict[str, Any]:
    s = series.dropna()
    if len(s) == 0:
        return {"avg_len": None, "min_len": None, "max_len": None, "n_checked": 0}
    vals = s.astype(str).head(n_sample)
    lens = vals.str.len()
    return {"avg_len": float(lens.mean()), "min_len": int(lens.min()), "max_len": int(lens.max()), "n_checked": int(len(vals))}


# -------------------------
# Wide matrix guardrails
# -------------------------

def detect_wide_matrix(df: pd.DataFrame, max_check_cols: int = 50, min_cols: int = 20) -> Dict[str, Any]:
    """
    Detects typical omics wide matrices:
      - first column is feature ID-like (genes/CpGs): high uniqueness or not mostly numeric
      - most other columns are numeric (sample columns)
    Works for small cohorts too (e.g., TCGA-ACC ~80 columns).
    """
    n_cols = df.shape[1]
    if n_cols < min_cols:
        return {"is_wide_matrix": False, "reason": f"n_cols < {min_cols}", "n_cols": int(n_cols)}

    cols = list(df.columns)
    check_cols = cols[:min(n_cols, max_check_cols)]

    # Evaluate numeric-ness of non-first columns (sample columns)
    numeric_fracs = []
    for c in check_cols[1:]:
        try:
            numeric_fracs.append(numeric_profile(df[c])["numeric_fraction"])
        except Exception:
            numeric_fracs.append(0.0)

    if not numeric_fracs:
        return {"is_wide_matrix": False, "reason": "no columns checked", "n_cols": int(n_cols)}

    frac_numeric_cols = sum(f >= 0.95 for f in numeric_fracs) / len(numeric_fracs)

    # Evaluate first column as ID-like (feature id)
    first = df[cols[0]]
    non_null = first.dropna()
    uniq_ratio = float(non_null.nunique() / len(non_null)) if len(non_null) else 0.0
    first_num = numeric_profile(first)["numeric_fraction"]

    # ID-like if mostly unique OR not numeric (e.g., ENSG..., cg...)
    first_is_id_like = (uniq_ratio > 0.7) or (first_num < 0.5)

    # Final decision
    is_matrix = (frac_numeric_cols >= 0.8) and first_is_id_like

    return {
        "is_wide_matrix": bool(is_matrix),
        "n_cols": int(n_cols),
        "min_cols_threshold": int(min_cols),
        "frac_numeric_cols_checked": float(frac_numeric_cols),
        "first_col": str(cols[0]),
        "first_col_uniqueness_ratio": float(uniq_ratio),
        "first_col_numeric_fraction": float(first_num),
        "reason": "mostly numeric sample cols + ID-like first col" if is_matrix else "criteria not met",
    }

def choose_columns_to_profile(columns: List[str], max_cols: int = 200) -> List[str]:
    if len(columns) <= max_cols:
        return columns
    keep = [columns[0]]
    remaining = columns[1:]
    step = max(1, len(remaining) // (max_cols - 1))
    sampled = remaining[::step][: (max_cols - 1)]
    keep.extend(sampled)
    return keep
#------------------------------------------------------
def should_treat_hash_as_comment(file_path: str) -> bool:
    """
    True  -> treat '#' lines as comments (skip cBioPortal-style metadata)
    False -> do NOT treat '#' as comments (e.g., manifests where header starts with '#id')
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # If the header is '#id<TAB>gene<TAB>chrom...' then '#' is part of header, not metadata.
            if line.lower().startswith("#id\t") or line.lower().startswith("#id,"):
                return False

            # Typical metadata style: lots of lines start with '#'
            if line.startswith("#"):
                return True

            # First real line is not '#'
            return False
    return False


def normalize_hash_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a file uses '#id' as the first header name, strip the leading '#'.
    """
    if len(df.columns) > 0 and isinstance(df.columns[0], str) and df.columns[0].startswith("#"):
        df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("#")})
    return df

# -------------------------
# Profiling
# -------------------------

def profile_file(file_path: str, sample_rows: int = 500, delimiter: Optional[str] = None, max_profile_cols: int = 200) -> Dict[str, Any]:
    file_path = str(file_path)
    if delimiter is None:
        delimiter = sniff_delimiter(file_path)

    use_hash_comments = should_treat_hash_as_comment(file_path)

    read_kwargs = dict(
        sep=delimiter,
        nrows=sample_rows,
        engine="python",
        on_bad_lines="skip",
    )

    try:
        df = pd.read_csv(
            file_path,
            **read_kwargs,
            comment="#" if use_hash_comments else None,
            encoding_errors="replace",
        )
    except TypeError:
        df = pd.read_csv(
            file_path,
            **read_kwargs,
            comment="#" if use_hash_comments else None,
        )
        # If header begins with '#id', strip leading '#' from the first column name
        df = normalize_hash_header(df)
    matrix_info = detect_wide_matrix(df)
    cols_to_profile = list(df.columns)
    if matrix_info["is_wide_matrix"]:
        cols_to_profile = choose_columns_to_profile(cols_to_profile, max_cols=max_profile_cols)

    profiles: Dict[str, Any] = {}
    n_rows = len(df)

    for col in cols_to_profile:
        ser = df[col]
        header = normalize_header(col)

        missing_rate = float(ser.isna().mean()) if n_rows else 0.0
        non_null = ser.dropna()
        uniq_ratio = float(non_null.nunique() / len(non_null)) if len(non_null) else 0.0

        num_prof = numeric_profile(ser)
        dt_flags = date_time_flags(ser)
        dtype = infer_dtype(num_prof, dt_flags)

        profiles[str(col)] = {
            "header": header,
            "missing_rate": missing_rate,
            "uniqueness_ratio": uniq_ratio,
            "inferred_dtype": dtype,
            "numeric_profile": num_prof,
            "string_profile": string_profile(ser),
            "patterns": pattern_flags(ser),
            "date_time": dt_flags,
        }

    return {
        "file_path": file_path,
        "delimiter": delimiter,
        "sample_rows_profiled": int(sample_rows),
        "n_rows_read": int(n_rows),
        "n_cols_total": int(df.shape[1]),
        "wide_matrix_detection": matrix_info,
        "n_cols_profiled": int(len(cols_to_profile)),
        "profiled_columns": [str(c) for c in cols_to_profile],
        "columns": profiles,
    }


def profile_files(file_paths: List[str], sample_rows: int = 500, delimiter: Optional[str] = None, max_profile_cols: int = 200) -> Dict[str, Any]:
    bundle = {"version": "1.0", "n_files": len(file_paths), "files": {}}
    for fp in file_paths:
        bundle["files"][fp] = profile_file(fp, sample_rows=sample_rows, delimiter=delimiter, max_profile_cols=max_profile_cols)
    return bundle


def save_profile(profile: Dict[str, Any], out_path: str) -> None:
    Path(out_path).write_text(json.dumps(profile, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, nargs="+", help="One or more CSV/TSV files (same module)")
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--delim", default=None)
    ap.add_argument("--max_cols", type=int, default=200, help="Max columns to profile if wide matrix detected")
    args = ap.parse_args()

    if len(args.file) == 1:
        prof = profile_file(args.file[0], sample_rows=args.rows, delimiter=args.delim, max_profile_cols=args.max_cols)
    else:
        prof = profile_files(args.file, sample_rows=args.rows, delimiter=args.delim, max_profile_cols=args.max_cols)

    if args.out:
        save_profile(prof, args.out)
        print(f"Wrote profile to {args.out}")
    else:
        print(json.dumps(prof, indent=2))