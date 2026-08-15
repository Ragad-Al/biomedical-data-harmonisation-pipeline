#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import os
import sys
from datetime import datetime
import json
import logging
import traceback
import glob
import math

'''
PYTHONPATH=. python onboarding/methylation.py TCGA-ACC.methylation450.tsv HM450.hg38.manifest.gencode.v36.tsv \
  --sample-folder tcga_acc1 \
  --mapping onboarding/ACC_METH_mapping.json

'''

class SmartError(Exception):
    """Custom exception for SmartWay errors."""
    pass

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)  # Suppress FutureWarnings

# --- New imports & logging config ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def stop_smartway(message, exit_code=1, exc=None):
    """
    Centralized stop routine: logs error, prints a clear message, optionally prints
    a traceback for debugging, and exits with non-zero status.
    This mimics "stopping in a SmartWay".
    """
    logging.error("SMARTWAY STOP: %s", message)
    print("\n=== SMARTWAY STOP ===")
    print("ERROR:", message)
    if exc is not None:
        print("\n--- Exception Traceback (for debugging) ---")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stdout)
    print("=====================\n")
    sys.exit(exit_code)

# Make pandas show all columns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# -----------------------------
# 1. Methylation schema
# -----------------------------
METHYLATION_COLUMNS = [
    "Sample_ID",
    "CpG_ID",
    "Modification_Type",
    "methylation_caller_software",
    "Chromosome",
    "Position_Start",
    "Position_End",
    "Gene_Symbol",
    "Region_Context",
    "Beta_Value",
    "Adjusted_P_Value",
    "Expression_Status"
]

# -----------------------------
# Column lookup loader
# -----------------------------
# Validate JSON files exist and load
def load_json_safe(path, expect_type=None, varname=""):
    if not os.path.exists(path):
        stop_smartway(f"Required JSON file not found: {path}")
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        stop_smartway(f"Failed to load JSON file {path}: {e}", exc=e)
    if expect_type is not None and not isinstance(data, expect_type):
        stop_smartway(f"JSON file {path} did not contain expected type {expect_type.__name__}. Got {type(data).__name__}.")
    logging.info("Loaded JSON %s from %s", varname or path, path)
    return data

try:
    COLUMN_LOOKUP = load_json_safe("column_name_lookup.json", expect_type=dict, varname="COLUMN_LOOKUP")
except SystemExit:
    raise
except Exception as e:
    stop_smartway("Unexpected error loading column_name_lookup.json", exc=e)
#========================================
try:
    from onboarding.transform_library import TEMPLATES
except Exception:
    from transform_library import TEMPLATES
# -----------------------------
# Load  possible modifications
# -----------------------------
try:
    mods = load_json_safe("modification_type.json", expect_type=list, varname="MODIFICATION_TYPES")
    MODIFICATION_TYPES = set(mods)
except SystemExit:
    raise
except Exception as e:
    stop_smartway("Unexpected error loading modification_type.json", exc=e)

#--------------------------------

# -----------------------------
# Load possible methylation callers
# -----------------------------
try:
    callers = load_json_safe("methylation_caller_software.json", expect_type=list, varname="METHYLATION_CALLERS")
    METHYLATION_CALLERS = set(callers)
except SystemExit:
    raise
except Exception as e:
    stop_smartway("Unexpected error loading methylation_caller_software.json", exc=e)

#--------------------------------
def load_any_table(file_path, nrows=None, chunksize=None, low_memory=False):
    """
    Load CSV, TSV, or XLSX file into a pandas DataFrame or iterator.
    Automatically detects file extension and sets appropriate reader.
    """
    ext = os.path.splitext(file_path.lower())[1]

    if ext in [".csv"]:
        return pd.read_csv(file_path, sep=",", nrows=nrows, chunksize=chunksize, low_memory=low_memory)
    elif ext in [".tsv", ".txt"]:
        return pd.read_csv(file_path, sep="\t", nrows=nrows, chunksize=chunksize, low_memory=low_memory)
    elif ext in [".xlsx", ".xls"]:
        if chunksize is not None:
            raise SmartError("Chunked reading not supported for Excel files. Use CSV/TSV for large datasets.")
        return pd.read_excel(file_path, nrows=nrows)
    else:
        stop_smartway(f"Unsupported file type: {ext}. Supported: .csv, .tsv, .xlsx")

#-----------------------------------
def resolve_sample_csv(sample_folder: str) -> str:
    """Return the newest sample_output_*.csv from sample_folder or exit."""
    folder = os.path.abspath(os.path.expanduser(sample_folder))
    if not os.path.isdir(folder):
        stop_smartway(f"Invalid sample folder: {sample_folder}")
    matches = glob.glob(os.path.join(folder, "sample_output_*.csv"))
    if not matches:
        stop_smartway(
            f"No sample_output_*.csv found in {folder}. "
            "Run sample.py first and point --sample-folder to that folder."
        )
    return max(matches, key=os.path.getmtime)

#-----------------------------------

def normalize_column(series, valid_values, default_val=None):
    """
    Generic normalization for any column against a set of valid values.
    - Converts to string
    - Strips whitespace
    - Matches case-insensitively to known valid values
    - Replaces unknowns with default_val (None if not specified)
    
    Args:
        series (pd.Series or None): input column
        valid_values (set): canonical values loaded from JSON
        default_val (str or None): value to use if unknown
    
    Returns:
        pd.Series: normalized column
    """
    if series is None:
        return None

    series = series.astype(str).str.strip()
    lookup = {v.lower(): v for v in valid_values}

    def map_value(val):
        return lookup.get(val.lower(), default_val)

    return series.map(map_value)

#---------------------------------

def normalize_modification_type(series, valid_mods):
    """
    Normalize a Modification_Type column against the set of valid modifications.
    - Converts to string
    - Strips whitespace
    - Matches case-insensitively to known mods from JSON
    - Replaces unknowns with 'mod'
    
    Args:
        series (pd.Series or None): input modification column
        valid_mods (set): canonical modification types loaded from JSON
    """
    if series is None:
        return None

    # Ensure strings + strip spaces
    series = series.astype(str).str.strip()

    # Build a lowercase lookup dict from valid_mods
    lookup = {m.lower(): m for m in valid_mods}

    # Map each value (case-insensitive)
    def map_value(val):
        key = val.lower()
        return lookup.get(key)   

    return series.map(map_value)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------


def standardize_columns(df):
    """
    Rename dataframe columns to standardized schema names using COLUMN_LOOKUP.
    """
    rename_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    for target_col, aliases in COLUMN_LOOKUP.items():
        for alias in aliases:
            if alias.lower() in lower_cols:
                rename_map[lower_cols[alias.lower()]] = target_col
                break  # stop after first match

    return df.rename(columns=rename_map)

#------------------------------

def get_column(df, target_col):
    """
    Retrieve the column from df based on COLUMN_LOOKUP (case-insensitive) 
    and content heuristics. Always double-check that the candidate column matches expected content.
    Returns the Series if found, else None.
    """

    # Build lowercase mapping of columns
    lower_cols = {c.lower(): c for c in df.columns}

    # --- Step 1: Check by JSON aliases (case-insensitive) ---
    aliases = COLUMN_LOOKUP.get(target_col, [])
    name_candidate = None
    for alias in aliases:
        if alias.lower() in lower_cols:
            name_candidate = df[lower_cols[alias.lower()]]
            break

    # --- Step 1b: Check by direct column name (case-insensitive) ---
    if target_col.lower() in lower_cols:
        name_candidate = df[lower_cols[target_col.lower()]]

    # --- Step 2: Content heuristics if no name match ---
    content_candidate = None
    if name_candidate is None:
        for c in df.columns:
            series = df[c].dropna().astype(str).head(200)

            if target_col == "Chromosome":
                # Accept chr1–chr22, chrX, chrY, chrM
                if series.str.match(r"^(chr)?[0-9XYM]+$", case=False).mean() > 0.8:
                    content_candidate = df[c]
                    break

            elif target_col in ["Position_Start", "Position_End"]:
                numeric_candidates = []
                for cc in df.columns:
                    s = df[cc].dropna().astype(str).head(200)
                    if s.str.match(r"^\d+$").mean() > 0.8:
                        nums = pd.to_numeric(s, errors="coerce")
                        if nums.between(1, 3e9).mean() > 0.8:
                            numeric_candidates.append(cc)
                if numeric_candidates:
                    # Decide start vs end by average value
                    avg_values = {col: pd.to_numeric(df[col], errors="coerce").mean() for col in numeric_candidates}
                    sorted_cols = sorted(avg_values.items(), key=lambda x: x[1])
                    if target_col == "Position_Start":
                        content_candidate = df[sorted_cols[0][0]]
                    else:  # Position_End
                        content_candidate = df[sorted_cols[-1][0]]
                    break

            elif target_col == "Gene_Symbol":
                if series.str.isnumeric().mean() < 0.5 and series.str.match(r"^[A-Z0-9\-]{2,10}$").mean() > 0.5:
                    if not series.str.match(r"^cg\d{6,8}$", case=False).mean() > 0.5:
                        content_candidate = df[c]
                        break

            elif target_col == "Region_Context":
                if series.isin(["+", "-", ".", "NA"]).mean() > 0.5 or \
                   series.str.contains(r"promoter|exon|intron|utr|body|island", case=False).mean() > 0.3:
                    content_candidate = df[c]
                    break

            elif target_col == "Modification_Type":
                # Build regex from JSON list
                mod_pattern = "|".join(re.escape(m) for m in MODIFICATION_TYPES)
                if series.str.contains(mod_pattern, case=False).mean() > 0.3:
                    content_candidate = df[c]
                    break

            elif target_col == "methylation_caller_software":
                if series.str.contains(r"bismark|bs-seeker|methylkit|methylpy|bwa-meth|methylDackel", case=False).mean() > 0.3:
                    content_candidate = df[c]
                    break

    # --- Step 3: Resolve conflicts ---
    if name_candidate is not None and content_candidate is not None:
        if name_candidate.equals(content_candidate):
            return name_candidate
        else:
            print(f"⚠️ Conflict for {target_col}: name vs content mismatch. Using name-based match.")
            return name_candidate

    if name_candidate is not None:
        return name_candidate
    if content_candidate is not None:
        return content_candidate

    return None
#==================================

def apply_transforms(df: pd.DataFrame, transforms: dict) -> pd.DataFrame:
    if not transforms:
        return df

    def snake_to_legacy(s: str) -> str:
        return "_".join([p.capitalize() for p in s.split("_")])

    for target_field, spec in transforms.items():
        fn = TEMPLATES.get(spec.get("template"))
        if not fn:
            continue

        inputs = spec.get("inputs")
        if inputs is None:
            inputs = target_field if target_field in df.columns else snake_to_legacy(target_field)

        params = spec.get("params", {}) or {}
        df[target_field] = fn(df, inputs, params)

    # fold helper clamp into beta_value
    if "beta_value__clamp" in df.columns:
        df["beta_value"] = df["beta_value__clamp"]
        df.drop(columns=["beta_value__clamp"], inplace=True)

    return df

import re

def _to_snake_case(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return s.replace("__", "_").lower()

# -----------------------------
# 2. Detection helpers
# -----------------------------
def detect_cpg_id(df):
    """
    Detect CpG_ID column using COLUMN_LOOKUP and column content inspection.
    Returns the standardized CpG_ID column name if found.
    """

    # --- Step 1: Already standardized?
    if "CpG_ID" in df.columns:
        return "CpG_ID"

    # --- Step 2: Look for CpG_ID aliases in COLUMN_LOOKUP
    aliases = COLUMN_LOOKUP.get("CpG_ID", [])
    lower_cols = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_cols:
            return lower_cols[alias.lower()]

    # --- Step 3: Column content inspection
    regex_cpg = re.compile(r"^cg\d{6,8}$", re.I)  # Typical CpG format

    for c in df.columns:
        sample_vals = df[c].dropna().astype(str).head(500)
        match_count = sum(bool(regex_cpg.match(v.strip())) for v in sample_vals)
        match_ratio = match_count / max(len(sample_vals), 1)

        if match_ratio > 0.8:  # If >80% of samples match CpG pattern
            return c

    # --- Step 4: Fallback — first column
    return df.columns[0]

def normalize_manifest(df):
    """
    Normalize annotation/manifest file into standardized schema.
    Uses COLUMN_LOOKUP (name aliases) + content heuristics (get_column).
    Guarantees both Position_Start and Position_End exist in the output.
    """

    used = set()

    # --- Detect CpG column ---
    cpg_col = detect_cpg_id(df)
    df[cpg_col] = df[cpg_col].astype(str).str.strip().str.upper()
    data = {"CpG_ID": df[cpg_col]}
    used.add(cpg_col)

    # --- Detect other schema columns except positions ---
    for col in METHYLATION_COLUMNS:
        if col in ["CpG_ID", "Position_Start", "Position_End"]:
            continue
        detected = get_column(df, col)
        if detected is not None:
            data[col] = detected
            if isinstance(detected, pd.Series):
                used.add(detected.name)
        else:
            data[col] = None

    # --- Handle Position_Start and Position_End ---
    pos_start = get_column(df, "Position_Start")
    pos_end = get_column(df, "Position_End")

    if pos_start is not None and pos_end is not None:
        data["Position_Start"] = pos_start
        data["Position_End"] = pos_end
    elif pos_start is not None:
        data["Position_Start"] = pos_start
        data["Position_End"] = pos_start
    elif pos_end is not None:
        data["Position_Start"] = pos_end
        data["Position_End"] = pos_end
    else:
        data["Position_Start"] = None
        data["Position_End"] = None


    # ---  Normalize Modification_Type  ---
    if "Modification_Type" in data and data["Modification_Type"] is not None:
        data["Modification_Type"] = normalize_modification_type(data["Modification_Type"], MODIFICATION_TYPES)
    else:
        # Ensure column exists but is empty if not detected
        data["Modification_Type"] = pd.Series([pd.NA] * len(df))


    # ---  Normalize methylation_caller_software  ---
    if "methylation_caller_software" in data and data["methylation_caller_software"] is not None:
        data["methylation_caller_software"] = normalize_column(data["methylation_caller_software"], METHYLATION_CALLERS)
    else:
        # Ensure column exists but is empty if not detected
        data["methylation_caller_software"] = pd.Series([pd.NA] * len(df))



    # --- Build standardized DataFrame ---
    df_norm = pd.DataFrame(data, columns=METHYLATION_COLUMNS)

    return df_norm




def is_wide_format(df):
    """Check if file is wide format (CpGs as rows, samples as columns)."""
    if "Sample_ID" in df.columns and "Beta_Value" in df.columns:
        return False  # long format
    return True

# -----------------------------
# 3. Wide-format mapping
# -----------------------------
def map_methylation_batchwise(df_batch, log_detection=False):
    """Convert wide-format chunk to long-format standardized schema."""
     # ✅ ADD: standardize column names using JSON lookup
    df_batch = standardize_columns(df_batch)
    
    cpg_col = detect_cpg_id(df_batch)
    if log_detection:
        print("Detected CpG column:", cpg_col)

    # ✅ ADD: detect Gene_Symbol column (by JSON alias / heuristics)
    gene_series = get_column(df_batch, "Gene_Symbol")
    gene_col = gene_series.name if isinstance(gene_series, pd.Series) else None

    id_vars = [cpg_col]
    if gene_col and gene_col in df_batch.columns:
        id_vars.append(gene_col)
    value_vars = [c for c in df_batch.columns if c not in id_vars]

    df_long = pd.melt(
        df_batch,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="Sample_ID",
        value_name="Beta_Value"
    )

    mapped = pd.DataFrame({
        "Sample_ID": df_long["Sample_ID"],
        "CpG_ID": df_long[cpg_col],
        "Modification_Type": None,
        "methylation_caller_software": None,
        "Chromosome": None,
        "Position_Start": None,
        "Position_End": None,
        "Gene_Symbol": df_long["Gene_Symbol"] if "Gene_Symbol" in df_long.columns else None,
        "Region_Context": None,
        "Beta_Value": df_long["Beta_Value"],
        "Adjusted_P_Value": None,
        "Expression_Status": None
        
    }, columns=METHYLATION_COLUMNS)

    return mapped

# -----------------------------
# 4. Long-format mapping
# -----------------------------
def map_long_format(df):
    """Normalize long-format input into standardized schema."""
    mapped = pd.DataFrame({
        "Sample_ID": df.get("Sample_ID"),
        "CpG_ID": df.get("CpG_ID"),
        "Modification_Type": normalize_modification_type(df.get("Modification_Type"), MODIFICATION_TYPES),
        "methylation_caller_software": normalize_column(df.get("methylation_caller_software"), METHYLATION_CALLERS),
        "Chromosome": df.get("Chromosome"),
        "Position_Start": df.get("Position_Start"),
        "Position_End": df.get("Position_End"),
        "Gene_Symbol": df.get("Gene_Symbol"),
        "Region_Context": df.get("Region_Context"),
        "Beta_Value": df.get("Beta_Value"),
        "Adjusted_P_Value": df.get("Adjusted_P_Value"),
        "Expression_Status": df.get("Expression_Status")
    }, columns=METHYLATION_COLUMNS)

    return mapped

# -----------------------------
# 5. File ingestion
# -----------------------------
def process_one_file(file_path, batch_size=1000):
    """Ingest a single methylation file into standardized format."""
    if not os.path.exists(file_path):
        stop_smartway(f"Input file not found: {file_path}")

    delim = "\t" if file_path.endswith(".tsv") else ","
    try:
        first_chunk = load_any_table(file_path, nrows=100)
    except pd.errors.EmptyDataError as e:
        stop_smartway(f"File {file_path} appears empty or unreadable: {e}", exc=e)
    except pd.errors.ParserError as e:
        stop_smartway(f"Parsing error reading {file_path}: {e}", exc=e)
    except Exception as e:
        stop_smartway(f"Unexpected error reading {file_path}: {e}", exc=e)

    if first_chunk is None or first_chunk.shape[0] == 0:
        stop_smartway(f"No data found in first 100 rows of {file_path} — check delimiter and file content.")

    first_chunk = standardize_columns(first_chunk)

    # Validate that detect_cpg_id will return a valid column
    try:
        candidate_cpg = detect_cpg_id(first_chunk)
    except Exception as e:
        stop_smartway(f"detect_cpg_id failed on {file_path}: {e}", exc=e)

    if candidate_cpg not in first_chunk.columns:
        # It's possible detect_cpg_id returned a fallback value (first column) but it should be present
        stop_smartway(f"Detected CpG column '{candidate_cpg}' not present in dataframe columns for {file_path}.")

    if is_wide_format(first_chunk):
        # Process in chunks
        try:
            reader = load_any_table(file_path, chunksize=batch_size, low_memory=False)
        except Exception as e:
            stop_smartway(f"Failed to create CSV chunk reader for {file_path}: {e}", exc=e)

        all_chunks = []
        for i, chunk in enumerate(reader):
            if chunk is None or chunk.shape[0] == 0:
                logging.warning("Skipping empty chunk %d for file %s", i, file_path)
                continue
            try:
                mapped_chunk = map_methylation_batchwise(chunk, log_detection=(i == 0))
            except Exception as e:
                stop_smartway(f"Error mapping wide-format chunk {i} of {file_path}: {e}", exc=e)
            # Basic validation on chunk
            if not set(["Sample_ID", "CpG_ID", "Beta_Value"]).issubset(mapped_chunk.columns):
                stop_smartway(f"Mapped chunk missing required columns after melt for {file_path}.")
            all_chunks.append(mapped_chunk)
        if len(all_chunks) == 0:
            stop_smartway(f"No valid chunks were mapped from {file_path}.")
        try:
            mapped_df = pd.concat(all_chunks, ignore_index=True)
        except Exception as e:
            stop_smartway(f"Failed to concatenate mapped chunks for {file_path}: {e}", exc=e)
    else:
        try:
            df = load_any_table(file_path, low_memory=False)
        except Exception as e:
            stop_smartway(f"Failed to read long-format file {file_path}: {e}", exc=e)
        if df is None or df.shape[0] == 0:
            stop_smartway(f"No rows found in long-format file {file_path}.")
        df = standardize_columns(df)
        try:
            mapped_df = map_long_format(df)
        except Exception as e:
            stop_smartway(f"Error mapping long-format file {file_path}: {e}", exc=e)

    # Ensure all schema columns exist
    for col in METHYLATION_COLUMNS:
        if col not in mapped_df.columns:
            mapped_df[col] = None

    # Reorder columns into schema
    mapped_df = mapped_df[METHYLATION_COLUMNS]

    # Basic sanity checks on result
    if mapped_df.shape[0] == 0:
        stop_smartway(f"Mapped dataframe from {file_path} is empty after processing.")
    if mapped_df["CpG_ID"].isnull().all():
        stop_smartway(f"All CpG_ID values are null/empty after processing {file_path} — aborting.")

    logging.info("Ingested %d rows from %s", len(mapped_df), file_path)
    print(f"Ingested {len(mapped_df)} rows from {file_path}")
    return mapped_df

########################################################
def _determine_status(row, hyper_thresh=0.7, hypo_thresh=0.3):
    """
    Derive methylation Expression_Status from Beta_Value.

    - Beta_Value >= hyper_thresh  -> "Hypermethylated"
    - Beta_Value <= hypo_thresh   -> "Hypomethylated"
    - Otherwise                   -> "Intermediate"
    """
    import math

    beta_val = row.get("Beta_Value")

    def _to_float(val):
        if val in ["", "NA", "NaN", None]:
            return None
        try:
            return float(str(val).strip())
        except (ValueError, TypeError):
            return None

    beta = _to_float(beta_val)

    # If we don't have a valid numeric beta, leave status empty
    if beta is None or (isinstance(beta, float) and math.isnan(beta)):
        return None

    if beta >= hyper_thresh:
        return "Hypermethylated"
    elif beta <= hypo_thresh:
        return "Hypomethylated"
    else:
        return "Intermediate"

#---------------------------------------------
def iter_methylation_chunks(file_path, batch_size=1000):
    """
    Yield standardized methylation chunks (DataFrames with METHYLATION_COLUMNS)
    from a potentially very large file, without loading everything into memory.

    This mirrors the logic of process_one_file, but yields one chunk at a time.
    """
    if not os.path.exists(file_path):
        stop_smartway(f"Input file not found: {file_path}")

    # Peek at first rows to decide wide vs long
    try:
        first_chunk = load_any_table(file_path, nrows=100)
    except pd.errors.EmptyDataError as e:
        stop_smartway(f"File {file_path} appears empty or unreadable: {e}", exc=e)
    except pd.errors.ParserError as e:
        stop_smartway(f"Parsing error reading {file_path}: {e}", exc=e)
    except Exception as e:
        stop_smartway(f"Unexpected error reading {file_path}: {e}", exc=e)

    if first_chunk is None or first_chunk.shape[0] == 0:
        stop_smartway(
            f"No data found in first 100 rows of {file_path} — check delimiter and file content."
        )

    # Standardize first_chunk for detection / validation
    first_chunk = standardize_columns(first_chunk)

    # Validate CpG detection
    try:
        candidate_cpg = detect_cpg_id(first_chunk)
    except Exception as e:
        stop_smartway(f"detect_cpg_id failed on {file_path}: {e}", exc=e)

    if candidate_cpg not in first_chunk.columns:
        stop_smartway(
            f"Detected CpG column '{candidate_cpg}' not present in dataframe columns for {file_path}."
        )

    if is_wide_format(first_chunk):
        # WIDE FORMAT: CpGs as rows, samples as columns → melt each chunk
        try:
            reader = load_any_table(file_path, chunksize=batch_size, low_memory=False)
        except Exception as e:
            stop_smartway(f"Failed to create chunked reader for {file_path}: {e}", exc=e)

        for i, chunk in enumerate(reader):
            if chunk is None or chunk.shape[0] == 0:
                continue

            try:
                mapped_chunk = map_methylation_batchwise(
                    chunk,
                    log_detection=(i == 0)
                )
            except Exception as e:
                stop_smartway(
                    f"Error mapping wide-format chunk {i} of {file_path}: {e}",
                    exc=e
                )

            # Basic validation
            required_cols = {"Sample_ID", "CpG_ID", "Beta_Value"}
            if not required_cols.issubset(mapped_chunk.columns):
                stop_smartway(
                    f"Mapped chunk {i} missing required columns after melt for {file_path}."
                )

            # Ensure all schema columns exist and reorder
            for col in METHYLATION_COLUMNS:
                if col not in mapped_chunk.columns:
                    mapped_chunk[col] = None
            mapped_chunk = mapped_chunk[METHYLATION_COLUMNS].copy()

            yield mapped_chunk

    else:
        # LONG FORMAT: already one row per (sample, CpG)
        try:
            reader = load_any_table(file_path, chunksize=batch_size, low_memory=False)
        except Exception as e:
            stop_smartway(f"Failed to read long-format file {file_path}: {e}", exc=e)

        for i, df in enumerate(reader):
            if df is None or df.shape[0] == 0:
                continue

            df = standardize_columns(df)
            try:
                mapped_chunk = map_long_format(df)
            except Exception as e:
                stop_smartway(
                    f"Error mapping long-format chunk {i} of {file_path}: {e}",
                    exc=e
                )

            for col in METHYLATION_COLUMNS:
                if col not in mapped_chunk.columns:
                    mapped_chunk[col] = None
            mapped_chunk = mapped_chunk[METHYLATION_COLUMNS].copy()

            yield mapped_chunk

#---------------------------------------------

# -----------------------------
# 6. Ingestion workflow (1 or 2 files)
# -----------------------------
def ingest_methylation_files(file_paths, batch_size=1000, sample_folder=None, methylation_output=None, mapping_path=None):
     
    if len(file_paths) == 0:
        stop_smartway("No input files provided to ingest_methylation_files")

    # Validate input file existence early
    for p in file_paths:
        if not os.path.exists(p):
            stop_smartway(f"Input file does not exist: {p}")

    # --- Figure out which is methylation vs manifest (if two files) ---
    has_manifest = False
    methyl_file = None
    manifest_file = None

    if len(file_paths) == 2:
        f1, f2 = file_paths
        f1_is_methyl = any(x in f1.lower() for x in ["methyl"])
        f2_is_methyl = any(x in f2.lower() for x in ["methyl"])

        # ❌ If user reversed them, stop immediately
        if not f1_is_methyl and f2_is_methyl:
            stop_smartway(
                "Wrong file order: manifest file was provided first. "
                "Please pass the methylation file first, then the manifest."
            )
        elif not f1_is_methyl and not f2_is_methyl:
            stop_smartway(
                "Could not detect a methylation file among inputs — please provide it first."
            )

        methyl_file = f1
        manifest_file = f2
        has_manifest = True

    elif len(file_paths) == 1:
        methyl_file = file_paths[0]
    else:
        stop_smartway(
            f"Expected 1 or 2 input files (methylation[, manifest]), got {len(file_paths)}."
        )

    # --- Sample folder handling (non-interactive) ---
    if sample_folder is None:
        stop_smartway("--sample-folder is required in non-interactive mode.")
    sample_csv = resolve_sample_csv(sample_folder)

    try:
        sample_table = pd.read_csv(sample_csv)
    except Exception as e:
        stop_smartway(f"Failed to read sample table {sample_csv}: {e}", exc=e)

    # Normalize external_sample_id for joining
    sample_table["external_sample_id"] = (
        sample_table["external_sample_id"].astype(str).str.strip().str.lower()
    )
    print(f"✅ Loaded {len(sample_table)} samples from: {os.path.basename(sample_csv)}")

    # Precompute normalized external IDs
    sample_table = sample_table.copy()
    sample_table["external_sample_id_norm"] = (
        sample_table["external_sample_id"].astype(str).str.strip().str.lower()
    )


    #=======
    mapping_json = None
    if mapping_path:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_json = json.load(f)

    #========
    # --- Load and normalize manifest (if present) ---
    df_manifest = None
    if has_manifest:
        if not os.path.exists(manifest_file):
            stop_smartway(f"Manifest file not found: {manifest_file}")

        try:
            df_manifest = load_any_table(manifest_file, low_memory=False)
        except Exception as e:
            stop_smartway(f"Failed to read manifest file {manifest_file}: {e}", exc=e)

        if df_manifest is None or df_manifest.shape[0] == 0:
            stop_smartway(f"Manifest file {manifest_file} is empty or unreadable.")

        # Standardize and normalize manifest
        df_manifest = df_manifest.copy()
        # If mapping exists, select manifest columns using mapping JSON sources
        if mapping_json and mapping_json.get("manifest_file") and mapping_json.get("mappings"):
            # Strip leading '#' in first column name if needed
            if len(df_manifest.columns) > 0 and str(df_manifest.columns[0]).startswith("#"):
                df_manifest.rename(columns={df_manifest.columns[0]: str(df_manifest.columns[0]).lstrip("#")}, inplace=True)
                
            m = mapping_json["mappings"]

            # CpG id column in manifest
            # Manifest uses "id" after stripping '#'
            manifest_id_col = "id" if "id" in df_manifest.columns else df_manifest.columns[0]
            df_manifest["CpG_ID"] = df_manifest[manifest_id_col].astype(str).str.strip().str.upper()

            # Pull only the mapped annotation columns if they exist
            def _get_manifest_col(target_key, out_name):
                spec = m.get(target_key, {})
                if spec.get("mode") != "map":
                    return None
                col = spec.get("source", {}).get("column")
                if col and col in df_manifest.columns:
                    return df_manifest[col]
                return None

            chrom = _get_manifest_col("chromosome", "Chromosome")
            start = _get_manifest_col("position_start", "Position_Start")
            end = _get_manifest_col("position_end", "Position_End")
            gene = _get_manifest_col("gene_symbol", "Gene_Symbol")
            region = _get_manifest_col("region_context", "Region_Context")

            df_manifest = pd.DataFrame({
                "CpG_ID": df_manifest["CpG_ID"],
                "Chromosome": chrom,
                "Position_Start": start,
                "Position_End": end,
                "Gene_Symbol": gene,
                "Region_Context": region
            })
        else:
            # Old behavior (no mapping): use your heuristics
            df_manifest = standardize_columns(df_manifest)
            df_manifest = normalize_manifest(df_manifest)

        # Normalize CpG_ID in manifest
        try:
            df_manifest = df_manifest.copy()
            df_manifest["CpG_ID"] = (
                df_manifest["CpG_ID"].astype(str).str.strip().str.upper()
            )
        except Exception as e:
            stop_smartway(f"Failed to normalize CpG_ID in manifest: {e}", exc=e)

    # --- Prepare output file path ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if methylation_output:
        output_file = methylation_output
    else:
        output_file = os.path.join(sample_folder, f"methylation_output_{timestamp}.csv")

    # --- Stream methylation data in chunks ---
    first = True
    total_rows_written = 0

    for chunk in iter_methylation_chunks(methyl_file, batch_size=batch_size):
        if chunk is None or chunk.shape[0] == 0:
            continue

        # Normalize CpG_ID in the chunk
        try:
            chunk.loc[:, "CpG_ID"] = (
                chunk["CpG_ID"].astype(str).str.strip().str.upper()
            )
        except Exception as e:
            stop_smartway(f"Failed to normalize CpG_ID column in methylation data: {e}", exc=e)

        # --- Merge with manifest, if provided ---
        if df_manifest is not None:
            try:
                merged = pd.merge(
                    chunk,
                    df_manifest,
                    on="CpG_ID",
                    how="left",
                    suffixes=("", "_manifest")
                )
            except Exception as e:
                stop_smartway(
                    f"Failed to merge methylation data chunk with manifest: {e}",
                    exc=e
                )

            # Prefer methylation values for overlapping columns
            for col in METHYLATION_COLUMNS:
                if col in ["Sample_ID", "CpG_ID", "Beta_Value"]:
                    continue
                manifest_col = f"{col}_manifest"
                if col in merged.columns and manifest_col in merged.columns:
                    try:
                        merged[col] = merged[col].fillna(merged[manifest_col])
                        merged.drop(columns=[manifest_col], inplace=True)
                    except Exception as e:
                        stop_smartway(
                            f"Failed to merge/cleanup manifest column '{manifest_col}': {e}",
                            exc=e
                        )

            chunk = merged

        # --- Enforce schema columns and order ---
        for col in METHYLATION_COLUMNS:
            if col not in chunk.columns:
                chunk[col] = None
        chunk = chunk[METHYLATION_COLUMNS].copy()

        # --- Compute Expression_Status if empty in this chunk ---
        if chunk["Expression_Status"].isna().all():
            chunk["Expression_Status"] = chunk.apply(_determine_status, axis=1)

        chunk = apply_transforms(chunk, mapping_json.get("transforms", {}) if mapping_json else {})

        # --- Join with sample_table ---
        chunk = chunk.copy()
        chunk["Sample_ID_norm"] = (
            chunk["Sample_ID"].astype(str).str.strip().str.lower()
        )

        df_merged = chunk.merge(
            sample_table[["external_sample_id_norm", "sample_id"]],
            left_on="Sample_ID_norm",
            right_on="external_sample_id_norm",
            how="inner"  # keep only rows that have a match
        )

        if df_merged.shape[0] == 0:
            # This chunk had no matching samples; skip it but continue streaming
            continue

        df_merged = df_merged.copy()
        df_merged["Sample_ID"] = df_merged["sample_id"]

        helper_cols = ["Sample_ID_norm", "external_sample_id_norm", "sample_id"]
        df_processed = df_merged.drop(
            columns=[c for c in helper_cols if c in df_merged.columns]
        )

        # --- Append this chunk to disk ---
        mode = "w" if first else "a"
        header = first

        df_output = df_processed.copy()

        # if snake_case columns exist, drop legacy duplicates
        prefer_snake = {
            "chromosome": "Chromosome",
            "position_start": "Position_Start",
            "position_end": "Position_End",
            "beta_value": "Beta_Value",
            "cpg_id": "CpG_ID",
            "gene_symbol": "Gene_Symbol",
            "region_context": "Region_Context",
        }
        for snake, legacy in prefer_snake.items():
            if snake in df_output.columns and legacy in df_output.columns:
                df_output.drop(columns=[legacy], inplace=True)

        # rename all columns to snake_case
        df_output = df_output.rename(columns={c: _to_snake_case(c) for c in df_output.columns})

        # keep canonical methylation columns
        canonical_cols = [
            "sample_id","cpg_id","modification_type","methylation_caller_software",
            "chromosome","position_start","position_end","gene_symbol","region_context",
            "beta_value","adjusted_p_value","expression_status"
        ]
        canonical_cols = [c for c in canonical_cols if c in df_output.columns]
        df_output = df_output[canonical_cols]

        df_output.to_csv(
            output_file,
            index=False,
            encoding="utf-8",
            mode=mode,
            header=header
        )
        total_rows_written += len(df_output)        
        first = False

    # --- Final checks ---
    if total_rows_written == 0:
        stop_smartway("Final merged dataframe is empty. Aborting before saving.")

    logging.info("Ingested %d rows → %s", total_rows_written, output_file)

    # --- Preview output (first 10 rows) ---
    try:
        preview = pd.read_csv(output_file, nrows=10)
        print("\n📊 Preview of methylation output (first 10 rows):")
        with pd.option_context('display.max_columns', None, 'display.width', None):
            print(preview)
    except Exception as e:
        print(f"⚠️ Could not read preview from {output_file}: {e}")

    print(f"\n✅ Methylation data successfully saved to: {output_file}\n")
    return output_file


# ==========================================================
# 🏁 MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    import argparse
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Custom ArgumentParser to override default error behavior
    class SmartArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            if "required: input_files" in message or "too few arguments" in message:
                print("⚠️ No input files.")
                print("   Usage: python methylation.py <input_file1> [<input_file2> ...]")
                sys.exit(1)
            else:
                self.print_help()
                sys.exit(1)

    parser = SmartArgumentParser(
        description="Ingest methylation files into standardized format and link to samples."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Paths to input files (CSV, TSV, XLSX). Example: meth.tsv [manifest.csv]"
    )

    parser.add_argument(
        "--sample-folder",
        required=True,
        help="Folder containing sample_output_*.csv produced by sample.py (newest file will be used)."
    )
    
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path. If omitted, a timestamped file is written under --sample-folder."
    )

    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional mapping JSON (e.g., onboarding/ACC_METH_mapping.json)."
    )

    args = parser.parse_args()

    try:
        out_file = ingest_methylation_files(
            args.input_files,
            sample_folder=args.sample_folder,
            methylation_output=args.output,
            mapping_path=args.mapping
        )

        logging.info(f"✅ Methylation data successfully saved to {out_file}")
    except SmartError as e:
        logging.error(f"🛑 SMARTWAY FATAL HALT: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"🛑 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc(limit=3)
        sys.exit(1)

