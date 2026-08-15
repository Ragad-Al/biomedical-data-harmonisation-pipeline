#!/usr/bin/env python
# coding: utf-8

'''
python onboarding/sample.py TCGA-ACC.clinical.tsv TCGA-ACC.survival.tsv \
  --entity-folder tcga_acc1 \
  --mapping onboarding/ACC_Sample_mapping.json

'''

import os
import sys
import pandas as pd
import json
import uuid
from datetime import datetime
from pathlib import Path
import re
import numpy as np
import difflib
import traceback
import glob
import argparse

# ==========================================================
# 🧠 SMART ERROR GUARD SYSTEM
# ==========================================================

class SmartError(Exception):
    """Custom fatal error for Smartway pipeline."""
    pass

def smart_check(condition, message):
    """Raise SmartError if condition is False."""
    if not condition:
        raise SmartError(message)

def safe_execute(step_name):
    """Decorator for wrapping critical functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except SmartError:
                raise
            except Exception as e:
                print(f"\n🚨 Smartway HALT: Error in step '{step_name}'")
                print(f"➡️ {type(e).__name__}: {e}")
                print("📄 Traceback:")
                traceback.print_exc(limit=2)
                raise SmartError(f"Fatal error in '{step_name}' — stopping execution.")
        return wrapper
    return decorator


# ==========================================================
# 🚦 SMART VALIDATION HELPERS
# ==========================================================

def validate_file_exists(path, description="file"):
    if not os.path.exists(path):
        raise SmartError(f"❌ Required {description} not found: '{path}'")

def validate_dataframe(df, context="DataFrame"):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise SmartError(f"❌ Invalid or empty {context}.")
    if len(df.columns) == 0:
        raise SmartError(f"❌ {context} has no columns.")

def validate_json(path, description="JSON"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as e:
        raise SmartError(f"❌ {description} file '{path}' is not valid JSON: {e}")

def resolve_entity_csv(entity_folder: str) -> str:
    """
    Resolve and return the newest entity_output_*.csv inside entity_folder.
    Raises SmartError with a clear message if the folder or files are missing.
    """
    # (1) Resolve the folder path and verify it exists
    entity_dir = os.path.abspath(os.path.expanduser(entity_folder))
    if not os.path.isdir(entity_dir):
        raise SmartError(f"Missing entity folder. Provide it with --entity-folder. Got: {entity_folder}")

    # (2) Glob for files and fail clearly if none are found (with debug prints)
    pattern = os.path.join(entity_dir, "*entity_output_*.csv")
    entity_csvs = sorted(glob.glob(pattern))
    print(f"🔎 entity_folder={entity_dir}\n🔎 looking for={pattern}\n🔎 found={len(entity_csvs)}")
    if not entity_csvs:
        raise SmartError(f"No entity_output_*.csv found in {entity_dir}. Run entity.py first.")

    # (3) Pick the newest file
    entity_csv = max(entity_csvs, key=os.path.getmtime)
    return entity_csv


# ------------------ CONFIG ------------------ #
COLUMN_NAME_LOOKUP_PATH = "column_name_lookup.json"
COLUMN_CONTENT_LOOKUP_PATH = "column_content_lookup.json"




SAMPLE_COLUMNS = [
    "sample_id", "entity_id", "external_sample_id", "sample_type", "cell_type",
    "tissue_site", "collection_date", "collection_time", "collection_method",
    "preservation_method", "storage_location", "storage_conditions_json",
    "is_tumor_sample", "tumor_type", "tumor_grade", "tumor_stage",
    "normal_sample_id_paired", "disease_status_at_collection", "viability_percent",
    "concentration_ng_ul", "purity_metrics_json", "study_id",
    "creation_timestamp", "last_updated_timestamp"
]


STORAGE_CONDITIONS_KEYS = ["temperature", "container_type", "storage_medium", "location"]
PURITY_METRICS_KEYS = ["rna_integrity_number", "tumor_purity_estimate", "contamination_percent"]

# ==========================================================
# 🧩 COLUMN TYPE RULES
# ==========================================================
NUMERIC_ONLY_COLUMNS = [
    "collection_date", "collection_time", "temperature", "rna_integrity_number", "tumor_purity_estimate", "contamination_percent"
]

TEXT_ONLY_COLUMNS = [
    "sample_id", "entity_id", "external_sample_id", "sample_type", "cell_type",
    "tissue_site", "collection_method", "preservation_method", "storage_location",
    "tumor_type", "tumor_grade", "tumor_stage", "normal_sample_id_paired", "disease_status_at_collection", "viability_percent",
    "concentration_ng_ul", "container_type", "storage_medium", "location"
]

# ------------------ HELPERS ------------------ #
@safe_execute("load_lookup_tables")
def load_lookup_tables():
    validate_file_exists(COLUMN_NAME_LOOKUP_PATH, "column_name_lookup.json")
    validate_file_exists(COLUMN_CONTENT_LOOKUP_PATH, "column_content_lookup.json")
    validate_json(COLUMN_NAME_LOOKUP_PATH)
    validate_json(COLUMN_CONTENT_LOOKUP_PATH)

    with open(COLUMN_NAME_LOOKUP_PATH) as f:
        name_lookup = json.load(f)
    with open(COLUMN_CONTENT_LOOKUP_PATH) as f:
        content_lookup = json.load(f)

    smart_check(isinstance(name_lookup, dict), "column_name_lookup must be a JSON object.")
    smart_check(isinstance(content_lookup, dict), "column_content_lookup must be a JSON object.")
    return name_lookup, content_lookup
#==================================================
def load_mapping_json(mapping_path):
    if not mapping_path:
        return None
    validate_file_exists(mapping_path, "sample_mapping.json")
    validate_json(mapping_path, "sample_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sample_id_overrides(mapping_json):
    """
    Returns two dicts:
      sample_id_per_file[basename] = source_column_for_external_sample_id
      entity_id_per_file[basename] = source_column_for_external_entity_id
    """
    sample_id_per_file = {}
    entity_id_per_file = {}

    if not mapping_json or "mappings" not in mapping_json:
        return sample_id_per_file, entity_id_per_file

    m = mapping_json["mappings"]

    # external_sample_id
    s = m.get("external_sample_id", {})
    if isinstance(s, dict) and s.get("mode") == "map":
        src = s.get("source", {})
        f = src.get("file")
        c = src.get("column")
        if f and c:
            sample_id_per_file[os.path.basename(f)] = c

    # external_entity_id
    e = m.get("external_entity_id", {})
    if isinstance(e, dict) and e.get("mode") == "map":
        src = e.get("source", {})
        f = src.get("file")
        c = src.get("column")
        if f and c:
            entity_id_per_file[os.path.basename(f)] = c

    return sample_id_per_file, entity_id_per_file
#==================================================
def detect_column_type(df):
    """
    Safely detect column types (numeric / boolean / text).
    Handles duplicate column names and mixed types.
    """
    import pandas as pd

    # Remove duplicate columns if any (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    col_types = {}
    for col in df.columns:
        series = df[col]

        # If somehow it's still a DataFrame (duplicate name), flatten it
        if isinstance(series, pd.DataFrame):
            # Combine all sub-columns into a single string column
            series = series.astype(str).agg(";".join, axis=1)

        if series.empty:
            col_types[col] = "unknown"
            continue

        # Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            col_types[col] = "numeric"

        # Boolean detection
        elif pd.api.types.is_bool_dtype(series):
            col_types[col] = "boolean"
        elif series.astype(str).str.lower().isin(
            ["true", "false", "yes", "no", "t", "f", "y", "n"]
        ).any():
            col_types[col] = "boolean"

        # Default to text
        else:
            col_types[col] = "text"

    return col_types



#===================

def normalize_column_names(df, name_lookup):
    """
    Map input column names to standardized schema using lookup.
    Handles multi-alias lists and performs case-insensitive matching.
    """
    # Build alias → canonical map
    alias_to_canonical = {}
    for canonical, aliases in name_lookup.items():
        # expect aliases to be iterable
        for alias in aliases:
            alias_to_canonical[str(alias).lower()] = canonical
        alias_to_canonical[str(canonical).lower()] = canonical  # include canonical itself

    # Normalize DataFrame columns
    new_columns = {}
    for col in df.columns:
        canonical_name = alias_to_canonical.get(str(col).lower().strip())
        if canonical_name:
            new_columns[col] = canonical_name
    df = df.rename(columns=new_columns)

    return df


def normalize_content(df, content_lookup):
    """
    Normalize cell values in categorical columns using lookup mappings.
    Performs case-insensitive matching for known aliases.
    """
    for canonical, allowed_values in content_lookup.items():
        if canonical not in df.columns:
            continue

        # Build alias → canonical mapping for allowed values
        alias_map = {}
        if isinstance(allowed_values, dict):
            # If JSON defines alias → canonical pairs directly
            for canon_val, aliases in allowed_values.items():
                alias_map[str(canon_val).lower()] = canon_val
                for alias in aliases:
                    alias_map[str(alias).lower()] = canon_val
        else:
            # If JSON just lists allowed canonical values
            for val in allowed_values:
                alias_map[str(val).lower()] = val

        # Apply normalization
        df[canonical] = (
            df[canonical]
            .astype(str)
            .apply(lambda v: alias_map.get(v.lower().strip(), v) if isinstance(v, str) else v)
        )

    return df

#===================
        
def detect_column(df, canonical_name, name_lookup, content_lookup=None):
    """
    Detect the best column in df for a canonical_name using:
    1. Alias names from name_lookup
    2. Content matching from content_lookup (optional)
    """
    aliases = name_lookup.get(canonical_name, [canonical_name])
    expected_type = None
    if canonical_name in NUMERIC_ONLY_COLUMNS:
        expected_type = "numeric"
    elif canonical_name in TEXT_ONLY_COLUMNS:
        expected_type = "text"
    col = detect_column_by_name_fuzzy(df, aliases)
    
    if not col and content_lookup and canonical_name in content_lookup:
        col = detect_column_by_content_match(df, content_lookup[canonical_name])
    
    return col

    
def detect_column_by_name_fuzzy(df, name_patterns, cutoff=0.8):
    cols_lower = [c.lower() for c in df.columns]
    best_col = None
    best_score = 0

    for pattern in name_patterns:
        pattern_lower = pattern.lower().strip()

        # 1️⃣ Exact match
        if pattern_lower in cols_lower:
            return df.columns[cols_lower.index(pattern_lower)]

        # 2️⃣ Starts-with match
        for idx, col in enumerate(cols_lower):
            if col.startswith(pattern_lower):
                return df.columns[idx]

        # 3️⃣ Full word match
        for idx, col in enumerate(cols_lower):
            words = re.split(r"[._\s]", col)
            if pattern_lower in words:
                return df.columns[idx]

        # 4️⃣ Fuzzy match
        for idx, col in enumerate(cols_lower):
            score = difflib.SequenceMatcher(None, pattern_lower, col).ratio()
            if score > best_score and score >= cutoff:
                best_score = score
                best_col = df.columns[idx]

        # 5️⃣ Substring match (lowest priority)
        for idx, col in enumerate(cols_lower):
            if pattern_lower in col and best_score < 0.95:
                best_col = df.columns[idx]

    return best_col


def validate_column_content(df, col, allowed_values, threshold=0.2):
    """Check whether a column’s content reasonably matches expected values."""
    if col not in df.columns or not isinstance(allowed_values, (list, dict, set)):
        return True  # skip validation if not applicable

    allowed_set = set(str(v).lower() for v in (
        allowed_values.keys() if isinstance(allowed_values, dict) else allowed_values
    ))

    series = df[col].dropna().astype(str).str.lower()
    match_ratio = series.isin(allowed_set).sum() / len(series)
    return match_ratio >= threshold



def detect_column_by_content_match(df, allowed_values, threshold=0.7):
    """
    Detect a column whose values best match the given allowed_values list.
    Returns column name or None.
    """
    if not isinstance(allowed_values, (list, set)) or len(allowed_values) == 0:
        return None

    allowed_set = {str(v).lower() for v in allowed_values}
    best_col = None
    best_ratio = 0.0

    for colname in df.columns:
        # make sure colname is a string, not a list
        if isinstance(colname, (list, tuple)):
            continue

        series = df[colname]
        # ensure we’re working with a Series, not a DataFrame
        if isinstance(series, pd.DataFrame):
            # in case of duplicate column names
            series = series.iloc[:, 0]

        series = series.dropna()
        if series.empty:
            continue

        # force to string before lowercasing
        series = series.astype(str).map(str.lower)

        match_ratio = (series.isin(allowed_set)).sum() / len(series)
        if match_ratio > best_ratio:
            best_ratio = match_ratio
            best_col = colname

    return best_col if best_ratio >= threshold else None



def make_unique_columns(columns):
    """Ensure column names are unique by appending suffixes like .1, .2."""
    seen = {}
    new_cols = []
    for col in columns:
        col = str(col).strip()
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    return new_cols


def build_json_fields(row, keys):
    """
    Extract specified fields from a DataFrame row and build a clean JSON subobject.
    Returns None if no valid fields are found.
    """
    result = {}
    null_like = {"", "na", "n/a", "null", "none", "nan"}

    for k in keys:
        if k in row.index:
            val = row[k]

            if isinstance(val, pd.Series):
                val = val.dropna().iloc[0] if not val.dropna().empty else None

            if pd.notna(val):
                sval = str(val).strip().lower()
                if sval not in null_like:
                    result[k] = val

    return result if result else None

def flatten_listlike_cells(df):
    def flatten_cell(x):
        # Handle pandas Series or non-scalar entries gracefully
        if isinstance(x, pd.Series):
            # Convert to string list or just join as string
            return ";".join(map(str, x.values))
        elif isinstance(x, (list, tuple)):
            return ";".join(map(str, x))
        else:
            return x
    return df.apply(lambda col: col.map(flatten_cell))

    
def detect_id_col_strict(df, name_lookup, key):
    """Detect ID column by exact or prefix name match using lookup aliases."""
    aliases = name_lookup.get(key, [key])
    cols_lower = [c.lower() for c in df.columns]
    for alias in aliases:
        alias_lower = alias.lower()
        for idx, col in enumerate(cols_lower):
            if (
                alias_lower == col
                or col.startswith(alias_lower)
                or alias_lower in col.split(".")
                or alias_lower in col.split("_")
            ):
                return df.columns[idx]
    return None
#-----------------------------------------------------------
def _detect_header_row_from_dataframe_preview(preview_df):
    """
    Detects which row in an Excel preview likely contains column headers.
    Uses heuristics similar to CSV detection:
    - Chooses the first row with mostly text (non-numeric) cells.
    """
    header_idx = 0
    for i in range(len(preview_df)):
        row = preview_df.iloc[i].dropna().astype(str)
        if len(row) < 2:
            continue
        non_numeric_fraction = sum(not x.replace(".", "", 1).isdigit() for x in row) / len(row)
        if non_numeric_fraction > 0.8:
            header_idx = i
            break
    return header_idx
#-----------------------------------------------------------

def read_file_by_extension(path, nrows=None, max_preview=15):
    ext = Path(path).suffix.lower()

    def _detect_header_row_from_lines(lines):
        header_idx = 0
        for i, line in enumerate(lines):
            sep = "," if line.count(",") >= line.count("\t") else "\t"
            parts = [p.strip() for p in line.strip().split(sep) if p.strip()]
            if len(parts) < 2:
                continue
            non_numeric_fraction = sum(not p.replace(".", "", 1).isdigit() for p in parts) / len(parts)
            if non_numeric_fraction > 0.8:
                header_idx = i
                break
        return header_idx

    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        preview_lines = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(max_preview):
                try:
                    preview_lines.append(next(fh))
                except StopIteration:
                    break

        header_line_index = _detect_header_row_from_lines(preview_lines)

        df = pd.read_csv(
            path,
            sep=sep,
            skiprows=range(header_line_index),
            header=0,
            comment="#",  # ✅ ignore metadata comment lines
            encoding="utf-8",
            on_bad_lines="skip",
            nrows=nrows
        )

    elif ext in [".xlsx", ".xls"]:
        preview_df = pd.read_excel(path, header=None, nrows=max_preview)
        header_line_index = _detect_header_row_from_dataframe_preview(preview_df)

        df = pd.read_excel(path, skiprows=range(header_line_index), header=0, nrows=nrows)

    else:
        raise SmartError(f"Unsupported input format: {ext}")

    return df
# ==========================================================
# 🧩 Ingestion (Sample-only Pipeline)
# ==========================================================

@safe_execute("ingest_files_sample")
def ingest_files(input_files, sample_output="sample_output.csv", entity_folder=None, sample_id_col=None, mapping_path=None):
    smart_check(isinstance(input_files, list) and len(input_files) > 0, "No input files provided.")
    for path in input_files:
        validate_file_exists(path, "input data file")

    # --- Entity folder handling (non-interactive) ---
    smart_check(entity_folder is not None, "Missing entity folder. Provide it with --entity-folder.")
    smart_check(os.path.isdir(entity_folder), f"Invalid entity folder: {entity_folder}")


    # --------------------
    # Load lookup tables
    # --------------------
    name_lookup, content_lookup = load_lookup_tables()


    mapping_json = load_mapping_json(mapping_path)
    sample_id_per_file, entity_id_per_file = build_sample_id_overrides(mapping_json)
    
    # --- Validate lookups ---
    smart_check("external_entity_id" in name_lookup, "Missing required 'external_entity_id' in lookup.")
    smart_check("external_sample_id" in name_lookup, "Missing required 'external_sample_id' in lookup.")


    # --- 4️⃣ Load Entity Table Once ---
    # Resolve the entity CSV produced by entity.py
    entity_csv = resolve_entity_csv(entity_folder)

    # Load entity table and prepare references
    entity_df = pd.read_csv(entity_csv)
    entity_table = entity_df.copy()
    validate_dataframe(entity_table, context="entity table")
    smart_check("external_entity_id" in entity_table.columns, "Entity table missing external_entity_id")
    smart_check("entity_id" in entity_table.columns, "Entity table missing entity_id")

    # Normalize and precompute reference ID set (lowercased & trimmed)
    entity_table["external_entity_id"] = (
        entity_table["external_entity_id"].astype(str).str.strip().str.lower()
    )
    ref_entity_ids = set(entity_table["external_entity_id"])



    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # -------------------------------------------------------------
    # 🌟 FIX: DYNAMICALLY ORDER INPUT FILES BY COLUMN COUNT
    # -------------------------------------------------------------
    file_column_counts = {}
    for path in input_files:
        try:
            # Read a small preview to count columns efficiently
            preview_df = read_file_by_extension(path, nrows=10)
            # Use original column count after basic header cleaning
            preview_df.columns = [str(c).strip().replace("\xa0", "").replace("\u200b", "") for c in preview_df.columns]
            file_column_counts[path] = len(preview_df.columns)
        except Exception:
            # If reading fails, assign 0 columns (lowest priority)
            file_column_counts[path] = 0

    # Sort files in descending order based on column count
    # This ensures the file with the most columns is processed first.
    input_files.sort(key=lambda x: file_column_counts[x], reverse=True)
    
    
    # -------------------------------------------------------------
    # --------------------
    # Global ID consistency check
    # --------------------
    id_presence_summary = []
    for path in input_files:
        df_header = read_file_by_extension(path, nrows=0)
        normalized_cols = [c.lower().strip() for c in df_header.columns]
        has_entity_id = any(alias.lower().strip() in normalized_cols
                            for alias in name_lookup.get("external_entity_id", []))
        has_sample_id = any(alias.lower().strip() in normalized_cols
                            for alias in name_lookup.get("external_sample_id", []))
        id_presence_summary.append({
            "file": os.path.basename(path),
            "has_entity_id": has_entity_id,
            "has_sample_id": has_sample_id
        })

    all_have_one_id = all((row["has_entity_id"] ^ row["has_sample_id"]) for row in id_presence_summary)
    any_have_both_ids = any((row["has_entity_id"] and row["has_sample_id"]) for row in id_presence_summary)
    perform_copy_step = False
    print("\n🔍 ID column summary:")
    for row in id_presence_summary:
        print(f"   • {row['file']}: entity_id={row['has_entity_id']}, sample_id={row['has_sample_id']}")
    if all_have_one_id:
        print("✅ All input files have exactly one ID column — will perform safe copy step.")
        perform_copy_step = True
    elif any_have_both_ids:
        print("⚠️ At least one file already has both ID columns — skipping copy step for all files.")
    else:
        print("⚠️ Mixed or missing ID columns detected — skipping copy step for safety.")


    
    # ======================================================
    # 🔧 Helper: normalize column names (handles all spaces, case, etc.)
    # ======================================================
    def clean_colname(name):
        if not isinstance(name, str):
            return ""
        # remove all whitespace (normal + non-breaking), lowercase, and underscores consistent
        name = name.replace("\xa0", " ")  # handle non-breaking spaces
        return re.sub(r"\s+", "", name.strip().lower())  # normalize whitespace + lowercase

    # --------------------
    # Build sample→entity map from files that have both IDs
    # --------------------
    sample_to_entity_map = {}
    for path in input_files:
        df_header = read_file_by_extension(path, nrows=0)
        normalized_cols = [c.lower().strip() for c in df_header.columns]
        has_entity_id = any(alias.lower().strip() in normalized_cols
                            for alias in name_lookup.get("external_entity_id", []))
        has_sample_id = any(alias.lower().strip() in normalized_cols
                            for alias in name_lookup.get("external_sample_id", []))
        if has_entity_id and has_sample_id:
            df_map = read_file_by_extension(path)
            df_map.columns = [str(c).lower().strip() for c in df_map.columns]
            
            # --- MODIFIED: Normalize the lookup list for matching ---
            # Get the list of aliases and lowercase them
            entity_aliases = [a.lower().strip() for a in name_lookup.get("external_entity_id", [])]
            sample_aliases = [a.lower().strip() for a in name_lookup.get("external_sample_id", [])]

            # Now check if the *lowercased* column name is in the *lowercased* alias list
            entity_col = next((c for c in df_map.columns if c in entity_aliases), None)
            sample_col = next((c for c in df_map.columns if c in sample_aliases), None)

            if entity_col and sample_col:
                # ... rest of the map creation (which already handles data cleaning)
                map_pairs = df_map[[sample_col, entity_col]].dropna().astype(str).apply(lambda s: s.str.lower().str.strip())
                sample_to_entity_map.update(dict(zip(map_pairs[sample_col], map_pairs[entity_col])))

    print(f"🧭 Built sample→entity map with {len(sample_to_entity_map)} pairs from files containing both IDs.")

    # --------------------
    # Process each file
    # --------------------
    dfs = []
    for input_path in input_files:
        
        df = read_file_by_extension(input_path)

        validate_dataframe(df, context=input_path)

        # --- Clean header characters before normalization ---
        df.columns = [str(c).strip().replace("\xa0", "").replace("\u200b", "") for c in df.columns]

        # ✅ Mapping overrides (before normalize_column_names so raw headers still exist)
        file_key = os.path.basename(input_path)

        # external_entity_id override
        if file_key in entity_id_per_file:
            chosen = entity_id_per_file[file_key]
            if chosen in df.columns:
                df.rename(columns={chosen: "external_entity_id"}, inplace=True)
                print(f"✅ Mapping override: '{file_key}' uses '{chosen}' as external_entity_id")

        # external_sample_id override
        if file_key in sample_id_per_file:
            chosen = sample_id_per_file[file_key]
            if chosen in df.columns:
                df.rename(columns={chosen: "external_sample_id"}, inplace=True)
                print(f"✅ Mapping override: '{file_key}' uses '{chosen}' as external_sample_id")

        # Continue with your existing normalization (fallback)
        df = normalize_column_names(df, name_lookup)
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces

        #=========================================================
        #=========================================================
        # --- Early detection for single-ID or ambiguous files ---
        #=========================================================
        normalized_cols = [c.lower().strip() for c in df.columns]
        has_entity_id = "external_entity_id" in normalized_cols
        has_sample_id = "external_sample_id" in normalized_cols

        # --- Step A: Global copy (only if all files have one ID) ---
        if perform_copy_step:
            if not has_entity_id and has_sample_id:
                df["external_entity_id"] = df["external_sample_id"]
                print(f"   ↳ Created external_entity_id in {os.path.basename(path)}")
            elif has_entity_id and not has_sample_id:
                df["external_sample_id"] = df["external_entity_id"]
                print(f"   ↳ Created external_sample_id in {os.path.basename(input_path)}")

        # --- Step B: Use sample→entity mapping if available ---
        elif has_sample_id and not has_entity_id and sample_to_entity_map:
            df = df.copy()
            df["external_sample_id"] = df["external_sample_id"].astype(str).str.lower().str.strip()
            df["external_entity_id"] = df["external_sample_id"].map(sample_to_entity_map)
            filled = df["external_entity_id"].notna().sum()
            total = len(df)
            if filled > 0:
                print(f"✅ Filled {filled}/{total} external_entity_id values in {os.path.basename(input_path)} "
                      f"using sample→entity map.")
            else:
                print(f"⚠️ No matches found for sample IDs in {os.path.basename(path)}.")
        
        # --- Step C: Heuristic detection if neither ID is present ---
        if not has_entity_id and not has_sample_id:
            # Try to detect a single column whose values match entity IDs
            match_scores = {}
            for col in df.columns:
                if df[col].notna().any():
                    values = df[col].astype(str).str.lower().tolist()
                    overlap = sum(any(eid in v for eid in ref_entity_ids) for v in values)
                    score = overlap / max(1, len(values))
                    match_scores[col] = score

            if match_scores:
                best_col = max(match_scores, key=match_scores.get)
                best_score = match_scores[best_col]
                
                # Automatically accept if match score is strong enough (e.g. >10%)
                if best_score > 0.1:
                    df["external_entity_id"] = df[best_col].astype(str).str.lower()
                    df["external_sample_id"] = df[best_col].astype(str).str.lower()
                    print(f"✅ Auto-used '{best_col}' for both external_entity_id and external_sample_id "f"(match rate: {best_score:.1%}).\n")
                else:
                    print(f"⚠️ Skipping automatic mapping for '{best_col}' (match too weak: {best_score:.1%}).")
            


        #=========================================================

        # --- Check for duplicate 'external_entity_id'-like columns ---
        candidate_cols = [c for c in df.columns if c.lower().strip() in 
                          [a.lower() for a in name_lookup.get("external_entity_id", ["external_entity_id"])]]

        if len(candidate_cols) > 1:
            #print(f"⚠️ Found multiple possible 'external_entity_id' columns in {os.path.basename(input_path)}: {candidate_cols}")

            best_col = None
            best_overlap = 0.0

            for c in candidate_cols:
                vals = df[c].dropna().astype(str).str.strip().str.lower()
                overlap = len(set(vals) & ref_entity_ids) / max(1, len(set(vals)))
                #print(f"   → {c}: {overlap:.2%} overlap with entity IDs")
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_col = c

            #print(f"✅ Keeping column '{best_col}' as external_entity_id (best overlap {best_overlap:.2%})")
            df.rename(columns={best_col: "external_entity_id"}, inplace=True)
            for c in candidate_cols:
                if c != best_col:
                    df.drop(columns=c, inplace=True)
        elif len(candidate_cols) == 1:
            df.rename(columns={candidate_cols[0]: "external_entity_id"}, inplace=True)      

        # --- 🩹 Fallback for external_sample_id ---
        if "external_sample_id" not in df.columns:
            # --- Clean lookup aliases using helper ---
            aliases = name_lookup.get("external_sample_id", [])
            aliases_clean = [clean_colname(a) for a in aliases]
            matches = []

            # --- Find all matching aliases in this file ---
            for alias in aliases_clean:
                for col in df.columns:
                    if clean_colname(col) == alias:
                        matches.append(col)

            # --- No matches found ---
            if len(matches) == 0:
                raise SmartError(
                    f"❌ Fatal: '{os.path.basename(input_path)}' does not contain any column "
                    f"matching 'external_sample_id'. Expected one of: {name_lookup.get('external_sample_id', [])}"
                )

            # --- Multiple matches found: require explicit --sample-id-col (non-interactive) ---
            elif len(matches) > 1:
                if sample_id_col and sample_id_col in matches:
                    selected_col = sample_id_col
                else:
                    raise SmartError(
                        "Multiple possible 'external_sample_id' columns found in "
                        f"'{os.path.basename(input_path)}': {matches}. "
                        "Re-run with --sample-id-col <one_of_the_above>."
                    )
            else:
                selected_col = matches[0]

            print(f"✅ Using '{selected_col}' as 'external_sample_id'.")
            df.rename(columns={selected_col: "external_sample_id"}, inplace=True)

            # 🚫 Immediately drop all other duplicate ID columns to avoid downstream errors
            for col in matches:
                if col != selected_col and col in df.columns:
                    df.drop(columns=col, inplace=True)
        #print(f"DEBUG BEFORE APPEND - Columns in {os.path.basename(input_path)}: {list(df.columns)}")

        # --- Normalize external_sample_id consistently across all files ---
        if "external_sample_id" in df.columns:
            df["external_sample_id"] = df["external_sample_id"].astype(str).str.strip().str.lower()

        # --- Append cleaned DataFrame ---
        dfs.append(df.loc[:, ~df.columns.duplicated()].copy())
 
    smart_check(len(dfs) > 0, "No DataFrames were loaded successfully.")
    # --- Debug info for loaded DataFrames ---
    print(f"DEBUG: Loaded {len(dfs)} DataFrames.")
    for i, df in enumerate(dfs, 1):
        print(f"  File {i}: {df.shape[0]} rows x {df.shape[1]} columns")

     # --------------------
    # Safe study ID
    # --------------------
    study_id = "STUDY_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    #+++++++++++++++++++
    # 🔹 Create unique output file names 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_output = os.path.join(entity_folder, f"sample_output_{timestamp}.csv")

    # ======================================================
    # 🧩 ENTITY-BASED MERGING LOGIC
    # ======================================================

    print("🔍 Starting entity-based merging...")
    # Ensure entity table is valid
    validate_dataframe(entity_table, context="entity table")
    smart_check("external_entity_id" in entity_table.columns, "Entity table missing external_entity_id")
    smart_check("entity_id" in entity_table.columns, "Entity table missing entity_id")

    # Create an empty list to hold per-entity DataFrames
    entity_sample_records = []

    #Clean up entity IDs (strip + lowercase)
    entity_table["external_entity_id"] = entity_table["external_entity_id"].astype(str).str.strip().str.lower()
    #+++++++++++++++++++++++++++++++++++++

    combined_input_df = pd.concat(dfs, ignore_index=True)
    # Ensure ID column is present and clean in the combined DF for the merge
    if "external_entity_id" in combined_input_df.columns:
        combined_input_df["external_entity_id"] = combined_input_df["external_entity_id"].astype(str).str.strip().str.lower()
    else:
         raise SmartError("Fatal: external_entity_id column is missing from the combined input data.")

    # 🟢 Vectorized join replaces the slow entity-by-entity iteration
    #print("\n🔄 Merging samples with entity table (vectorized INNER join)...")
    merged_df = pd.merge(
        combined_input_df,  # Use the cleaned combined input
        entity_table[["entity_id", "external_entity_id"]],  # only necessary columns
        on="external_entity_id",
        how="inner"
    )

    if merged_df.empty:
        raise SmartError(
            "❌ Invalid or empty merged dataframe. (Resulted in 0 rows after inner join)."
        )

    validate_dataframe(merged_df, context="merged dataframe")

    missing = merged_df["entity_id"].isna().sum()
    if missing > 0:
        raise SmartError(
            f"{missing} samples reference unknown external_entity_id values — "
            "please update the entity table before ingesting these samples."
        )

    # --------------------
    # FIX: AGGREGATE ROWS BY EXTERNAL_SAMPLE_ID TO COALESCE DATA
    # --------------------
    #print("\n🧹 Coalescing duplicate sample data before final deduplication...")
    
    # Create the aggregation dictionary for all columns
    agg_dict = {col: 'first' for col in merged_df.columns}
    
    # 🎯 FIX: REMOVE THE GROUPING COLUMN (external_sample_id) from the agg_dict
    # The groupby operation will automatically keep this column and reset_index will restore it.
    if "external_sample_id" in agg_dict:
        del agg_dict["external_sample_id"]
    
    # Aggregation step: fill in NaNs using data from other rows of the same sample.
    aggregated_df = merged_df.groupby("external_sample_id", dropna=False).agg(
        agg_dict
    ).reset_index()

    # Now, assign the aggregated DataFrame back to merged_df
    merged_df = aggregated_df
    
    #----------------------------------------
    # -------------------------
    # Coalesce duplicate columns created by merge
    # -------------------------
    def coalesce_similar_columns(df):
        """
        Group columns by a 'stem' (remove .<n> and _dup suffix), then for groups
        with >1 column keep the first non-null value (priority = left-most column).
        """
        cols = list(df.columns)
        stems = {}
        for col in cols:

            stem = re.sub(r'(\.\d+|_dup)$', '', col)
            stems.setdefault(stem, []).append(col)

        for stem, group in stems.items():
            if len(group) > 1:
                # create stem column as first non-null among group (preserve ordinal preference)
                df[stem] = df[group].apply(lambda row: next((v for v in row if pd.notna(v)), None), axis=1)
                # drop the non-stem columns
                for c in group:
                    if c != stem:
                        df.drop(columns=c, inplace=True)
        return df

    # --------------------
    # Clean duplicates / flatten columns
    # --------------------
    
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()
    merged_df.columns = make_unique_columns(merged_df.columns)
    

    # ✅ # FIX: Corrected flatten_listlike_cells calls from previous steps
    merged_df = flatten_listlike_cells(merged_df)
    merged_df = coalesce_similar_columns(merged_df)
    # -------------------------
    # Flatten any list/Series-like cells that sometimes appear after merges
    # -------------------------
    def flatten_cell(v):
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            for e in v:
                if pd.notna(e):
                    return e
            return None
        return v
   
    #------------------------------------------
    
    # ---------------- Build Sample DF ---------------- #
    sample_df = merged_df.drop_duplicates(subset=["external_sample_id"]).copy()

    # --- Initialize canonical columns ---
    for col in SAMPLE_COLUMNS:
        if col not in sample_df.columns:
            sample_df[col] = None

    # FIX: Corrected flatten_listlike_cells calls from previous steps
    sample_df = flatten_listlike_cells(sample_df)
    entity_table = flatten_listlike_cells(entity_table)
    
    # --- Remove duplicate columns ---
    sample_df = sample_df.loc[:, ~sample_df.columns.duplicated()]
    entity_table = entity_table.loc[:, ~entity_table.columns.duplicated()]


    # Normalize strings in external_entity_id
    for df in [sample_df, entity_table]:
        if "external_entity_id" in df.columns:
            # Ensure we have a Series, not a DataFrame
            if isinstance(df["external_entity_id"], pd.DataFrame):
                df["external_entity_id"] = df["external_entity_id"].iloc[:, 0]
            df["external_entity_id"] = df["external_entity_id"].astype(str).str.strip().str.lower()
        else:
            # Should not happen after merge, but adding a check to prevent KeyError
            print("⚠️ Warning: external_entity_id missing in dataframe during final normalization.")

    # Build mapping: external_entity_id -> entity_id
    mapping = dict(zip(entity_table["external_entity_id"], entity_table["entity_id"]))

    # Map entity_id in sample_df safely
    sample_df["entity_id"] = sample_df["external_entity_id"].map(mapping)

    # Optional: warn if some IDs could not be mapped
    if "external_entity_id" in sample_df.columns:
        missing_ids = set(sample_df["external_entity_id"]) - set(entity_table["external_entity_id"])
        if missing_ids:
            print(f"⚠️ Warning: {len(missing_ids)} external_entity_id(s) not found in entity table: {list(missing_ids)[:10]}…")


    # --- UUID generation for sample IDs  ---
    sample_id_map = {}

    def get_sample_id(external_id):
        if pd.isna(external_id):
            return None
        sval = str(external_id).strip().lower()
        if sval in {"", "na", "n/a", "nan", "null", "none"}:
            return None
        if external_id not in sample_id_map:
            sample_id_map[external_id] = str(uuid.uuid4())
        return sample_id_map[external_id]

    # --- Generate sample_id safely ---
    sample_df["sample_id"] = sample_df["external_sample_id"].apply(get_sample_id)

    # ======================================================
    # 🧮 Derive missing fields (is_tumor_sample, contamination_percent)
    # ======================================================
    # 1️⃣ Derive is_tumor_sample if not in input
    if "is_tumor_sample" not in sample_df.columns or sample_df["is_tumor_sample"].isna().all():
        if "sample_type" in sample_df.columns:
            tumor_keywords = ["tumor", "metastatic", "recurrence", "recurrent", "relapse", "primary"]
            pattern = "|".join(tumor_keywords)
            sample_df["is_tumor_sample"] = sample_df["sample_type"].astype(str).str.contains(
                pattern, case=False, na=False
            )
        else:
            sample_df["is_tumor_sample"] = None

    # 2️⃣ Derive contamination_percent if missing, using tumor_purity_estimate
    if "contamination_percent" not in sample_df.columns or sample_df["contamination_percent"].isna().all():
        if "tumor_purity_estimate" in sample_df.columns:
            purity = pd.to_numeric(sample_df["tumor_purity_estimate"], errors="coerce")

            # 🧠 Auto-detect: if most values are <= 1, treat as fraction (ABSOLUTE-style)
            if purity.dropna().le(1).mean() > 0.5:
                # Fractional values (e.g. 0.85) → multiply by 100
                purity = purity * 100
                
            # Compute contamination = 100 - purity
            sample_df["contamination_percent"] = (100 - purity).clip(lower=0, upper=100)
        else:
            sample_df["contamination_percent"] = None
    
    # --- JSON fields ---
    sample_df["storage_conditions_json"] = sample_df.apply(
        lambda row: json.dumps(build_json_fields(row, STORAGE_CONDITIONS_KEYS), ensure_ascii=False)
        if build_json_fields(row, STORAGE_CONDITIONS_KEYS) else None,
        axis=1
    )
    sample_df.loc[sample_df["storage_conditions_json"] == "{}", "storage_conditions_json"] = None

    sample_df["purity_metrics_json"] = sample_df.apply(
        lambda row: json.dumps(build_json_fields(row, PURITY_METRICS_KEYS), ensure_ascii=False)
        if build_json_fields(row, PURITY_METRICS_KEYS) else None,
        axis=1
    )

    sample_df.loc[sample_df["purity_metrics_json"] == "{}", "purity_metrics_json"] = None
    
    # --- Timestamps ---
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sample_df["creation_timestamp"] = now
    sample_df["last_updated_timestamp"] = now

    # ✂️ Restrict output columns for sample-only pipeline
    sample_df = sample_df[SAMPLE_COLUMNS]

    # 🔹 Apply study_id to all sample rows
    sample_df["study_id"] = study_id


    # --- Validate uniqueness ---
    dup_samples = sample_df["external_sample_id"].duplicated().sum()
    if dup_samples > 0:
        print(f"⚠️ Warning: {dup_samples} duplicate sample IDs detected.")
        sample_df.drop_duplicates(subset=["external_sample_id"], inplace=True)
        print("🧹 Duplicates removed.")

    # ---------------- Save ---------------- #
    sample_df.to_csv(sample_output, index=False, encoding="utf-8")
    print(f"✅ Sample table saved to: {sample_output}")
    sample_df.loc[sample_df["purity_metrics_json"] == "{}", "purity_metrics_json"] = None
    
    return sample_output


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
                print("   Usage: python sample.py <input_file1> [<input_file2> ...]")
                sys.exit(1)
            else:
                self.print_help()
                sys.exit(1)

    parser = SmartArgumentParser(
        description="Ingest sample files into standardized format and link to entities."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Paths to input files (CSV, TSV, XLSX). Example: file1.csv file2.tsv"
    )
    parser.add_argument(
        "--output",
        default="sample_output.csv",
        help="Output CSV filename (default: sample_output.csv)"
    )

    parser.add_argument(
        "--entity-folder",
        required=True,
        help="Path to the folder that contains entity_output_*.csv produced by entity.py"
    )
    
    parser.add_argument(
        "--sample-id-col",
        default=None,
        help="When multiple candidates for external_sample_id are present, pick this column name"
    )

    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional: sample mapping JSON produced by recommender (e.g., ACC_Sample_mapping.json)."
    )

    args = parser.parse_args()

    try:
        out_file = ingest_files(
            args.input_files,
            sample_output=args.output,
            entity_folder=args.entity_folder,
            sample_id_col=args.sample_id_col,
            mapping_path=args.mapping
        )
        
        logging.info(f"✅ Sample data successfully saved to {out_file}")
    except SmartError as e:
        logging.error(f"🛑 SMARTWAY FATAL HALT: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"🛑 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc(limit=3)
        sys.exit(1)


