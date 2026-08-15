#!/usr/bin/env python
# coding: utf-8

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
import logging 
import argparse

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


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


# ---------------- Safe Execute ---------------- #
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
def safe_execute(step_name):
    """Decorator for wrapping critical functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except SmartError:
                raise
            except Exception as e:
                # Unified logging of unexpected exceptions
                logging.error(f"🚨 Smartway HALT: Error in step '{step_name}': {type(e).__name__}: {e}", exc_info=True)
                # Raise SmartError without printing again
                raise SmartError(f"Fatal error in '{step_name}' — stopping execution.") from None
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



# ------------------ CONFIG ------------------ #
COLUMN_NAME_LOOKUP_PATH = "column_name_lookup.json"
COLUMN_CONTENT_LOOKUP_PATH = "column_content_lookup.json"
ORGANISM_MAP_PATH = "organism_map.json"

ENTITY_COLUMNS = [
    "entity_id", "external_entity_id", "organism", "gender", "date_of_birth", "overall_survival_time",
    "demographics_json", "clinical_profile_json",
    "experimental_profile_json", "consent_status", "consent_version",
    "data_sharing_restrictions", "creation_timestamp", "last_updated_timestamp"
]



DEMOGRAPHICS_KEYS = ["ethnicity", "race", "country_of_origin", "occupation",
                     "education_level", "smoking_status", "alcohol_use"]
CLINICAL_PROFILE_KEYS = ["vital_status", "date_of_death", "cause_of_death", "diagnosis",
                         "diagnosis_date", "age_at_diagnosis", "comorbidities", "prior_treatments",
                         "performance_status", "family_history", "cardiovascular_disease"]
EXPERIMENTAL_PROFILE_KEYS = ["vital_status", "date_of_death", "cause_of_death",
                             "genotype", "strain", "disease_model", "housing_conditions",
                             "diet", "procedures", "observations"]
CATEGORICAL_COLUMNS = [
    "gender", "ethnicity", "race", "country_of_origin",
    "consent_status", "consent_version", "data_sharing_restrictions"
]

# ==========================================================
# 🧩 COLUMN TYPE RULES
# ==========================================================
NUMERIC_ONLY_COLUMNS = [
    "age_at_diagnosis", "days_to_birth", "year_of_birth", "age_at_index.demographic", "performance_status", "overall_survival_time"]

TEXT_ONLY_COLUMNS = [
    "external_entity_id", "gender", "ethnicity", "race",
    "country_of_origin", "occupation", "education_level",
    "smoking_status", "alcohol_use", "vital_status",
    "diagnosis", "cause_of_death", "comorbidities", "family_history",
    "genotype", "strain", "disease_model", "housing_conditions",
    "diet", "procedures", "observations", "consent_status",
    "consent_version", "data_sharing_restrictions"
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
    validate_file_exists(mapping_path, "entity_mapping.json")
    validate_json(mapping_path, "entity_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_mapping_overrides(mapping_json):
    """
    Returns:
      overrides: canonical_key -> list of preferred source columns (strings)
      ext_id_per_file: basename -> preferred external_entity_id source column
    """
    overrides = {}
    ext_id_per_file = {}

    if not mapping_json or "mappings" not in mapping_json:
        return overrides, ext_id_per_file

    m = mapping_json["mappings"]

    # external_entity_id: stored per file as source_column
    ext = m.get("external_entity_id", {})
    if isinstance(ext, dict) and "per_file" in ext:
        for file_key, info in ext["per_file"].items():
            col = info.get("source_column")
            if col:
                ext_id_per_file[file_key] = col
                overrides.setdefault("external_entity_id", []).append(col)

    # generic map fields
    for key, spec in m.items():
        if not isinstance(spec, dict):
            continue

        if spec.get("mode") == "map":
            col = spec.get("source", {}).get("column")
            if col:
                overrides.setdefault(key, []).append(col)

        if spec.get("mode") == "compute" and "inputs" in spec:
            for in_key, in_spec in spec["inputs"].items():
                col = in_spec.get("column")
                if col:
                    overrides.setdefault(in_key, []).append(col)

        if spec.get("mode") == "json_subfields":
            for subf, sub_spec in spec.get("subfields", {}).items():
                if sub_spec.get("mode") == "map":
                    col = sub_spec.get("source", {}).get("column")
                    if col:
                        overrides.setdefault(subf, []).append(col)

    # de-duplicate while preserving order
    for k in list(overrides.keys()):
        seen = set()
        dedup = []
        for c in overrides[k]:
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        overrides[k] = dedup

    return overrides, ext_id_per_file
#===================================================

def derive_date_of_birth(row, date_of_diag_col=None, year_of_diag_col=None, sample_date_col=None, dob_col_days=None, dob_col_year=None):
    """
    Compute date_of_birth using available information.
    Preference:
    1. days_to_birth + reference date (diagnosis)
    2. year_of_birth fallback
    """

    ref_date = None

    # 1️⃣ Prefer diagnosis date
    if date_of_diag_col and pd.notna(row.get(date_of_diag_col)):
        ref_date = pd.to_datetime(row[date_of_diag_col], errors="coerce")

    # 2️⃣ Fallback to year_of_diagnosis
    elif year_of_diag_col and pd.notna(row.get(year_of_diag_col)):
        ref_date = pd.to_datetime(f"{int(row[year_of_diag_col])}-06-30", errors="coerce")

    # Compute from days_to_birth
    if dob_col_days and pd.notna(row.get(dob_col_days)) and ref_date is not None:
        days = float(row[dob_col_days])
        # Ensure birth is in the past relative to reference date
        return (ref_date - pd.to_timedelta(abs(days), unit="D")).strftime("%Y-%m-%d")


    # Fallback to year_of_birth
    elif dob_col_year and pd.notna(row.get(dob_col_year)):
        val = row.get(dob_col_year)
         # Handle missing or invalid values safely

        try:
            year = int(str(val).strip())
            if 1800 <= year <= 2100:
                return f"{year}-01-01"
            else:
                return None
        except (ValueError, TypeError):
            return None
        # invalid or non-numeric year
        #return f"{int(row[dob_col_year])}-01-01"

    #return None

#==================================================

def derive_date_of_death(row, date_of_diag_col=None, year_of_diag_col=None, sample_date_col=None, dod_col_days=None, dod_col_year=None):
    """
    Compute date_of_death using available information.
    Preference:
    1. days_to_death + reference date (diagnosis)
    2. year_of_death fallback
    """

    ref_date = None

    # 1️⃣ Prefer diagnosis date
    if date_of_diag_col and pd.notna(row.get(date_of_diag_col)):
        ref_date = pd.to_datetime(row[date_of_diag_col], errors="coerce")

    # 2️⃣ Fallback to year_of_diagnosis
    elif year_of_diag_col and pd.notna(row.get(year_of_diag_col)):
        ref_date = pd.to_datetime(f"{int(row[year_of_diag_col])}-06-30", errors="coerce")

    # Compute from days_to_death
    if dod_col_days and pd.notna(row.get(dod_col_days)) and ref_date is not None:
        try:
            days = float(row[dod_col_days])
        except (ValueError, TypeError):
            return None  # skip invalid entries

        # Ensure death is in the past relative to reference date
        return (ref_date - pd.to_timedelta(abs(days), unit="D")).strftime("%Y-%m-%d")


    # Fallback to year_of_death
    elif dod_col_year and pd.notna(row.get(dod_col_year)):
        val = row.get(dod_col_year)
         # Handle missing or invalid values safely

        try:
            year = int(str(val).strip())
            if 1800 <= year <= 2100:
                return f"{year}-01-01"
            else:
                return None
        except (ValueError, TypeError):
            return None
        # invalid or non-numeric year
        #return f"{int(row[dob_col_year])}-01-01"

    #return None




#====================================================

@safe_execute("load_organism_map")
def load_organism_map(path=ORGANISM_MAP_PATH):
    if not os.path.exists(path):
        print(f"⚠️ {path} not found. Using built-in defaults.")
        return {}
    validate_json(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
        
#-----------------------------------------------------------

@safe_execute("infer_organism_from_filename")
def infer_organism_from_filename(filepaths, mapping=None):
    if not isinstance(filepaths, (list, tuple)) or not filepaths:
        raise SmartError("File list for organism inference is empty or invalid.")
    return _infer_organism_from_filename(filepaths, mapping)

#-----------------------------------------------------------

# 🧩 Helper extraction for safety-wrapped version
def _infer_organism_from_filename(filepaths, mapping=None):
    if mapping is None:
        mapping = load_organism_map()
    joined_names = " ".join(str(p).lower() for p in filepaths)
    for keyword, organisms in mapping.items():
        if keyword.lower() in joined_names:
            org = organisms[0] if isinstance(organisms, list) else organisms
            return org
    #print("⚠️ No organism inferred from filenames.")
    return None
#-----------------------------------------------------------
def resolve_organism_from_user_input(user_input, mapping=None):
    """
    Map a user-provided organism/keyword to a canonical organism string
    using organism_map.json. Falls back to the raw user_input if
    no mapping is found.
    """
    if not user_input:
        return None

    if mapping is None:
        mapping = load_organism_map()

    norm = str(user_input).strip().lower()
    if not norm:
        return None

    # 1) Match against mapping keys (e.g. "human", "mouse")
    for keyword, organisms in mapping.items():
        if norm == str(keyword).lower():
            return organisms[0] if isinstance(organisms, list) else organisms

    # 2) Match against mapped organism values / aliases
    for keyword, organisms in mapping.items():
        if isinstance(organisms, list):
            for org in organisms:
                if norm == str(org).lower():
                    # Return canonical (first) organism for that keyword
                    return organisms[0]
        else:
            if norm == str(organisms).lower():
                return organisms

    # 3) No mapping found → warn and signal 'no match'
    print(
        f"⚠️ Provided organism '{user_input}' not found in organism_map.json; "
        f"ignoring it and falling back to filename-based inference."
    )
    return None

#----------------------------------------------------------

def detect_column_type(df, threshold=0.8):
    """
    Detect numeric vs string columns by content.
    Returns a dict: {col_name: "numeric" or "string"}
    """
    col_types = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            col_types[col] = "string"
            continue
        numeric_count = series.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().sum()
        ratio = numeric_count / len(series)
        col_types[col] = "numeric" if ratio >= threshold else "string"
    return col_types

#===================



RESERVED_COLS = {"demographics_json", "clinical_profile_json", "experimental_profile_json"}
def normalize_column_names(df, name_lookup):
    """
    Map input column names to standardized schema using lookup.
    Handles multi-alias lists and performs case-insensitive matching.
    """
    # Build alias → canonical map
    alias_to_canonical = {}
    for canonical, aliases in name_lookup.items():
        if canonical.lower() in RESERVED_COLS:
            continue  # 🚫 Skip overwriting reserved output columns
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
        
def detect_column(df, canonical_name, name_lookup, content_lookup=None, mapping_overrides=None):
    """
    Detect the best column in df for a canonical_name using:
    0) mapping_overrides (if provided)
    1) Alias names from name_lookup
    2) Content matching from content_lookup (optional)
    """
    # 0) mapping override
    if mapping_overrides and canonical_name in mapping_overrides:
        for cand in mapping_overrides[canonical_name]:
            if cand in df.columns:
                return cand

    aliases = name_lookup.get(canonical_name, [canonical_name])
    expected_type = None
    if canonical_name in NUMERIC_ONLY_COLUMNS:
        expected_type = "numeric"
    elif canonical_name in TEXT_ONLY_COLUMNS:
        expected_type = "text"

    col = detect_column_by_name_fuzzy(df, aliases, expected_type=expected_type)

    if not col and content_lookup and canonical_name in content_lookup:
        col = detect_column_by_content_match(df, content_lookup[canonical_name], expected_type=expected_type)

    return col

def is_text_like_column(series):
    """Return True if the column is mostly text (not numeric)."""
    if series.empty:
        return False
    numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
    return numeric_ratio < 0.5  # less than 50% numeric → text-like

def is_numeric_like_column(series):
    """Return True if the column is mostly numeric."""
    if series.empty:
        return False
    numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
    return numeric_ratio > 0.5



def detect_column_by_name_fuzzy(df, name_patterns, cutoff=0.9, expected_type=None):
    """
    Find column in df that exactly matches one of the given name patterns.
    Only allows exact case-insensitive matches with JSON aliases.
    """
    cols_lower = [c.lower().strip() for c in df.columns]
    for pattern in name_patterns:
        pattern_lower = pattern.lower().strip()
        if pattern_lower in cols_lower:
            col = df.columns[cols_lower.index(pattern_lower)]
            if expected_type == "numeric" and not is_numeric_like_column(df[col]):
                continue
            if expected_type == "text" and not is_text_like_column(df[col]):
                continue
            return col
    return None


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



def detect_column_by_content_match(df, allowed_values, threshold=0.7, expected_type=None):
    """
    Detect a column whose values best match the given allowed_values list.
    Only considers columns compatible with expected_type ('text' or 'numeric').
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

        # --- TYPE FILTER --- #
        if expected_type == "numeric" and not is_numeric_like_column(series):
            continue
        if expected_type == "text" and not is_text_like_column(series):
            continue
        # -------------------- #

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

def build_json_field(df, keys, name_lookup, content_lookup, field_name):
    """
    Build a JSON field (e.g., demographics_json, clinical_profile_json, experimental_profile_json)
    using both name-based and content-based column detection.
    """
    key_to_col = {}
    for key in keys:
        # 🔒 Always prefer explicitly derived date columns if present
        if key == "date_of_death" and "date_of_death" in df.columns:
            key_to_col[key] = "date_of_death"
            continue
        if key == "date_of_birth" and "date_of_birth" in df.columns:
            key_to_col[key] = "date_of_birth"
            continue

        # 🟢 If a canonical column already exists, use it directly
        if key in df.columns:
            key_to_col[key] = key
            continue

        # Otherwise fall back to detection via lookup tables
        col = detect_column(df, key, name_lookup, content_lookup)
        if col:
            key_to_col[key] = col

    # Values considered "null-like" and thus skipped
    null_like = {
        "",
        "na",
        "n/a",
        "null",
        "none",
        "nan",
        "[unknown]",
        "[not available]",
    }
    # Also treat stringified zeros as null-like in this context
    null_like.update({"0", "0.0", "0."})

    def build_row_json(row):
        data = {}
        for key, col in key_to_col.items():
            # 1) Prefer an exact column name match
            if col in row.index:
                val = row[col]
            else:
                # 2) Fallback: substring match for dotted / suffixed names
                candidates = [c for c in row.index if col.lower() in c.lower()]
                if not candidates:
                    continue

                # Prefer candidate whose lower-case name exactly matches col.lower()
                exact_candidates = [c for c in candidates if c.lower() == col.lower()]
                chosen_col = exact_candidates[0] if exact_candidates else candidates[0]
                val = row[chosen_col]

            # Skip null / NA / null-like values
            if pd.isna(val):
                continue
            sval = str(val).strip().lower()
            if sval in null_like:
                continue

            data[key] = val

        # If no fields for this row, return None; otherwise JSON-encode
        return None if not data else json.dumps(data)

    result = df.apply(build_row_json, axis=1)

    # Debug helper (optional, but useful while you’re checking BRCA)
    #if field_name == "clinical_profile_json":
        #print(
            #f"DEBUG: built clinical_profile_json for "
            #f"{result.notna().sum()}/{len(df)} rows"
        #)

    return result


def flatten_listlike_cells(df):
    """
    Cleans dataframe cells that contain list-like strings (e.g., "['a', 'b']" or "[1, 2]").
    Converts them into flat, comma-separated strings.
    """
    def flatten_cell(cell):
        if pd.isna(cell):
            return cell
        if isinstance(cell, (list, tuple, set)):
            return ", ".join(map(str, cell))
        if isinstance(cell, str):
            # Detect strings that look like lists (e.g., "['a','b']", "[1,2]")
            if re.match(r"^\[.*\]$", cell.strip()):
                try:
                    parsed = eval(cell)
                    if isinstance(parsed, (list, tuple, set)):
                        return ", ".join(map(str, parsed))
                except Exception:
                    pass  # Keep as original if eval fails
        return cell

    # Apply flattening to every column
    for col in df.columns:
        df[col] = df[col].apply(flatten_cell)
    return df

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
        #logging.info(f"🧾 Header line detected at row {header_line_index}")

        df = pd.read_csv(
            path,
            sep=sep,
            skiprows=range(header_line_index),
            header=0,
            comment="#",  # ✅ ignore metadata comment lines
            encoding="utf-8",
            on_bad_lines="skip"
        )

    elif ext in [".xlsx", ".xls"]:
        preview_df = pd.read_excel(path, header=None, nrows=max_preview)
        header_line_index = _detect_header_row_from_dataframe_preview(preview_df)
        #logging.info(f"🧾 Header line detected at row {header_line_index}")

        df = pd.read_excel(path, skiprows=range(header_line_index), header=0)

    else:
        raise SmartError(f"Unsupported input format: {ext}")

    return df



# ==========================================================
# 🧩 Ingestion (Entity Pipeline)
# ==========================================================

@safe_execute("ingest_files_entity")
def ingest_files(input_files,
                 entity_output="entity_output.csv",
                 outdir=None,
                 external_id_col=None,
                 organism=None,
                 os_unit=None,
                 mapping_path=None):   # os_unit: "days", "months", or None (auto-detect)
   
    smart_check(isinstance(input_files, list) and len(input_files) > 0, "No input files provided.")

    # ======================================================
    # 🗂️ Determine output folder (non-interactive)
    #   - if --outdir is given, use it (create if missing)
    #   - otherwise, create a timestamped folder
    # ======================================================
    if outdir:
        folder_name = outdir.strip().replace(" ", "_")
    else:
        folder_name = f"entity_out_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # allow letters, digits, underscore, dash
    if not re.match(r'^[A-Za-z0-9/_-]+$', folder_name):
        raise SmartError(f"❌ Invalid --outdir '{folder_name}'. Use letters, numbers, underscores, dashes, and '/'.")

    os.makedirs(folder_name, exist_ok=True)
    print(f"✅ Using output folder: '{folder_name}'")

    # Hint for overall survival time unit inferred from column names (if user doesn't provide one)
    os_unit_hint = None


    #--------------------------------------------------
    for path in input_files:
        validate_file_exists(path, "input data file")


    # --------------------
    # Load lookup tables
    # --------------------
    name_lookup, content_lookup = load_lookup_tables()

    # --- Validate lookups ---
    smart_check("external_entity_id" in name_lookup, "Missing required 'external_entity_id' in lookup.")
    #====================================================================================
    mapping_json = load_mapping_json(mapping_path)
    mapping_overrides, ext_id_per_file = build_mapping_overrides(mapping_json)

    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    # Check input file columns for external_entity_id
    # --------------------
    # Read input files once and check for external_entity_id
    dfs = []
    for input_path in input_files:
        df = read_file_by_extension(input_path)
        #+++++++++++++++++++++++++++

        file_key = os.path.basename(input_path)
        if file_key in ext_id_per_file:
            chosen = ext_id_per_file[file_key]
            if chosen in df.columns:
                df.rename(columns={chosen: "external_entity_id"}, inplace=True)
                print(f"✅ Mapping override: '{file_key}' uses '{chosen}' as external_entity_id")
            else:
                print(f"⚠️ Mapping override column '{chosen}' not found in '{file_key}'. Falling back to lookup detection.")


        #++++++++++++++++++++++++++

        validate_dataframe(df, context=input_path)
        
        
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces

        #--------------------------------------------------------------------

        # --- Infer OS unit from original column names if user did not specify one ---
        if os_unit is None and os_unit_hint is None:
            # phrases suggesting "months"
            month_keywords = ["month", "months", "mon", "mth", "mths"]

            for col_name in df.columns:
                col_lower = str(col_name).lower()

                # Split into tokens on non-alphanumeric boundaries:
                #   "OS_MONTHS"   -> ["os", "months"]
                #   "overall_survival_months" -> ["overall", "survival", "months"]
                tokens = re.split(r"[^a-z0-9]+", col_lower)
                tokens = [t for t in tokens if t]  # drop empty strings

                # Does this look like an overall survival column?
                has_os_marker = (
                    "os" in tokens or                              # e.g. "OS_MONTHS"
                    ("overall" in tokens and "survival" in tokens) or
                    "overallsurvival" in col_lower or
                    "survival_time" in col_lower or
                    "survivaltime" in col_lower
                )

                # Does the name suggest "months"?
                has_month_marker = (
                    any(k in col_lower for k in month_keywords) or
                    "m" in tokens or "mo" in tokens   # e.g. "OS_M" or "OS_MO"
                )

                if has_os_marker and has_month_marker:
                    os_unit_hint = "months"
                    break


        #-------------------------------------------------------------------------

        # --- Clean headers using helper ---
        header_cols_clean = [clean_colname(c) for c in df.columns]

        # --- Clean lookup aliases using helper ---
        aliases_clean = [clean_colname(a) for a in name_lookup.get("external_entity_id", [])]

        # --- Find all matching aliases in this file ---
        matches = []
        for alias in aliases_clean:
            for col in df.columns:
                if clean_colname(col) == alias:
                    matches.append(col)

        # --- No matches found ---
        if len(matches) == 0:
            raise SmartError(
                f"❌ Fatal: '{os.path.basename(input_path)}' does not contain any column "
                f"matching 'external_entity_id'. Expected one of: {name_lookup.get('external_entity_id', [])}"
            )

         # --- Exactly one match found ---
        elif len(matches) == 1:
            chosen = matches[0]
            df.rename(columns={matches[0]: "external_entity_id"}, inplace=True)
            print(f"✅ Found and renamed column '{matches[0]}' → 'external_entity_id'")

        # --- Multiple matches found: auto-select based on overlap with other input files ---
        else: # len(matches) > 1:
            selected_col = None

            # 1) Explicit override (optional but recommended to keep)
            if external_id_col:
                if external_id_col in matches:
                    selected_col = external_id_col
                    print(f"✅ Using '{selected_col}' as 'external_entity_id' (via --external-id-col).")
                else:
                    raise SmartError(
                        f"❌ --external-id-col '{external_id_col}' not in candidates {matches} "
                        f"for '{os.path.basename(input_path)}'."
                    )
            # 2) Auto-heuristic (your overlap code) if not overridden
            if not selected_col:
                # --- normalization to maximize overlap signal ---
                def norm(s):
                    return str(s).strip().upper().replace(" ", "").replace(".", "")
                overlap_scores = {col: 0 for col in matches}
                cand_vals = {
                    col: set(df[col].dropna().astype(str).map(norm).unique())
                    for col in matches
                }

                # Compare candidates against all other input files
                for other_path in input_files:
                    if other_path == input_path:
                        continue
                    try:
                        other_df = read_file_by_extension(other_path)
                        
                    except Exception as e:
                        continue

                    for other_col in other_df.columns:
                        other_vals = other_df[other_col].dropna().astype(str).map(norm)
                        if other_vals.empty:
                            continue
                        other_set = set(other_vals.unique())
                        for col in matches:
                            if cand_vals[col]:

                                overlap_scores[col] += len(cand_vals[col] & other_set)

            # choose max; break ties deterministically with first appearance
            if any(score > 0 for score in overlap_scores.values()):
                selected_col = max(matches, key=lambda c: (overlap_scores[c], -matches.index(c)))
                print(f"✅ Using '{selected_col}' (auto-selected by overlap). Scores: {overlap_scores}")
            else:
                # 3) No signal: pick policy
                # raise SmartError(...)   # safer

                selected_col = matches[0]  # convenience fallback
                print(f"⚠️ No overlap signal among {matches}. Defaulting to '{selected_col}'. "
                      f"Use --external-id-col to be explicit.")

            # Apply choice and drop the rest
            df.rename(columns={selected_col: "external_entity_id"}, inplace=True)
            for col in matches:
                if col != selected_col and col in df.columns:
                    df.drop(columns=col, inplace=True)


       

        # Normalize column names first
        df = normalize_column_names(df, name_lookup)

        #++++++++++++++++++++++++++
        #print(f"🔍 Normalized columns for {os.path.basename(input_path)}:\n", list(df.columns))

        #+++++++++++++++++++++++++
        dfs.append(df)
    smart_check(len(dfs) > 0, "No DataFrames were loaded successfully.")

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     # --------------------
    # Safe study ID
    # --------------------
    study_id = "STUDY_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    # 🔹 Create unique output file names using the study_id
    base_entity_name = Path(entity_output).stem
    # Combine folder name and output filename
    entity_output = os.path.join(folder_name, f"{base_entity_name}_{study_id}.csv")
    
    
    # --- Merge all DataFrames safely ---
    try:
        merged_df = dfs[0]
        for j, other_df in enumerate(dfs[1:], start=1):
            possible_keys = ["external_entity_id"]
            common_keys = [k for k in possible_keys if k in merged_df.columns and k in other_df.columns]
            smart_check(len(common_keys) > 0, f"No common merge keys between dataframes.")
            merged_df = pd.merge(merged_df, other_df, on=common_keys, how="outer", suffixes=("", "_dup"))
    except Exception as e:
        raise SmartError(f"Merge operation failed: {e}")

    validate_dataframe(merged_df, context="merged dataframe")
    
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
    merged_df = coalesce_similar_columns(merged_df)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()
    merged_df.columns = make_unique_columns(merged_df.columns)

   
    merged_df = flatten_listlike_cells(merged_df)

    # -------------------------------------------------------------------
    # 🎯 STEP 1: DERIVE CRITICAL DATE COLUMNS HERE
    # -------------------------------------------------------------------
    
    # Locate the necessary source columns once
    diag_date_col = detect_column(merged_df, "diagnosis_date", name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)
    diag_year_col = detect_column(merged_df, "year_of_diagnosis", name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)
    dod_days_col = detect_column(merged_df, "days_to_death", name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)
    dod_year_col = detect_column(merged_df, "year_of_death", name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)

    
    
    # Apply the derivation function and create the new column
    merged_df["date_of_death"] = merged_df.apply(
        derive_date_of_death,
        axis=1,
        date_of_diag_col=diag_date_col,
        year_of_diag_col=diag_year_col,
        dod_col_days=dod_days_col,
        dod_col_year=dod_year_col,
    )

    # -------------------------------------------------------------------
    # --- Derive / normalize overall survival time ---
    os_time_col = detect_column(merged_df, "overall_survival_time", name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)

    # Decide which unit to use:
    # 1) explicit user input (--os-unit)
    # 2) hint from column names (os_unit_hint)
    # 3) default to days
    effective_os_unit = (os_unit or os_unit_hint or "days")
    
    if os_time_col:
        # Coerce to numeric (raw values from detected OS column)
        os_series = pd.to_numeric(
            merged_df[os_time_col],
            errors="coerce"
        )

        # If the unit is months, convert to days before storing
        if str(effective_os_unit).lower() == "months":
            # ~30.4375 days per month (average)
            os_series = os_series * 30.4375
            os_series = os_series.round()

        merged_df["overall_survival_time"] = os_series
    else:
        # Ensure the column exists even if we couldn't detect anything
        merged_df["overall_survival_time"] = None

    #---------------------------------------------------------------------

    def detect_id_col_strict(df, name_lookup, key):
        """Detect ID column by exact or prefix name match using lookup aliases."""
        aliases = name_lookup.get(key, [key])
        cols_lower = [c.lower().strip() for c in df.columns]
        for alias in aliases:
            alias_lower = alias.lower().strip()
            for idx, col in enumerate(cols_lower):
                if (
                    alias_lower == col
                    or col.startswith(alias_lower)
                    or alias_lower in col.split(".")
                    or alias_lower in col.split("_")
                ):
                    return df.columns[idx]
        return None


    # --------------------
    # === Detect IDs ONCE ===
    # --------------------
    def detect_ids_once(df, name_lookup, content_lookup=None):
        """
        Detect external_entity_id using JSON lookups.
        Returns a tuple of column names: (entity_col)
        """
        # Detect entity ID
        entity_col = detect_id_col_strict(df, name_lookup, "external_entity_id")


        # Rename to canonical names if found
        if entity_col and entity_col != "external_entity_id":
            if "external_entity_id" not in df.columns:
                df.rename(columns={entity_col: "external_entity_id"}, inplace=True)
                
        return df, "external_entity_id" if entity_col else None

    merged_df, external_entity_col = detect_ids_once(merged_df, name_lookup, content_lookup)

    # -------------------------------------------------------------------
    # PRIORITY FIX: Always prefer explicitly named ID columns if present
    # ------------------------------------------------------------------- 
    if "external_entity_id" in merged_df.columns:
        external_entity_col = "external_entity_id"
    else:
        # id_aliases_entity referenced later in original; build it here
        id_aliases_entity = name_lookup.get("external_entity_id", ["external_entity_id"])
        external_entity_col = detect_id_col_strict(merged_df, {"external_entity_id": id_aliases_entity}, "external_entity_id")

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

    for c in merged_df.columns:
        # only apply when there is at least one list-like cell (cheap check)
        if merged_df[c].apply(lambda x: isinstance(x, (list, tuple, np.ndarray, pd.Series))).any():
            merged_df.loc[:, c] = merged_df[c].apply(flatten_cell)


    # -------------------------
    # Strict name-based ID detection using your JSON name_lookup (no pattern fallback)
    # -------------------------
    id_aliases_entity = name_lookup.get("external_entity_id", ["external_entity_id"])

    external_entity_col = detect_id_col_strict(merged_df, {"external_entity_id": id_aliases_entity}, "external_entity_id")
   
    # -------------------------
    # Basic validation — reject obviously incorrect ID columns (e.g. tiny integers like '1' or single-value columns)
    # -------------------------
    def looks_like_id_column(df, col, min_non_null_ratio=0.2, require_unique=2):
        if not col or col not in df.columns:
            return False
        s = df[col].dropna().astype(str)
        if s.empty:
            return False
        # Reject columns that are overwhelmingly small integers
        numeric_only_ratio = (s.str.match(r'^\d+$').sum()) / len(s)
        if numeric_only_ratio > 0.9:
            return False
        # require at least some fraction of non-null rows and at least a couple unique values
        if (len(s) / len(df)) < min_non_null_ratio:
            return False
        if s.nunique() < require_unique:
            return False
        return True

    if external_entity_col and not looks_like_id_column(merged_df, external_entity_col):
        external_entity_col = None

    # -------------------------
    # Rename to canonical names if we found valid columns
    # -------------------------
    if external_entity_col:
        merged_df.rename(columns={external_entity_col: "external_entity_id"}, inplace=True)
    
    #-----------------------------------------

    # Detect numeric vs string and IDs
    col_types = detect_column_type(merged_df)

    if "external_entity_id" not in merged_df.columns:
        id_aliases_entity = name_lookup.get("external_entity_id", ["external_entity_id"])
        entity_id_col = detect_id_col_strict(
            merged_df, {"external_entity_id": id_aliases_entity}, "external_entity_id"
        )
        if entity_id_col:
            merged_df.rename(columns={entity_id_col: "external_entity_id"}, inplace=True)

    # UUID generation
    entity_id_map = {}

    def get_entity_id(external_id):
        # Only generate UUID if external_id is valid
        if pd.isna(external_id):
            return None
        sval = str(external_id).strip().lower()

        if sval in {"", "na", "n/a", "nan", "null", "none"}:
            return None
        if external_id not in entity_id_map:
            entity_id_map[external_id] = str(uuid.uuid4())
        return entity_id_map[external_id]



    # --- Generate entity_id safely ---
    if "external_entity_id" in merged_df.columns:
        merged_df["entity_id"] = merged_df["external_entity_id"].apply(get_entity_id)
    else:
        merged_df["entity_id"] = None

    # Detect organism from user input (preferred) or filenames (fallback)
    organism_map = load_organism_map()

    if organism is not None:
        # Use organism_map.json to normalize user-provided organism/keyword
        resolved_organism = resolve_organism_from_user_input(organism, organism_map)
    else:
        # Fallback: infer from filenames as before
        resolved_organism = infer_organism_from_filename(input_files, organism_map)

    merged_df["organism"] = resolved_organism
    #-----------------------------------------------------------------------------------

    # Detect & normalize categorical columns
    for col in CATEGORICAL_COLUMNS:
        detected_col = detect_column(merged_df, name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)
        
        # 🔍 If not found, try fuzzy detection using all known aliases
        if not detected_col:
            name_aliases = name_lookup.get(col, [])
            detected_col = detect_column_by_name_fuzzy(merged_df, name_aliases + [col])
            
        # ✅ Validate column content before using it
        if detected_col and col in content_lookup:
            allowed_values = content_lookup[col]
            if not validate_column_content(merged_df, detected_col, allowed_values):
                #print(f"⚠️ Skipping {detected_col} for {col} (content mismatch)")
                detected_col = None
        
        # Proceed only if valid
        if detected_col:
            val = merged_df[detected_col]
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]
            merged_df.loc[:, col] = val
            
            if detected_col != col:
                merged_df.rename(columns={detected_col: col}, inplace=True)
            if col in content_lookup:
                merged_df = normalize_content(merged_df, {col: content_lookup[col]})

    # --------------------
    # Compute date_of_birth
    # --------------------
    # Detect columns dynamically for working out date of birth
    dob_col_days = detect_column(merged_df, "days_to_birth", name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)
    dob_col_year = detect_column(merged_df, "year_of_birth", name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)
    date_of_diag_col = detect_column(merged_df, "diagnosis_date", name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)
    year_of_diag_col = detect_column(merged_df, "year_of_diagnosis", name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)
    sample_date_col = detect_column(merged_df, "collection_date", name_lookup, content_lookup=content_lookup, mapping_overrides=mapping_overrides)

    # Now compute date_of_birth across merged_df (kept in same structure as original)
     
    if "date_of_birth" not in merged_df.columns:
        merged_df["date_of_birth"] = merged_df.apply(
            derive_date_of_birth,
            axis=1,
            date_of_diag_col=date_of_diag_col,
            year_of_diag_col=year_of_diag_col,
            sample_date_col=sample_date_col,
            dob_col_days=dob_col_days,
            dob_col_year=dob_col_year,
        )

   


    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()

    now = datetime.utcnow().isoformat()

    # ---------------- Build Entity DF ---------------- #
    # Ensure all expected columns exist, even if missing in input
    for col in ENTITY_COLUMNS:
        if col not in merged_df.columns:
            merged_df[col] = None  # Fill missing columns with blank/NaN

    entity_df = merged_df[ENTITY_COLUMNS].drop_duplicates(subset=["external_entity_id"])

    # --- Initialize canonical columns ---
    for col in ENTITY_COLUMNS:
        if col not in entity_df.columns:
            entity_df[col] = None

    # --- UUID generation: only if external_entity_id is valid ---
    entity_id_map = {}

    def get_entity_id_safe(external_id):
        if pd.isna(external_id):
            return None
        sval = str(external_id).strip()
        if sval.lower() in {"", "na", "n/a", "nan", "null", "none"}:
            return None
        if external_id not in entity_id_map:
            entity_id_map[external_id] = str(uuid.uuid4())
        return entity_id_map[external_id]

    entity_df["entity_id"] = entity_df["external_entity_id"].apply(get_entity_id_safe)
    #++++++++++++++++++++++++++++++++=
    for key in DEMOGRAPHICS_KEYS:
        col = detect_column(merged_df, key, name_lookup, mapping_overrides)
        #print(f"Key {key!r} detected column:", col)

    ######
    #print("\nDEBUG: sample of days_to_death vs derived date_of_death:")
    #print(merged_df[["external_entity_id", "days_to_death", "date_of_death"]].head(10))

    
    #====================================================================
    # ---------------- Build JSON fields from merged_df (not entity_df) ----------------
    # We build JSONs at the merged_df level (contains raw columns & aliases),
    # then reduce to one row per external_entity_id and merge into entity_df.
    
    # 1) Build per-row JSON series from merged_df (detects aliases & content)
    demographics_series = build_json_field(merged_df, DEMOGRAPHICS_KEYS, name_lookup, content_lookup, "demographics_json")

    clinical_series = build_json_field(merged_df, CLINICAL_PROFILE_KEYS, name_lookup, content_lookup, "clinical_profile_json")
    experimental_series = build_json_field(merged_df, EXPERIMENTAL_PROFILE_KEYS, name_lookup, content_lookup, "experimental_profile_json")

    merged_df["demographics_json"] = demographics_series
    merged_df["clinical_profile_json"] = clinical_series
    merged_df["experimental_profile_json"] = experimental_series
    
    # 2) Make temporary DataFrames with external_entity_id linking
    tmp_demo = pd.DataFrame({
        "external_entity_id": merged_df.get("external_entity_id"),
        "demographics_json": demographics_series
    })
    tmp_clin = pd.DataFrame({
        "external_entity_id": merged_df.get("external_entity_id"),
        "clinical_profile_json": clinical_series
    })
    tmp_exp = pd.DataFrame({
        "external_entity_id": merged_df.get("external_entity_id"),
        "experimental_profile_json": experimental_series
    })

    # 3) Reduce to one row per external_entity_id (keep first non-null)
    def first_non_null_grouped(df, value_col):
        out = (
            df.dropna(subset=["external_entity_id"])
              .groupby("external_entity_id")[value_col]
              .apply(lambda s: next((v for v in s if pd.notna(v) and v not in {None, "{}", ""}), None))
              .reset_index()
        )
        return out

    tmp_demo = first_non_null_grouped(tmp_demo, "demographics_json")
    tmp_clin = first_non_null_grouped(tmp_clin, "clinical_profile_json")
    tmp_exp = first_non_null_grouped(tmp_exp, "experimental_profile_json")

    # 4) Merge these back into entity_df (entity_df has one row per external_entity_id)
    entity_df = entity_df.merge(tmp_demo, on="external_entity_id", how="left")
    entity_df = entity_df.merge(tmp_clin, on="external_entity_id", how="left")
    entity_df = entity_df.merge(tmp_exp, on="external_entity_id", how="left")

    # 5) Ensure columns exist and coerced to the canonical names (they may already)
    for col in ["demographics_json", "clinical_profile_json", "experimental_profile_json"]:
        if col not in entity_df.columns:
            entity_df[col] = None

    #======================================================================
   
    # --- Clinical / Experimental profile JSON depending on organism ---
    def is_human(org_name):
        if org_name is None:
            return True
        return "homo sapiens" in org_name.lower()

    merged_df["is_human"] = merged_df["organism"].apply(is_human)

    def filter_json_by_organism(row):
        if row["is_human"]:
            # Keep clinical, remove experimental
            return row["clinical_profile_json"], None
        else:
            # Keep experimental, remove clinical
            return None, row["experimental_profile_json"]
            
    # --- Determine human vs non-human ---
    merged_df["is_human"] = merged_df["organism"].apply(is_human)

    # --- Apply organism-based JSON filtering ---
    merged_df[["clinical_profile_json", "experimental_profile_json"]] = merged_df.apply(
        lambda r: pd.Series(filter_json_by_organism(r)), axis=1
    )

    # --- Reduce per entity and merge into entity_df ---
    tmp_demo = merged_df[["external_entity_id", "demographics_json"]].drop_duplicates("external_entity_id")
    tmp_clin = merged_df[["external_entity_id", "clinical_profile_json"]].drop_duplicates("external_entity_id")
    tmp_exp = merged_df[["external_entity_id", "experimental_profile_json"]].drop_duplicates("external_entity_id")

    entity_df = entity_df.drop(columns=["demographics_json", "clinical_profile_json", "experimental_profile_json"], errors="ignore")
    entity_df = entity_df.merge(tmp_demo, on="external_entity_id", how="left")
    entity_df = entity_df.merge(tmp_clin, on="external_entity_id", how="left")
    entity_df = entity_df.merge(tmp_exp, on="external_entity_id", how="left")
    
    # --- Timestamps ---
    now = datetime.utcnow().isoformat()
    entity_df["creation_timestamp"] = now
    entity_df["last_updated_timestamp"] = now

    # ✂️ Restrict output columns for entity-only pipeline
    entity_df = entity_df[ENTITY_COLUMNS]

    # ---------------- Save ---------------- #
    entity_df.to_csv(entity_output, index=False)

    print(f"✅ Entity data saved to {entity_output}")
    return entity_output


# ==========================================================
# 🏁 MAIN EXECUTION
# ==========================================================
import sys
import argparse
import logging
import traceback

# ---------------- Safe Execute ---------------- 

def safe_execute(step_name):
    """Decorator for wrapping critical functions with unified logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except SmartError:
                # Custom SmartError is already safe
                raise
            except Exception as e:
                # Unified logging of unexpected exceptions
                logging.error(f"🚨 Smartway HALT: Error in step '{step_name}': {type(e).__name__}: {e}", exc_info=True)
                # Raise SmartError without printing again
                raise SmartError(f"Fatal error in '{step_name}' — stopping execution.") from None
        return wrapper
    return decorator

# ---------------- Main Execution ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest entity files into standardized format."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Paths to input files (CSV, TSV, XLSX). Example: file1.csv file2.tsv"
    )

    parser.add_argument(
        "--external-id-col",
        default=None,
        help="If multiple candidate ID columns exist, use this one."
    )

    parser.add_argument(
        "--organism",
        default=None,
        help="Organism name or keyword (e.g. 'human', 'mouse'); "
             "if omitted, inferred from filenames using organism_map.json."
    )

    parser.add_argument(
        "--os-unit",
        choices=["days", "months"],
        default=None,
        help="Unit of overall survival time in the input files. "
             "If omitted, the script will try to infer it from the OS column name (months vs days)."
    )

    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional entity_mapping.json from the recommender. If provided, it overrides column detection."
    )

    parser.add_argument(
        "--output",
        default="entity_output.csv",
        help="Output CSV filename (default: entity_output.csv)"
    )

    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to write results (no prompt). If omitted, a timestamped folder is created."
    )

    args = parser.parse_args()

    try:
        # Call your ingestion function with CLI files
        out_file = ingest_files(
            args.input_files,
            entity_output=args.output,
            outdir=args.outdir,
            external_id_col=args.external_id_col,
            organism=args.organism,
            os_unit=args.os_unit,  # passes user choice (or None)
            mapping_path=args.mapping
    )
        logging.info(f"✅ Entity data successfully saved to {out_file}")
    except SmartError as e:
        logging.error(f"🛑 SMARTWAY FATAL HALT: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"🛑 UNEXPECTED ERROR: {e}")
        traceback.print_exc(limit=3)
        sys.exit(1)
