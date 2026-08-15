#!/usr/bin/env python
# coding: utf-8

'''


PYTHONPATH=. python onboarding/RNAseq.py TCGA-ACC.star_tpm.tsv \
  --sample-folder tcga_acc1 \
  --mapping onboarding/ACC_RNASEQ_mapping.json

'''

import pandas as pd
import re
import os
import sys
import logging
from datetime import datetime
import json
import glob

try:
    from onboarding.transform_library import TEMPLATES
except ImportError:
    from transform_library import TEMPLATES

def apply_transforms(df: pd.DataFrame, transforms: dict) -> pd.DataFrame:
    if not transforms:
        return df

    # snake_case target -> legacy CamelCase input
    def snake_to_legacy(s: str) -> str:
        parts = s.split("_")
        if parts and parts[0] == "dna":
            parts = ["DNA"] + [p.capitalize() for p in parts[1:]]
        else:
            parts = [p.capitalize() for p in parts]
        return "_".join(parts)

    for target_field, spec in transforms.items():
        template = spec.get("template")
        fn = TEMPLATES.get(template)
        if not fn:
            continue

        inputs = spec.get("inputs", None)
        if inputs is None:
            # default: transform existing target or its legacy name
            if target_field in df.columns:
                inputs = target_field
            else:
                legacy = snake_to_legacy(target_field)
                inputs = legacy if legacy in df.columns else target_field

        params = spec.get("params", {}) or {}
        df[target_field] = fn(df, inputs, params)

    return df
# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class SmartError(Exception):
    """Custom exception for controlled exits in RNA-seq ingestion."""
    pass

# -----------------------------
# 1. RNA-seq schema
# -----------------------------
RNASEQ_COLUMNS = [
    "Sample_ID", "Gene_Symbol", "Entrez_Gene_ID", "Ensembl_ID", "Transcript_ID",
    "Quantification_Method", "Normalization_Method", "Expression_Value", 
    "Total_Reads_Input_for_Quantification", "Total_Reads_Mapped", "Percent_Reads_Mapped",
    "Expression_Matrix_Path_URL", "Log2_Fold_Change", "P_Value", "Adjusted_P_Value",
    "Expression_Status", "Differential_Expression_Method", "Gene_Annotation_Version"
]


# -----------------------------
# Safe exit helper
# -----------------------------
def smart_exit(message, code=1):
    logging.error(message)
    sys.exit(code)
#---------------------------------
def resolve_sample_csv(sample_folder: str) -> str:
    """Return the newest sample_output_*.csv from sample_folder or exit."""
    folder = os.path.abspath(os.path.expanduser(sample_folder))
    if not os.path.isdir(folder):
        logging.error(f"Invalid sample folder: {sample_folder}")
        sys.exit(1)
    matches = glob.glob(os.path.join(folder, "sample_output_*.csv"))
    if not matches:
        logging.error(
            f"No sample_output_*.csv found in {folder}. "
            "Run sample.py first and point --sample-folder to that folder."
        )
        sys.exit(1)
    return max(matches, key=os.path.getmtime)

#--------------------------------
def load_any_table(file_path, nrows=None, low_memory=False, chunksize=None):
    """
    Load CSV, TSV, or XLSX file into a pandas DataFrame or iterator.
    If chunksize is provided (for CSV/TSV), returns a TextFileReader iterator.
    """
    ext = os.path.splitext(file_path.lower())[1]

    if ext in [".csv"]:
        return pd.read_csv(
            file_path,
            sep=",",
            nrows=nrows,
            low_memory=low_memory,
            chunksize=chunksize
        )
    elif ext in [".tsv", ".txt"]:
        return pd.read_csv(
            file_path,
            sep="\t",
            nrows=nrows,
            low_memory=low_memory,
            chunksize=chunksize
        )
    elif ext in [".xlsx", ".xls"]:
        if chunksize is not None:
            # pandas.read_excel does not support chunksize; fall back to full read
            return pd.read_excel(file_path, nrows=nrows)
        return pd.read_excel(file_path, nrows=nrows)
    else:
        smart_exit(f"Unsupported file type: {ext}. Supported: .csv, .tsv, .xlsx")


#-----------------------------------


# -----------------------------
# Column name lookup
# -----------------------------
COLUMN_LOOKUP_FILE = "column_name_lookup.json"

def load_column_lookup(file_path=COLUMN_LOOKUP_FILE):
    """Load the column name lookup JSON file."""
    if not os.path.exists(file_path):
        logging.warning(f"{file_path} not found; using default hardcoded detection.")
        return {}
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load {file_path}: {e}")
        return {}

COLUMN_LOOKUP = load_column_lookup()

def lookup_column(df, key_name):
    """Try to find a column in df using lookup JSON definitions."""
    if not COLUMN_LOOKUP:
        return None
    possible_names = COLUMN_LOOKUP.get(key_name, [])
    for name in possible_names:
        for col in df.columns:
            if col.strip().lower() == name.lower():
                return col
    return None

# -----------------------------
# 2. Detection helpers
# -----------------------------

def detect_column(df, key_name, pattern_check=None):
    """
    Universal column detector using JSON lookup first, then optional pattern check.

    Args:
        df (pd.DataFrame): input dataframe
        key_name (str): key from column_name_lookup.json
        pattern_check (callable): optional function that takes a Series and returns True if it matches
        
    """
    # 1️⃣ Try JSON lookup first
    col = lookup_column(df, key_name)
    if col:
        return col

    # 2️⃣ Try pattern-based fallback if provided
    if pattern_check:
        for c in df.columns:
            series = df[c].dropna().astype(str).head(50)
            try:
                if pattern_check(series):
                    return c
            except Exception:
                continue

    # 4️⃣ Nothing found
    return None


#-----------------------------------
def is_gene_symbol(series):
    regex = re.compile(r"^(?!ENSG)\w{2,10}$", re.IGNORECASE)
    return any(regex.match(v.strip()) for v in series if isinstance(v, str))

def is_ensembl_id(series):
    regex_ensembl = re.compile(r"^ENSG\d+", re.IGNORECASE)
    return any(regex_ensembl.match(v.strip()) for v in series if isinstance(v, str))
    
def is_transcript_id(series):
    regex_transcript = re.compile(r"^ENST\d+", re.IGNORECASE)
    return any(regex_transcript.match(v.strip()) for v in series if isinstance(v, str))
    
def detect_gene_column(df):
    return detect_column(df, "Gene_Symbol", pattern_check=is_gene_symbol)

def detect_ensembl_id(df):
    return detect_column(df, "Ensembl_ID", pattern_check=is_ensembl_id)

def detect_transcript_id(df):
    return detect_column(df, "Transcript_ID", pattern_check=is_transcript_id)

def is_entrez_id(series):
    regex_entrez = re.compile(r"^\d+$")
    return any(regex_entrez.match(v.strip()) for v in series if isinstance(v, str))

def detect_entrez_id(df):
    return detect_column(df, "Entrez_Gene_ID", pattern_check=is_entrez_id)


CONFIG_FILE = "quantification_method.json"
DE_METHODS_FILE = "de_methods.json"

def load_quantification_methods(config_file=CONFIG_FILE):
    """Load quantification methods from JSON file, or return empty list if missing."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found. Please create it first.")
    with open(config_file, "r") as f:
        config = json.load(f)
    return config.get("quantification_methods", [])


def detect_quantification_method(df, file_path=None, config_file=CONFIG_FILE):
    # 1️⃣ Try column name–based detection first
    col = detect_column(df, "Quantification_Method")
    if col:
        return col

    # 2️⃣ Load known quantification tools from config
    tools = load_quantification_methods(config_file)
    if not tools:
        raise ValueError(f"No quantification methods found in {config_file}")

    regex = re.compile(r"^(" + "|".join(re.escape(tool) for tool in tools) + r")$", re.IGNORECASE)

    # 3️⃣ Check dataframe values for known tool names
    for c in df.columns:
        sample_vals = df[c].dropna().astype(str).head(50)
        if any(regex.search(v) for v in sample_vals):
            return c

    # 4️⃣ Check filename for tool hint
    if file_path:
        fname = os.path.basename(file_path).lower()
        for tool in tools:
            if tool.lower() in fname:
                return tool.capitalize() if tool != "htseq" else "HTSeq"

    return None


def detect_normalization_method(df, file_path=None):
    # Try column name–based detection first
    col = detect_column(df, "Normalization_Method")
    if col:
        return col
        
    regex = re.compile(r"^(tpm|fpkm|rpkm|counts)$", re.IGNORECASE)
    for c in df.columns:
        sample_vals = df[c].dropna().astype(str).head(50)
        if any(regex.search(v) for v in sample_vals):
            return c
            
    if file_path:
        fname = os.path.basename(file_path).lower()
        if "tpm" in fname:
            return "TPM"
        elif "fpkm" in fname:
            return "FPKM"
        elif "rpkm" in fname:
            return "RPKM"
        elif "count" in fname:
            return "Counts"
    return None

def is_expression_matrix_path(series):
    regex_transcript = re.compile(r"(http|ftp|/).+", re.IGNORECASE)
    return any(regex_transcript.match(v.strip()) for v in series if isinstance(v, str))

def detect_expression_matrix_path(df):
    return detect_column(df, "Expression_Matrix_Path_URL", pattern_check=is_expression_matrix_path)
    


def load_de_methods(config_file=DE_METHODS_FILE):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found. Please create it with your DE methods.")

    with open(config_file, "r") as f:
        config = json.load(f)

    de_methods = config.get("de_methods")
    if not de_methods:
        raise ValueError(f"Expected key 'de_methods' in {config_file}, but it was missing or empty.")
    return de_methods



def detect_de_method(df, config_file=DE_METHODS_FILE):
    # Try column name–based detection first
    col = detect_column(df, "Differential_Expression_Method")
    if col:
        return col
    
    """Detect DE method in dataframe using regex from JSON config."""
    methods = load_de_methods(config_file)
    regex = re.compile(r"(" + "|".join(re.escape(m) for m in methods) + r")", re.IGNORECASE)

    for c in df.columns:
        sample_vals = df[c].dropna().astype(str).head(50)
        if any(regex.search(v) for v in sample_vals):
            return c
    return None

def detect_annotation_version(df):
    # Try column name–based detection first
    col = detect_column(df, "Gene_Annotation_Version")
    if col:
        return col
    regex = re.compile(r"(gencode|ensembl|refseq).*[0-9]+", re.IGNORECASE)
    for c in df.columns:
        sample_vals = df[c].dropna().astype(str).head(50)
        if any(regex.search(v) for v in sample_vals):
            return c
    return None

# -----------------------------
# Format helpers
# -----------------------------
def is_wide_format(df):
    # Step 1: Detect identifier column
    gene_col = detect_gene_column(df)
    ensembl_col = detect_ensembl_id(df)
    transcript_col = detect_transcript_id(df)
    entrez_col = detect_entrez_id(df)
    
    id_col = gene_col or ensembl_col or transcript_col or entrez_col
    if not id_col:
        return False  # Can't determine without identifier

    # Step 2: Examine remaining columns
    value_cols = [c for c in df.columns if c != id_col]
    numeric_counts = sum(
        pd.to_numeric(df[c].dropna().head(10), errors='coerce').notnull().sum() > 0
        for c in value_cols
    )
    
    # Step 3: Heuristic decision
    # If more than 1 numeric column (excluding ID), treat as wide
    return numeric_counts > 1


def _is_missing(val):
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False

#-----------------------------------
def _resolve_scalar_from_df(df, key_or_value):
    """
    Given either a column name or a scalar, return a scalar value.
    - If key_or_value is a column name present in df: use df[col].iloc[0]
    - Else, if key_or_value is a non-empty string: return as-is
    - Else return None.
    """
    if not key_or_value:
        return None
    if isinstance(key_or_value, str) and key_or_value in df.columns:
        if df.shape[0] == 0:
            return None
        return df[key_or_value].iloc[0]
    return key_or_value
#-----------------------------------
def build_rnaseq_mapping_plan(df_sample, file_path):
    """
    Inspect a small sample dataframe and decide:
      - format_type: 'entrez_only', 'wide', or 'long'
      - identifier columns
      - metadata/statistics columns
      - per-file scalar metadata values
    This plan is reused for each chunk.
    """
    if df_sample is None or df_sample.empty:
        smart_exit(f"Sample from {file_path} is empty; nothing to ingest.")

    # Special-case: TCGA-style Entrez-only matrix
    first_col = str(df_sample.columns[0]).strip().lower()
    if first_col == "entrez_gene_id":
        entrez_col = df_sample.columns[0]

        # Detect metadata/stat columns from the sample
        logfc_col   = detect_column(df_sample, "Log2_Fold_Change")
        pval_col    = detect_column(df_sample, "P_Value")
        padj_col    = detect_column(df_sample, "Adjusted_P_Value")
        status_col  = detect_column(df_sample, "Expression_Status")
        quant_col_or_val  = detect_quantification_method(df_sample, file_path)
        norm_col_or_val   = detect_normalization_method(df_sample, file_path)
        de_col_or_val     = detect_de_method(df_sample)
        anno_col_or_val   = detect_annotation_version(df_sample)
        total_in_col      = detect_column(df_sample, "Total_Reads_Input_for_Quantification")
        total_mapped_col  = detect_column(df_sample, "Total_Reads_Mapped")
        percent_mapped_col= detect_column(df_sample, "Percent_Reads_Mapped")
        expr_matrix_col   = detect_expression_matrix_path(df_sample)

        plan = {
            "format_type": "entrez_only",
            "gene_col": None,
            "ensembl_col": None,
            "transcript_col": None,
            "entrez_col": entrez_col,
            "logfc_col": logfc_col,
            "pval_col": pval_col,
            "padj_col": padj_col,
            "status_col": status_col,
            # Per-file scalar metadata
            "quant_value": _resolve_scalar_from_df(df_sample, quant_col_or_val),
            "norm_value": _resolve_scalar_from_df(df_sample, norm_col_or_val),
            "de_value": _resolve_scalar_from_df(df_sample, de_col_or_val),
            "anno_value": _resolve_scalar_from_df(df_sample, anno_col_or_val),
            "total_in_value": _resolve_scalar_from_df(df_sample, total_in_col),
            "total_mapped_value": _resolve_scalar_from_df(df_sample, total_mapped_col),
            "percent_mapped_value": _resolve_scalar_from_df(df_sample, percent_mapped_col),
            "expr_matrix_value": _resolve_scalar_from_df(df_sample, expr_matrix_col),
        }
        return plan

    # General case: detect identifier columns
    gene_col       = detect_gene_column(df_sample)
    ensembl_col    = detect_ensembl_id(df_sample)
    transcript_col = detect_transcript_id(df_sample)
    entrez_col     = detect_entrez_id(df_sample)

    if not (gene_col or ensembl_col or transcript_col or entrez_col):
        smart_exit("Could not detect any identifier column (Gene/Ensembl/Transcript/Entrez).")

    # If gene and ensembl are the same column, prefer one
    if gene_col and ensembl_col and gene_col == ensembl_col:
        logging.warning("Gene and Ensembl columns appear identical, dropping Gene column.")
        gene_col = None

    # Determine wide vs long from the sample
    wide = is_wide_format(df_sample)
    format_type = "wide" if wide else "long"

    # Detect metadata/stat columns using the sample
    logfc_col   = detect_column(df_sample, "Log2_Fold_Change")
    pval_col    = detect_column(df_sample, "P_Value")
    padj_col    = detect_column(df_sample, "Adjusted_P_Value")
    status_col  = detect_column(df_sample, "Expression_Status")
    quant_col_or_val  = detect_quantification_method(df_sample, file_path)
    norm_col_or_val   = detect_normalization_method(df_sample, file_path)
    de_col_or_val     = detect_de_method(df_sample)
    anno_col_or_val   = detect_annotation_version(df_sample)
    total_in_col      = detect_column(df_sample, "Total_Reads_Input_for_Quantification")
    total_mapped_col  = detect_column(df_sample, "Total_Reads_Mapped")
    percent_mapped_col= detect_column(df_sample, "Percent_Reads_Mapped")
    expr_matrix_col   = detect_expression_matrix_path(df_sample)

    plan = {
        "format_type": format_type,
        "gene_col": gene_col,
        "ensembl_col": ensembl_col,
        "transcript_col": transcript_col,
        "entrez_col": entrez_col,
        "logfc_col": logfc_col,
        "pval_col": pval_col,
        "padj_col": padj_col,
        "status_col": status_col,
        # Per-file scalar metadata
        "quant_value": _resolve_scalar_from_df(df_sample, quant_col_or_val),
        "norm_value": _resolve_scalar_from_df(df_sample, norm_col_or_val),
        "de_value": _resolve_scalar_from_df(df_sample, de_col_or_val),
        "anno_value": _resolve_scalar_from_df(df_sample, anno_col_or_val),
        "total_in_value": _resolve_scalar_from_df(df_sample, total_in_col),
        "total_mapped_value": _resolve_scalar_from_df(df_sample, total_mapped_col),
        "percent_mapped_value": _resolve_scalar_from_df(df_sample, percent_mapped_col),
        "expr_matrix_value": _resolve_scalar_from_df(df_sample, expr_matrix_col),
    }
    logging.info(f"Detected RNA-seq format: {format_type} (gene={gene_col}, ensembl={ensembl_col}, transcript={transcript_col}, entrez={entrez_col})")
    return plan

#------------------------------
def _map_entrez_only_chunk(df_chunk, plan):
    """
    Chunked version of _do_mapping_entrez_only.
    df_chunk: a subset of rows, with the same columns as the full file.
    """
    entrez_col = plan["entrez_col"]
    mapped_df = pd.DataFrame(columns=RNASEQ_COLUMNS)

    value_vars = [c for c in df_chunk.columns if c != entrez_col]
    if not value_vars:
        raise ValueError("No expression value columns detected in wide Entrez-only format.")

    df_long = pd.melt(
        df_chunk,
        id_vars=[entrez_col],
        value_vars=value_vars,
        var_name="Sample_ID",
        value_name="Expression_Value"
    )

    mapped_df["Sample_ID"]       = df_long["Sample_ID"]
    mapped_df["Expression_Value"] = df_long["Expression_Value"]
    mapped_df["Entrez_Gene_ID"]  = df_long[entrez_col]

    # Metadata: use per-file scalar values computed in the plan
    if plan["quant_value"] is not None:
        mapped_df["Quantification_Method"] = plan["quant_value"]
    if plan["norm_value"] is not None:
        mapped_df["Normalization_Method"] = plan["norm_value"]
    if plan["de_value"] is not None:
        mapped_df["Differential_Expression_Method"] = plan["de_value"]
    if plan["anno_value"] is not None:
        mapped_df["Gene_Annotation_Version"] = plan["anno_value"]

    if plan["total_in_value"] is not None:
        mapped_df["Total_Reads_Input_for_Quantification"] = plan["total_in_value"]
    if plan["total_mapped_value"] is not None:
        mapped_df["Total_Reads_Mapped"] = plan["total_mapped_value"]
    if plan["percent_mapped_value"] is not None:
        mapped_df["Percent_Reads_Mapped"] = plan["percent_mapped_value"]
    if plan["expr_matrix_value"] is not None:
        mapped_df["Expression_Matrix_Path_URL"] = plan["expr_matrix_value"]

    # Expression_Status: original logic calls _determine_status when missing
    mapped_df["Expression_Status"] = mapped_df.apply(_determine_status, axis=1)

    return mapped_df.reindex(columns=RNASEQ_COLUMNS)


def _map_wide_chunk(df_chunk, plan):
    """
    Chunked mapping for wide format (general case, similar to _do_mapping wide branch).
    """
    mapped_df = pd.DataFrame(columns=RNASEQ_COLUMNS)

    id_vars = []
    if plan["gene_col"]:
        id_vars.append(plan["gene_col"])
    if plan["ensembl_col"]:
        id_vars.append(plan["ensembl_col"])
    if plan["transcript_col"]:
        id_vars.append(plan["transcript_col"])
    if plan["entrez_col"]:
        id_vars.append(plan["entrez_col"])
    if not id_vars:
        raise ValueError("Could not detect any identifier column (Gene/Ensembl/Transcript/Entrez).")

    value_vars = [c for c in df_chunk.columns if c not in id_vars]
    if not value_vars:
        raise ValueError("No expression value columns detected in wide format chunk.")

    df_long = pd.melt(
        df_chunk,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="Sample_ID",
        value_name="Expression_Value"
    )

    mapped_df["Sample_ID"]       = df_long["Sample_ID"]
    mapped_df["Expression_Value"] = df_long["Expression_Value"]

    if plan["gene_col"] and plan["gene_col"] in df_long.columns:
        mapped_df["Gene_Symbol"] = df_long[plan["gene_col"]]
    if plan["ensembl_col"] and plan["ensembl_col"] in df_long.columns:
        mapped_df["Ensembl_ID"] = df_long[plan["ensembl_col"]]
    if plan["transcript_col"] and plan["transcript_col"] in df_long.columns:
        mapped_df["Transcript_ID"] = df_long[plan["transcript_col"]]
    if plan["entrez_col"] and plan["entrez_col"] in df_long.columns:
        mapped_df["Entrez_Gene_ID"] = df_long[plan["entrez_col"]]

    # Metadata: same per-file scalar behaviour
    if plan["quant_value"] is not None:
        mapped_df["Quantification_Method"] = plan["quant_value"]
    if plan["norm_value"] is not None:
        mapped_df["Normalization_Method"] = plan["norm_value"]
    if plan["de_value"] is not None:
        mapped_df["Differential_Expression_Method"] = plan["de_value"]
    if plan["anno_value"] is not None:
        mapped_df["Gene_Annotation_Version"] = plan["anno_value"]

    if plan["total_in_value"] is not None:
        mapped_df["Total_Reads_Input_for_Quantification"] = plan["total_in_value"]
    if plan["total_mapped_value"] is not None:
        mapped_df["Total_Reads_Mapped"] = plan["total_mapped_value"]
    if plan["percent_mapped_value"] is not None:
        mapped_df["Percent_Reads_Mapped"] = plan["percent_mapped_value"]
    if plan["expr_matrix_value"] is not None:
        mapped_df["Expression_Matrix_Path_URL"] = plan["expr_matrix_value"]

    # Expression_Status: recompute if missing
    if "Expression_Status" not in mapped_df.columns or mapped_df["Expression_Status"].isnull().all():
        mapped_df["Expression_Status"] = mapped_df.apply(_determine_status, axis=1)

    return mapped_df.reindex(columns=RNASEQ_COLUMNS)


def _map_long_chunk(df_chunk, plan):
    """
    Chunked mapping for long format (similar to _do_mapping long branch).
    """
    df_long = df_chunk.copy()
    mapped_df = pd.DataFrame(columns=RNASEQ_COLUMNS)

    if "Sample_ID" in df_long.columns:
        mapped_df["Sample_ID"] = df_long["Sample_ID"]

    if plan["gene_col"] and plan["gene_col"] in df_long.columns:
        mapped_df["Gene_Symbol"] = df_long[plan["gene_col"]]
    if plan["ensembl_col"] and plan["ensembl_col"] in df_long.columns:
        mapped_df["Ensembl_ID"] = df_long[plan["ensembl_col"]]
    if plan["transcript_col"] and plan["transcript_col"] in df_long.columns:
        mapped_df["Transcript_ID"] = df_long[plan["transcript_col"]]
    if plan["entrez_col"] and plan["entrez_col"] in df_long.columns:
        mapped_df["Entrez_Gene_ID"] = df_long[plan["entrez_col"]]

    # Expression and statistics columns
    if "Expression_Value" in df_long.columns:
        mapped_df["Expression_Value"] = df_long["Expression_Value"]
    if plan["logfc_col"] and plan["logfc_col"] in df_long.columns:
        mapped_df["Log2_Fold_Change"] = df_long[plan["logfc_col"]]
    if plan["pval_col"] and plan["pval_col"] in df_long.columns:
        mapped_df["P_Value"] = df_long[plan["pval_col"]]
    if plan["padj_col"] and plan["padj_col"] in df_long.columns:
        mapped_df["Adjusted_P_Value"] = df_long[plan["padj_col"]]
    if plan["status_col"] and plan["status_col"] in df_long.columns:
        mapped_df["Expression_Status"] = df_long[plan["status_col"]]

    # Per-file metadata
    if plan["quant_value"] is not None:
        mapped_df["Quantification_Method"] = plan["quant_value"]
    if plan["norm_value"] is not None:
        mapped_df["Normalization_Method"] = plan["norm_value"]
    if plan["de_value"] is not None:
        mapped_df["Differential_Expression_Method"] = plan["de_value"]
    if plan["anno_value"] is not None:
        mapped_df["Gene_Annotation_Version"] = plan["anno_value"]

    if plan["total_in_value"] is not None:
        mapped_df["Total_Reads_Input_for_Quantification"] = plan["total_in_value"]
    if plan["total_mapped_value"] is not None:
        mapped_df["Total_Reads_Mapped"] = plan["total_mapped_value"]
    if plan["percent_mapped_value"] is not None:
        mapped_df["Percent_Reads_Mapped"] = plan["percent_mapped_value"]
    if plan["expr_matrix_value"] is not None:
        mapped_df["Expression_Matrix_Path_URL"] = plan["expr_matrix_value"]

    # Recompute Expression_Status if missing
    if "Expression_Status" not in mapped_df.columns or mapped_df["Expression_Status"].isnull().all():
        mapped_df["Expression_Status"] = mapped_df.apply(_determine_status, axis=1)

    return mapped_df.reindex(columns=RNASEQ_COLUMNS)
#------------------------------------
def iter_rnaseq_mapped_chunks(file_path, batch_size=1000, mapping_json=None):
    """
    Yield mapped RNA-seq chunks conforming to RNASEQ_COLUMNS, without
    loading the full file into memory.
    For Excel inputs, fall back to one-shot mapping (file assumed small).
    """
    ext = os.path.splitext(file_path.lower())[1]

    # Excel: no chunksize support in pandas → use existing one-shot logic
    if ext in [".xlsx", ".xls"]:
        df_full = load_any_table(file_path, low_memory=False)
        if df_full is None or df_full.empty:
            smart_exit(f"Input file {file_path} is empty after parsing.")
        rnaseq_df = map_rnaseq(df_full, file_path)
        rnaseq_df = validate_rnaseq_df(rnaseq_df)
        yield rnaseq_df
        return

    # CSV/TSV streaming path
    try:
        df_sample = load_any_table(file_path, nrows=batch_size, low_memory=False)
    except pd.errors.EmptyDataError:
        smart_exit(f"No data could be read from input file: {file_path}")
    except pd.errors.ParserError as e:
        smart_exit(f"Parsing error in input file {file_path}: {e}")
    except Exception as e:
        smart_exit(f"Unexpected error reading file {file_path}: {e}")

    if df_sample is None or df_sample.empty:
        smart_exit(f"Input file {file_path} is empty after parsing.")

    plan = build_rnaseq_mapping_plan(df_sample, file_path)

    try:
        reader = load_any_table(file_path, low_memory=False, chunksize=batch_size)
    except Exception as e:
        smart_exit(f"Failed to create chunked reader for {file_path}: {e}")

    for i, df_chunk in enumerate(reader):
        if df_chunk is None or df_chunk.empty:
            continue

        try:
            if plan["format_type"] == "entrez_only":
                mapped = _map_entrez_only_chunk(df_chunk, plan)
            elif plan["format_type"] == "wide":
                mapped = _map_wide_chunk(df_chunk, plan)
            else:
                mapped = _map_long_chunk(df_chunk, plan)
        except Exception as e:
            smart_exit(f"Error mapping RNA-seq chunk {i} of {file_path}: {e}")

        if mapped is None or mapped.empty:
            continue

        # Validate each mapped chunk (same rules as before, just chunked)
        mapped = apply_transforms(mapped, mapping_json.get("transforms", {}) if mapping_json else {})
        mapped = validate_rnaseq_df(mapped)
        yield mapped

# -----------------------------
# 4. RNA-seq mapping
# -----------------------------
def map_rnaseq(df, file_path=None):
    if df is None or df.empty:
        smart_exit("Input DataFrame is empty or invalid.")

    mapped_df = pd.DataFrame(columns=RNASEQ_COLUMNS)
    #-------------------------------------------------------

    # 🔹 Special-case: TCGA FPKM matrix with Entrez_Gene_Id as first column
    first_col = str(df.columns[0]).strip().lower()
    if first_col == "entrez_gene_id" or first_col == "entrez_gene_id":
        entrez_col = df.columns[0]
        print(f"Using Entrez-only mapper on column: {entrez_col}")
        try:
            return _do_mapping_entrez_only(df, entrez_col, file_path)
        except Exception as e:
            smart_exit(f"Mapping failed (Entrez-only handler): {e}")



    #-------------------------------------------------------

    gene_col = detect_gene_column(df)
    ensembl_col = detect_ensembl_id(df)
    transcript_col = detect_transcript_id(df)
    entrez_col = detect_entrez_id(df)

    if not (gene_col or ensembl_col or transcript_col or entrez_col):  # <-- aligned properly
        smart_exit("Could not detect any identifier column (Gene/Ensembl/Transcript/Entrez).")

    # Special-case: TCGA-style matrix with ONLY Entrez as ID
    if not (gene_col or ensembl_col or transcript_col) and entrez_col:
        try:
            rnaseq_mapped = _do_mapping_entrez_only(df, entrez_col, file_path)
        except Exception as e:
            smart_exit(f"Mapping failed (Entrez-only handler): {e}")
        return rnaseq_mapped


    # Ensure not overlapping incorrectly
    if gene_col and ensembl_col and gene_col == ensembl_col:
        logging.warning("Gene and Ensembl columns appear identical, dropping Gene column.")
        gene_col = None

    # Use your original mapping logic
    try:
        rnaseq_mapped = _do_mapping(df, gene_col, ensembl_col, transcript_col, entrez_col, file_path)
    except Exception as e:
        smart_exit(f"Mapping failed: {e}")

    return rnaseq_mapped

#-----------------------------------------------------------------------------
def _do_mapping_entrez_only(df, entrez_col, file_path):
    """
    Special-case mapping for wide matrices with only Entrez_Gene_Id + sample columns.
    """
    mapped_df = pd.DataFrame(columns=RNASEQ_COLUMNS)

    # All columns except Entrez are assumed to be sample expression columns
    value_vars = [c for c in df.columns if c != entrez_col]
    if not value_vars:
        raise ValueError("No expression value columns detected in wide Entrez-only format.")

    df_long = pd.melt(
        df,
        id_vars=[entrez_col],
        value_vars=value_vars,
        var_name="Sample_ID",
        value_name="Expression_Value"
    )

    # Core schema
    mapped_df["Sample_ID"] = df_long["Sample_ID"]
    mapped_df["Expression_Value"] = df_long["Expression_Value"]
    mapped_df["Entrez_Gene_ID"] = df_long[entrez_col]

    # No Gene_Symbol / Ensembl / Transcript in this file → leave as NaN/empty

    # Detect metadata as usual (constants per file)
    logfc_col = detect_column(df, "Log2_Fold_Change")
    pval_col = detect_column(df, "P_Value")
    padj_col = detect_column(df, "Adjusted_P_Value")
    status_col = detect_column(df, "Expression_Status")
    quant_method = detect_quantification_method(df, file_path)
    norm_method = detect_normalization_method(df, file_path)
    de_method = detect_de_method(df)
    anno_ver = detect_annotation_version(df)
    total_in_col = detect_column(df, "Total_Reads_Input_for_Quantification")
    total_mapped_col = detect_column(df, "Total_Reads_Mapped")
    percent_mapped_col = detect_column(df, "Percent_Reads_Mapped")
    expr_matrix_col = detect_expression_matrix_path(df)

    # Fill metadata (single values copied to all rows)
    if quant_method and quant_method in df.columns:
        mapped_df["Quantification_Method"] = df[quant_method].iloc[0]
    elif quant_method:
        mapped_df["Quantification_Method"] = quant_method

    if norm_method and norm_method in df.columns:
        mapped_df["Normalization_Method"] = df[norm_method].iloc[0]
    elif norm_method:
        mapped_df["Normalization_Method"] = norm_method

    if de_method and de_method in df.columns:
        mapped_df["Differential_Expression_Method"] = df[de_method].iloc[0]
    elif de_method:
        mapped_df["Differential_Expression_Method"] = de_method

    if anno_ver and anno_ver in df.columns:
        mapped_df["Gene_Annotation_Version"] = df[anno_ver].iloc[0]
    elif anno_ver:
        mapped_df["Gene_Annotation_Version"] = anno_ver

    if total_in_col and total_in_col in df.columns:
        mapped_df["Total_Reads_Input_for_Quantification"] = df[total_in_col].iloc[0]
    if total_mapped_col and total_mapped_col in df.columns:
        mapped_df["Total_Reads_Mapped"] = df[total_mapped_col].iloc[0]
    if percent_mapped_col and percent_mapped_col in df.columns:
        mapped_df["Percent_Reads_Mapped"] = df[percent_mapped_col].iloc[0]
    if expr_matrix_col and expr_matrix_col in df.columns:
        mapped_df["Expression_Matrix_Path_URL"] = df[expr_matrix_col].iloc[0]

    # Expression_Status (if present in original long-format columns — unlikely here)
    if status_col and status_col in df.columns:
        mapped_df["Expression_Status"] = df[status_col].iloc[0]
    else:
        # Let the generic status logic run
        mapped_df["Expression_Status"] = mapped_df.apply(
            lambda row: _determine_status(row), axis=1
        )

    return mapped_df.reindex(columns=RNASEQ_COLUMNS)


#------------------------------------------------------------------------------
def _do_mapping(df, gene_col, ensembl_col, transcript_col, entrez_col, file_path):
    # ✅ Initialize mapped_df with schema columns
    mapped_df = pd.DataFrame(columns=RNASEQ_COLUMNS)

    # Collect identifier columns
    id_vars = []
    if gene_col:
        id_vars.append(gene_col)
    if ensembl_col:
        id_vars.append(ensembl_col)
    if transcript_col:
        id_vars.append(transcript_col)
    if entrez_col:
        id_vars.append(entrez_col)
    if not id_vars:
        raise ValueError("Could not detect any identifier column (Gene/Ensembl/Transcript/Entrez).")

    # Detect additional metadata/statistical columns
    logfc_col = detect_column(df, "Log2_Fold_Change")
    pval_col = detect_column(df, "P_Value")
    padj_col = detect_column(df, "Adjusted_P_Value")
    status_col = detect_column(df, "Expression_Status")
    
    quant_method = detect_quantification_method(df, file_path)
    norm_method = detect_normalization_method(df, file_path)
    de_method = detect_de_method(df)
    anno_ver = detect_annotation_version(df)
    total_in_col = detect_column(df, "Total_Reads_Input_for_Quantification")
    total_mapped_col = detect_column(df, "Total_Reads_Mapped")
    percent_mapped_col = detect_column(df, "Percent_Reads_Mapped")
    expr_matrix_col = detect_expression_matrix_path(df)

    # Wide format → melt into long
    if is_wide_format(df):
        print("Detected wide format → melting to long format")
        value_vars = [c for c in df.columns if c not in id_vars]
        if not value_vars:
            raise ValueError("No expression value columns detected in wide format.")
            
        df_long = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="Sample_ID",
            value_name="Expression_Value"
        )
        mapped_df["Sample_ID"] = df_long["Sample_ID"]
        mapped_df["Expression_Value"] = df_long["Expression_Value"]
        
        if gene_col and gene_col in df_long.columns:
            mapped_df["Gene_Symbol"] = df_long[gene_col]
        if ensembl_col and ensembl_col in df_long.columns:
            mapped_df["Ensembl_ID"] = df_long[ensembl_col]
        if transcript_col and transcript_col in df_long.columns:
            mapped_df["Transcript_ID"] = df_long[transcript_col]
        if entrez_col and entrez_col in df_long.columns:
            mapped_df["Entrez_Gene_ID"] = df_long[entrez_col]
        elif entrez_col and entrez_col not in df_long.columns:
            logging.warning(
                f"Detected Entrez ID column '{entrez_col}' not present after melting; "
                "Entrez_Gene_ID will be left empty."
            )
        
    else:
        print("Detected long format → using directly")
        df_long = df.copy()
        
        if "Sample_ID" in df_long.columns:
            mapped_df["Sample_ID"] = df_long["Sample_ID"]
        
        if gene_col and gene_col in df_long.columns:
            mapped_df["Gene_Symbol"] = df_long[gene_col]
        if ensembl_col and ensembl_col in df_long.columns:
            mapped_df["Ensembl_ID"] = df_long[ensembl_col]
        if transcript_col and transcript_col in df_long.columns:
            mapped_df["Transcript_ID"] = df_long[transcript_col]
        if entrez_col and entrez_col in df_long.columns:
            mapped_df["Entrez_Gene_ID"] = df_long[entrez_col]

        # Populate expression values/statistics
        if "Expression_Value" in df_long.columns:
            mapped_df["Expression_Value"] = df_long["Expression_Value"]
        if logfc_col and logfc_col in df_long.columns:
            mapped_df["Log2_Fold_Change"] = df_long[logfc_col]
        if pval_col and pval_col in df_long.columns:
            mapped_df["P_Value"] = df_long[pval_col]
        if padj_col and padj_col in df_long.columns:
            mapped_df["Adjusted_P_Value"] = df_long[padj_col]
        if status_col and status_col in df_long.columns:
            mapped_df["Expression_Status"] = df_long[status_col]

    # Populate metadata columns (constants for all rows)
    if quant_method and quant_method in df.columns:
        mapped_df["Quantification_Method"] = df[quant_method].iloc[0]
    elif quant_method:
        mapped_df["Quantification_Method"] = quant_method

    if norm_method and norm_method in df.columns:
        mapped_df["Normalization_Method"] = df[norm_method].iloc[0]
    elif norm_method:
        mapped_df["Normalization_Method"] = norm_method

    if de_method and de_method in df.columns:
        mapped_df["Differential_Expression_Method"] = df[de_method].iloc[0]
    elif de_method:
        mapped_df["Differential_Expression_Method"] = de_method
        
    if anno_ver and anno_ver in df.columns:
        mapped_df["Gene_Annotation_Version"] = df[anno_ver].iloc[0]
    elif anno_ver:
        mapped_df["Gene_Annotation_Version"] = anno_ver

    

    if total_in_col and total_in_col in df.columns:
        mapped_df["Total_Reads_Input_for_Quantification"] = df[total_in_col].iloc[0]
    if total_mapped_col and total_mapped_col in df.columns:
        mapped_df["Total_Reads_Mapped"] = df[total_mapped_col].iloc[0]
    if percent_mapped_col and percent_mapped_col in df.columns:
        mapped_df["Percent_Reads_Mapped"] = df[percent_mapped_col].iloc[0]
    if expr_matrix_col and expr_matrix_col in df.columns:
        mapped_df["Expression_Matrix_Path_URL"] = df[expr_matrix_col].iloc[0]

    # Recompute Expression_Status if missing
    if "Expression_Status" not in mapped_df.columns or mapped_df["Expression_Status"].isnull().all():
        mapped_df["Expression_Status"] = mapped_df.apply(
            lambda row: _determine_status(row), axis=1
        )
        
    # ✅ Ensure all schema columns exist and in correct order
    return mapped_df.reindex(columns=RNASEQ_COLUMNS)

########################################################
def _determine_status(row, alpha=0.05, lfc_thresh=1.0):
    import math
    
    logfc_val = row.get("Log2_Fold_Change")
    pval_val = row.get("P_Value")
    padj_val = row.get("Adjusted_P_Value")

    def _to_float(val):
        if val in ["", "NA", "NaN", None]:
            return None

        try:
            return float(str(val).strip())
        except (ValueError, TypeError):
            return None

    logfc = _to_float(logfc_val)
    pval = _to_float(pval_val)
    padj = _to_float(padj_val)

    # Treat None or NaN as missing
    logfc_is_missing = logfc is None or (isinstance(logfc, float) and math.isnan(logfc))
    pval_is_missing = pval is None or (isinstance(pval, float) and math.isnan(pval))
    padj_is_missing = padj is None or (isinstance(padj, float) and math.isnan(padj))

    if logfc_is_missing and pval_is_missing and padj_is_missing:
        return None  # empty cell

    # Choose which p-value to use
    p_use = padj if not padj_is_missing else (pval if not pval_is_missing else None)
    if p_use is None:
        return None  # treat as missing

    # Determine significance and direction
    if p_use < alpha:
        if not logfc_is_missing:
            if logfc > lfc_thresh:
                return "Upregulated"
            elif logfc < -lfc_thresh:
                return "Downregulated"
            else:
                return "Significant (Low Fold Change)"
        else:
            return "Significant (No LogFC)"
    else:
        return "Not Significant"

# -----------------------------
# Validation with summary report
# -----------------------------
def validate_rnaseq_df(df):
    """Perform sanity checks on the mapped RNA-seq dataframe with a summary report."""

    issues = []

    if df.empty:
        issues.append(("CRITICAL", "Mapped dataframe is empty."))

    # Sample_ID check
    if df["Sample_ID"].isnull().any() or (df["Sample_ID"].astype(str).str.strip() == "").any():
        issues.append(("CRITICAL", "Some rows have missing or empty Sample_ID values."))

        # Expression_Value must be numeric and >= 0
    try:
        raw = df["Expression_Value"]

        # Coerce to numeric; non-numeric → NaN
        expr = pd.to_numeric(raw, errors="coerce")

        # Treat common "missing" tokens as acceptable (not an error)
        raw_str = raw.astype(str).str.strip()
        missing_like = raw_str.isin(["", "NA", "NaN", "nan", "N/A", "null", "NULL"])

        # Real (non-missing) cells where conversion failed
        bad_mask = (~missing_like) & expr.isnull()
        if bad_mask.any():
            issues.append((
                "WARNING",
                "Expression_Value has non-numeric values in non-empty cells (treated as missing)."
            ))

        # Negative values are still a hard error
        if (expr[~expr.isnull()] < 0).any():
            issues.append(("CRITICAL", "Expression_Value contains negative values."))

        # Optionally store the coerced numeric values back
        df["Expression_Value"] = expr

    except Exception:
        issues.append(("CRITICAL", "Expression_Value column could not be validated."))


    # Log2 Fold Change sanity check
    if "Log2_Fold_Change" in df.columns and not df["Log2_Fold_Change"].isnull().all():
        logfc = pd.to_numeric(df["Log2_Fold_Change"], errors="coerce")
        if logfc.isnull().any():
            issues.append(("WARNING", "Log2_Fold_Change has non-numeric values."))
        if ((logfc < -50) | (logfc > 50)).any():
            issues.append(("WARNING", "Log2_Fold_Change has extreme values (< -50 or > 50)."))

    # P-value and adjusted p-value
    for col in ["P_Value", "Adjusted_P_Value"]:
        if col in df.columns and not df[col].isnull().all():
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.isnull().any():
                issues.append(("WARNING", f"{col} has non-numeric values."))
            if ((vals < 0) | (vals > 1)).any():
                issues.append(("CRITICAL", f"{col} contains values outside [0, 1]."))

    # Total Reads checks
    for col in ["Total_Reads_Input_for_Quantification", "Total_Reads_Mapped"]:
        if col in df.columns and not df[col].isnull().all():
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.isnull().any():
                issues.append(("WARNING", f"{col} has non-numeric values."))
            if (vals < 0).any():
                issues.append(("CRITICAL", f"{col} contains negative values."))

    # -----------------------------
    # Print summary report
    # -----------------------------
    if issues:
        logging.warning("Validation Report:")
        report_df = pd.DataFrame(issues, columns=["Severity", "Issue"])
        counts = report_df.groupby("Severity").size().to_dict()
        for sev, issue in issues:
            logging.warning(f"[{sev}] {issue}")
        logging.warning("Summary: " + ", ".join([f"{k}={v}" for k, v in counts.items()]))

        if any(sev == "CRITICAL" for sev, _ in issues):
            smart_exit("Validation failed due to critical errors.")
    else:
        logging.info("Sanity checks passed")

    return df


# -----------------------------
# 5. Ingestion workflow
# -----------------------------

def ingest_rnaseq_file(file_path, source_name=None, sample_folder=None, rnaseq_output=None, mapping_path=None):
             
    if not os.path.exists(file_path):
        smart_exit(f"Input file not found: {file_path}")
    if not os.path.isfile(file_path):
        smart_exit(f"Input path is not a file: {file_path}")

    # --- Sample folder handling ---
    sample_csv = resolve_sample_csv(sample_folder)
    out_dir = os.path.dirname(os.path.abspath(sample_csv))

    try:
        sample_table = pd.read_csv(sample_csv)
    except Exception as e:
        smart_exit(f"Failed to read sample table {sample_csv}: {e}")

    sample_table["external_sample_id"] = (
        sample_table["external_sample_id"].astype(str).str.strip().str.lower()
    )
    print(f"✅ Loaded {len(sample_table)} samples from: {os.path.basename(sample_csv)}")

    # Normalize external_sample_id for joining
    sample_table = sample_table.copy()
    sample_table["external_sample_id_norm"] = (
        sample_table["external_sample_id"].astype(str).str.strip().str.lower()
    )
    #============
    mapping_json = None
    if mapping_path:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_json = json.load(f)
    #============

    # --- Prepare output path ---
    if rnaseq_output:
        out_file = os.path.abspath(rnaseq_output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(out_dir, f"rnaseq_output_{timestamp}.csv")

    # --- Stream RNA-seq data in mapped chunks ---
    first = True
    total_rows_written = 0

    for mapped_chunk in iter_rnaseq_mapped_chunks(file_path, batch_size=1000, mapping_json=mapping_json):
        if mapped_chunk is None or mapped_chunk.empty:
            continue

        # Normalize Sample_ID for joining
        chunk = mapped_chunk.copy()
        chunk["Sample_ID_norm"] = (
            chunk["Sample_ID"].astype(str).str.strip().str.lower()
        )

        # Join with sample_table on external_sample_id_norm
        df_merged = chunk.merge(
            sample_table[["external_sample_id_norm", "sample_id"]],
            left_on="Sample_ID_norm",
            right_on="external_sample_id_norm",
            how="inner"  # keep only rows that have a match
        )

        if df_merged.shape[0] == 0:
            continue

        df_merged = df_merged.copy()
        df_merged["Sample_ID"] = df_merged["sample_id"]

        helper_cols = ["Sample_ID_norm", "external_sample_id_norm", "sample_id"]
        df_processed = df_merged.drop(
            columns=[c for c in helper_cols if c in df_merged.columns]
        )

        #=======================================
        import re

        def _to_snake_case(name: str) -> str:
            s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
            return s.replace("__", "_").lower()

        # If snake_case exists, drop legacy CamelCase duplicates
        prefer_snake = {
            "ensembl_id": "Ensembl_ID",
            "expression_value": "Expression_Value",
        }

        for snake, camel in prefer_snake.items():
            if snake in df_processed.columns and camel in df_processed.columns:
                df_processed = df_processed.drop(columns=[camel])

        # Rename everything to snake_case
        df_output = df_processed.rename(columns={c: _to_snake_case(c) for c in df_processed.columns})

        # Keep only canonical RNA-seq columns in snake_case order
        canonical_cols = [_to_snake_case(c) for c in RNASEQ_COLUMNS]
        canonical_cols = [c for c in canonical_cols if c in df_output.columns]
        df_output = df_output[canonical_cols]


        #=========================================

        # Append chunk to disk
        mode = "w" if first else "a"
        header = first
        df_processed.to_csv(
            out_file,
            index=False,
            encoding="utf-8",
            mode=mode,
            header=header
        )

        total_rows_written += len(df_processed)
        first = False

    if total_rows_written == 0:
        smart_exit("Final merged RNA-seq dataframe is empty. Aborting before saving.")

    logging.info(f"Ingested {total_rows_written} rows from {file_path} → {out_file}")
    logging.info(f"Writing RNA-seq output to: {out_file}")
    return out_file


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
            if "the following arguments are required: input_file" in message:
                print("⚠️ No input file provided.")
                print("   Usage: python RNAseq.py <input_file>")
                sys.exit(1)
            else:
                self.print_help()
                sys.exit(1)

    parser = SmartArgumentParser(
        description="Ingest an RNA-seq file into standardized format and link to samples."
    )

    # ✅ Single input file (no list)
    parser.add_argument(
        "input_file",
      
        help="Path to input file (CSV, TSV, XLSX). Example: file1.csv"
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
        help="Optional RNA-seq mapping JSON from recommender (e.g., ACC_RNASEQ_mapping.json)."
    )
    args = parser.parse_args()

    try:
        out_file = ingest_rnaseq_file(
            file_path=args.input_file,
            sample_folder=args.sample_folder,
            rnaseq_output=args.output,
            mapping_path=args.mapping
        )

        logging.info(f"✅ RNA-seq data successfully saved to {out_file}")
    except SmartError as e:
        logging.error(f"🛑 SMARTWAY FATAL HALT: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"🛑 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc(limit=3)
        sys.exit(1)