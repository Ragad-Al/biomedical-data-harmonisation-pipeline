#!/usr/bin/env python
# coding: utf-8

'''
PYTHONPATH=. python onboarding/sv.py pcawg_consensus_1.6.161116.somatic_svs.xena.donor.tsv \
  --sample-folder pcawg_donor_centric \
  --mapping onboarding/PCAWG_SV_mapping.json

'''

import pandas as pd
import re
import uuid
import os
from datetime import datetime
import json

import sys
import logging
import traceback

import tempfile
from pathlib import Path
import glob

#=============================================================

try:
    from onboarding.transform_library import TEMPLATES
except ImportError:
    from transform_library import TEMPLATES

def apply_transforms(df: pd.DataFrame, transforms: dict) -> pd.DataFrame:
    if not transforms:
        return df

    def snake_to_legacy(s: str) -> str:
        parts = s.split("_")
        return "_".join([p.capitalize() for p in parts])

    for target_field, spec in transforms.items():
        template = spec.get("template")
        fn = TEMPLATES.get(template)
        if not fn:
            continue

        inputs = spec.get("inputs", None)
        if inputs is None:
            inputs = target_field if target_field in df.columns else snake_to_legacy(target_field)

        params = spec.get("params", {}) or {}
        df[target_field] = fn(df, inputs, params)

    return df


# ==========================================================
# 🧠 SMART ERROR GUARD SYSTEM
# ==========================================================

class SmartError(Exception):
    """Custom fatal error for Smartway pipeline."""
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EXIT_ERROR = 1
# --- END ADDED ---

# -----------------------------
# # 1. Structural Variant (SV) Schema
# -----------------------------
SV_COLUMNS = [
    "Sample_ID", "Gene_Symbol", "Ensembl_ID", "Chromosome",
    "Position_Start", "Position_End", "SV_Type", "Consequence",
    "Clinical_Significance", "Disease_Association", "Mutation_Origin",
    "variant_caller_software", "reference_genome_version", "annotation_databases_used"
]


# ------------------ CONFIG ------------------ #

SV_COLUMN_LOOKUP_FILE = "onboarding/column_name_lookup.json"
#-----------------------------------------------

# ---- SV type normalization helpers ----
SV_CODE_MAP = {
    # full words -> codes
    "deletion": "DEL",
    "duplication": "DUP",
    "inversion": "INV",
    "insertion": "INS",
    "translocation": "TRA",
    "breakend": "BND",
    "copynumbervariant": "CNV",
    "copy number variant": "CNV",
    "copy-number-variant": "CNV",
    # already-coded variants stay the same
    "del": "DEL", "dup": "DUP", "inv": "INV", "ins": "INS",
    "tra": "TRA", "bnd": "BND", "cnv": "CNV",
}

def normalize_sv_type_series(s):
    """
    Normalize a pandas Series of SV types to compact codes using SV_CODE_MAP
    and some tolerant regex matching.
    """
    if s is None:
        return s
    # to string, lower
    sl = s.astype(str).str.strip().str.lower()

    # exact map first
    mapped = sl.map(lambda v: SV_CODE_MAP.get(v, None))

    # fill any None using regex-based synonyms
    need = mapped.isna()
    if need.any():
        # broad, tolerant patterns
        mapped.loc[need & sl.str.contains(r"del|delet")] = "DEL"
        mapped.loc[need & sl.str.contains(r"dup|dupli|gain")] = "DUP"
        mapped.loc[need & sl.str.contains(r"inv|inver")] = "INV"
        mapped.loc[need & sl.str.contains(r"ins(?!tru)|insert")] = "INS"
        mapped.loc[need & sl.str.contains(r"tra|transloc|translo")] = "TRA"
        mapped.loc[need & sl.str.contains(r"bnd|breakend")] = "BND"
        mapped.loc[need & sl.str.contains(r"cnv|copy[_\- ]?number|copynumber")] = "CNV"

    # anything still unmapped → "UNK"
    mapped = mapped.fillna("UNK")
    return mapped


# -----------------------------
# 2. Field detection functions
# -----------------------------
def detect_column_by_name(df, keywords, expected_dtype="any"):
    """
    Detect column name by keyword match.
    Optionally restrict search to 'string' or 'numeric' columns.
    """
    if not keywords:
        return None

    keywords = [str(k).lower() for k in keywords]

    for c in df.columns:
        # Skip based on expected dtype
        if expected_dtype == "string" and pd.api.types.is_numeric_dtype(df[c]):
            continue
        if expected_dtype == "numeric" and not pd.api.types.is_numeric_dtype(df[c]):
            continue

        col_name = c.strip().lower()

        # Name match (case-insensitive, exact match)
        if col_name in keywords:
            return c

    return None


def detect_column_by_content(df, pattern=None, numeric_range=None, numeric_min=None, numeric_max=None):
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        if pattern:
            vals = s.astype(str).unique()[:50]
            if any(re.match(pattern, v) for v in vals):
                return c
        elif numeric_range:
            if pd.api.types.is_numeric_dtype(s):
                vals = s.head(50)
                if all(numeric_range[0] <= v <= numeric_range[1] for v in vals):
                    return c
        elif numeric_min is not None and numeric_max is not None:
            if pd.api.types.is_numeric_dtype(s):
                vals = s.head(50)
                if all(numeric_min <= v <= numeric_max for v in vals):
                    return c
    return None

# -----------------------------
# 🔹 Load dynamic column keyword mappings
# -----------------------------

def load_sv_column_name_lookup(filename=SV_COLUMN_LOOKUP_FILE):
    """Load keyword lookup for SV column name detection."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Structural Variant column name lookup file not found: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
# -----------------------------
# 3. Column mapping
# -----------------------------

def map_columns_dynamic(df, source_name=None, file_name=None, mapping_json=None):
    def _mapped_col(key):
        if not mapping_json or "mappings" not in mapping_json:
            return None
        spec = mapping_json["mappings"].get(key)
        if isinstance(spec, dict) and spec.get("mode") == "map":
            col = spec.get("source", {}).get("column")
            return col if col in df.columns else None
        return None
    
    mapped = {}

    # 🔹 Load dynamic keyword mappings
    try:
        column_lookup = load_sv_column_name_lookup()
    except FileNotFoundError as e:
        logging.error("Required file missing: %s", e)
        sys.exit(EXIT_ERROR)
    except json.JSONDecodeError as e:
        logging.error("sv column_name_lookup.json is not valid JSON: %s", e)
        sys.exit(EXIT_ERROR)

    # Sample_ID
    sample_col = _mapped_col("external_sample_id") or detect_column_by_name(df, column_lookup.get("Sample_ID", []), expected_dtype="string") or \
                 detect_column_by_content(df, pattern=r"^[A-Z0-9\-]+$")
    mapped["Sample_ID"] = df[sample_col] if sample_col else None

    # Gene_Symbol
    gene_col = _mapped_col("gene_symbol") or detect_column_by_name(df, column_lookup.get("Gene_Symbol", []), expected_dtype="string") or \
               detect_column_by_content(df, pattern=r"^(?!ENSG)\w{2,10}$")
    mapped["Gene_Symbol"] = df[gene_col] if gene_col else None

    # Ensembl_ID
    ens_col = detect_column_by_name(df, column_lookup.get("Ensembl_ID", []), expected_dtype="string") or \
              detect_column_by_content(df, pattern=r"ENSG\d+")
    mapped["Ensembl_ID"] = df[ens_col] if ens_col else None

    # Chromosome
    chr_col = _mapped_col("chromosome") or detect_column_by_name(df, column_lookup.get("Chromosome", [])) or \
              detect_column_by_content(df, pattern=r"^(chr)?[0-9XYM]+$")
    mapped["Chromosome"] = df[chr_col] if chr_col else None

    # Positions — dynamic keyword detection first, fallback to numeric inference
    start_col =  _mapped_col("position_start") or detect_column_by_name(df, column_lookup.get("Position_Start", []), expected_dtype="numeric")
    end_col   = _mapped_col("position_end") or detect_column_by_name(df, column_lookup.get("Position_End", []), expected_dtype="numeric")

    if start_col and end_col:
        mapped["Position_Start"] = df[start_col]
        mapped["Position_End"] = df[end_col]
    else:
        pos_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].max() > 1000]
        mapped["Position_Start"] = df[pos_candidates[0]] if len(pos_candidates) > 0 else None
        mapped["Position_End"]   = df[pos_candidates[1]] if len(pos_candidates) > 1 else None

    # ==========================================================
    # 🧬 SV_Type (trust file first; if missing or invalid, infer biologically)
    # ==========================================================
       # ==========================================================
    # 🧬 SV_Type (trust file first; if missing or invalid, infer biologically)
    # ==========================================================
    # First: detect which column actually contains consequence-like text
    cons_guess = detect_column_by_name(df, column_lookup.get("Consequence", []), expected_dtype="string")
    if not cons_guess:
        # fallback: look for common consequence-like column names
        for cand in ["consequence", "conseq", "consequence_type", "sv_consequence", "effect", "ann", "annotation", "variant_class", "svtype", "type"]:
            if any(cand in c.lower() for c in df.columns):
                cons_guess = [c for c in df.columns if cand in c.lower()][0]
                break

    # If still not found, try content-based detection (search for typical SV tokens)
    if not cons_guess:
        cons_guess = detect_column_by_content(df, pattern=r"(DEL|DUP|INV|INS|TRA|BND|CNV|deletion|duplication|inversion|insertion|transloc|breakend|copy[_\- ]?number)")

    # Create a normalized temporary column we will use for inference
    consequence_col_for_infer = None
    if cons_guess:
        consequence_col_for_infer = f"_CONSEQ_FOR_INFER"
        # safe assignment: cast to string and lowercase for matching
        df[consequence_col_for_infer] = df[cons_guess].astype(str).fillna("").str.lower()
    else:
        # no detected consequence column; create empty normalized column
        consequence_col_for_infer = f"_CONSEQ_FOR_INFER"
        df[consequence_col_for_infer] = ""

    # Debug print: show sample unique consequence strings so you can confirm detection
    try:
        sample_vals = df[consequence_col_for_infer].replace("", pd.NA).dropna().unique()[:30]
        
    except Exception:
        print("🧪 Unable to show consequence samples (check column detection).")

    # Now handle SV_Type: use file column if valid, else infer using the normalized column
    valid_types = {"DEL", "INS", "INV", "DUP", "BND", "TRA", "CNV", "TRANS"}
    vt_col = detect_column_by_name(df, column_lookup.get("SV_Type", []), expected_dtype="string")

    if vt_col:
        # take what's in the file, then normalize to codes
        df["SV_Type"] = df[vt_col]
    else:
        logging.info("ℹ️ No SV_Type column found — inferring SV_Type automatically.")
        df = infer_sv_type(df, consequence_col=consequence_col_for_infer)

    # normalize to codes either way
    df["SV_Type"] = normalize_sv_type_series(df["SV_Type"])
    # optional: validate after normalization
    bad = ~df["SV_Type"].isin(valid_types | {"UNK"})
    if bad.any():
        logging.warning("Found unexpected SV_Type values after normalization; marking as UNK.")
        df.loc[bad, "SV_Type"] = "UNK"

    mapped["SV_Type"] = df["SV_Type"]


    
    # Consequence
    cons_col = detect_column_by_name(
        df,
        column_lookup.get("Consequence", []),
        expected_dtype="string"
    )

    if cons_col:
        # Use whatever the JSON explicitly mapped as "Consequence"
        mapped["Consequence"] = df[cons_col]
    else:
        # Explicit fallback priority:
        # 1) Annotation
        # 2) Site2_Effect_on_Frame
        # 3) Site1_Description / Site2_Description combined
        annotation_col = None
        frame_col = None
        s1_desc_col = None
        s2_desc_col = None

        for c in df.columns:
            cl = c.lower()
            if cl == "annotation":
                annotation_col = c
            elif cl == "site2_effect_on_frame":
                frame_col = c
            elif cl == "site1_description":
                s1_desc_col = c
            elif cl == "site2_description":
                s2_desc_col = c

        if annotation_col is not None:
            # Best source for functional effect
            mapped["Consequence"] = df[annotation_col]
        elif frame_col is not None:
            # Next best: frame information
            mapped["Consequence"] = df[frame_col]
        elif s1_desc_col is not None or s2_desc_col is not None:
            # Last resort: combine descriptions into a single text field
            s1 = df[s1_desc_col].astype(str) if s1_desc_col is not None else ""
            s2 = df[s2_desc_col].astype(str) if s2_desc_col is not None else ""
            mapped["Consequence"] = (s1.fillna("") + " / " + s2.fillna("")).str.strip(" /")
        else:
            # No suitable column found
            mapped["Consequence"] = None


    # Clinical_Significance
    clin_col = detect_column_by_name(df, column_lookup.get("Clinical_Significance", []), expected_dtype="string")
    mapped["Clinical_Significance"] = df[clin_col] if clin_col else None

    # Disease_Association
    disease_col = detect_column_by_name(df, column_lookup.get("Disease_Association", []), expected_dtype="string")
    mapped["Disease_Association"] = df[disease_col] if disease_col else None

    # Mutation_Origin — dynamic keyword detection first, fallback to inference
    origin_col = detect_column_by_name(df, column_lookup.get("Mutation_Origin", []), expected_dtype="string")
    if origin_col:
        mapped["Mutation_Origin"] = df[origin_col]
    else:
        mapped["Mutation_Origin"] = infer_mutation_origin(df, source_name, file_name=file_name)


    # Variant caller software
    caller_col = detect_column_by_name(df, column_lookup.get("variant_caller_software", []), expected_dtype="string")
    mapped["variant_caller_software"] = df[caller_col] if caller_col else "unknown"

    # Reference genome version — first check if a column explicitly contains it
    ref_col = detect_column_by_name(df, column_lookup.get("reference_genome_version", []), expected_dtype="string")
    if ref_col:
        mapped["reference_genome_version"] = df[ref_col]
    else:
        mapped["reference_genome_version"] = infer_reference_genome(df, source_name)


    # Annotation databases
    annot_col = detect_column_by_name(df, column_lookup.get("annotation_databases_used", []), expected_dtype="string")
    mapped["annotation_databases_used"] = df[annot_col] if annot_col else "unknown"

    return pd.DataFrame(mapped, columns=SV_COLUMNS)

#========================================================
def infer_sv_type(df, consequence_col="_CONSEQ_FOR_INFER"):
    """
    Infer the SV_Type column using 'consequence_col' (normalized, lowercased string),
    otherwise falling back to length-based classification.
    """

    if "SV_Type" not in df.columns:
        df["SV_Type"] = pd.NA

    def classify_sv(row):
        # Keep existing SV_Type if present and non-empty
        existing = row.get("SV_Type")
        if pd.notnull(existing) and str(existing).strip():
            return existing

        consequence = ""
        if consequence_col in row and pd.notnull(row[consequence_col]):
            consequence = str(row[consequence_col]).lower().strip()
        else:
            # safety: try other common names if normalized col not present per-row
            for alt in ["consequence", "Consequence", "conseq", "annotation"]:
                if alt in row and pd.notnull(row[alt]):
                    consequence = str(row[alt]).lower().strip()
                    break

        # --- 1) direct substring checks (loose, to match PCAWG/TCGA styles) ---
        if re.search(r"del|delet", consequence):
            return "Deletion"
        if re.search(r"dup|dupli|gain", consequence):
            return "Duplication"
        if re.search(r"inv|inver", consequence):
            return "Inversion"
        if re.search(r"ins|insert", consequence):
            return "Insertion"
        if re.search(r"tra|transloc|translo", consequence):
            return "Translocation"
        if re.search(r"bnd|breakend", consequence):
            return "Breakend"
        if re.search(r"cnv|copy[_\- ]?number|copynumber", consequence):
            return "CopyNumberVariant"

         # --- Fallback: infer by genomic span length ---
        try:
            start = float(row.get("Position_Start", 0))
            end = float(row.get("Position_End", 0))
            length = abs(end - start)

            if length < 50:
                return "Small_variant"
            elif length < 1_000:
                return "MicroSV"
            elif length < 100_000:
                return "Small_SV"
            elif length < 1_000_000:
                return "Medium_SV"
            else:
                return "Large_SV"
        except Exception:
            return "Unknown"

    df["SV_Type"] = df.apply(classify_sv, axis=1)
    return df


# -----------------------------
# 4. Mutation origin inference
# -----------------------------

# Path to JSON file containing source categories
MUTATION_SOURCES_FILE = "mutation_sources.json"

def load_mutation_sources(filename=MUTATION_SOURCES_FILE):
    """
    Load mutation sources and clinical significance mapping from JSON file.
    Returns a dict with all categories.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Mutation sources file not found: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_mutation_origin(df, source_name=None, file_name=None):
    """
    Infer mutation origin ("Somatic", "Germline", "Non-Human", "Unknown")
    using source_name, Clinical_Significance, or allele frequency.
    """
    try:
        mutation_sources = load_mutation_sources()
    except FileNotFoundError as e:
        # --- ADDED: fail-fast instead of returning silent Series ---
        logging.error("Required file missing: %s", e)
        # Print stack for SmartWay logs
        traceback.print_exc()
        sys.exit(EXIT_ERROR)
    except json.JSONDecodeError as e:
        logging.error("mutation_sources.json is not valid JSON: %s", e)
        traceback.print_exc()
        sys.exit(EXIT_ERROR)


    fname = os.path.basename(file_name)
    file_hint = fname.lower() if fname else ""
    

    # ✅ Priority 1: Detect from file name
    if file_hint:
        for category, keywords in {
            "Somatic": mutation_sources.get("somatic_sources", []),
            "Germline": mutation_sources.get("germline_sources", []),
            "Non-Human": mutation_sources.get("nonhuman_sources", [])
        }.items():
            if any(k.lower() in file_hint for k in keywords):
                
                return pd.Series([category] * len(df))

        # Fallback heuristic if JSON keywords didn’t match
        if any(x in file_hint for x in ["somatic", "tumor", "cancer"]):
            
            return pd.Series(["Somatic"] * len(df))
        elif any(x in file_hint for x in ["germline", "inherited", "normal"]):
            
            return pd.Series(["Germline"] * len(df))
        elif any(x in file_hint for x in ["mouse", "zebrafish", "arabidopsis", "nonhuman"]):
           
            return pd.Series(["Non-Human"] * len(df))

    # ---- Priority 2: Source name ----
    if source_name:
        source = source_name.lower()
        for category, sources in {
            "Somatic": mutation_sources.get("somatic_sources", []),
            "Germline": mutation_sources.get("germline_sources", []),
            "Non-Human": mutation_sources.get("nonhuman_sources", [])
        }.items():
            if any(s.lower() in source for s in sources):
                return pd.Series([category] * len(df))

    # ---- Priority 3: Clinical_Significance ----
    if "Clinical_Significance" in df.columns:
        cs_map = mutation_sources.get("clinical_significance_mapping", {})
        # ✅ normalize keys to lowercase for safe matching
        cs_map = {k.lower(): v for k, v in cs_map.items()}
        cs = df["Clinical_Significance"].astype(str).str.lower()
        origins = [cs_map.get(val, "Unknown") for val in cs]
        if any(o != "Unknown" for o in origins):
            return pd.Series(origins)

    # ---- Priority 4: Allele Frequency heuristic ----
    if "DNA_Variant_Allele_Frequency" in df.columns:
        vaf = pd.to_numeric(df["DNA_Variant_Allele_Frequency"], errors="coerce").fillna(0)
        return pd.Series(["Somatic" if v < 0.5 else "Germline" for v in vaf])

    # ---- Default ----
    return pd.Series(["Unknown"] * len(df))



# -----------------------------
# 5. Reference genome inference
# -----------------------------

def load_reference_genomes(json_path="reference_genomes.json"):
    """Load reference genomes mapping from JSON file."""
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_reference_from_file_header(filename, json_path="reference_genomes.json"):
    """
    Infer reference genome from a file's header or first row.
    Works for VCF, TSV, CSV, and other text-based formats.
    Uses reference_genomes.json for genome keywords.
    """
    if not os.path.exists(filename):
        return None

    reference_genomes = load_reference_genomes(json_path)

    # Flatten all genome keywords into a single lookup list
    genome_keywords = []
    for species_genomes in reference_genomes.values():
        for aliases in species_genomes.values():
            genome_keywords.extend(alias.lower() for alias in aliases)

    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip().lower()

                # Priority: VCF-style reference header
                if line.startswith("##reference="):
                    return line.split("##reference=")[1].strip()

                # Check comment/header lines
                if line.startswith("#") or not line:
                    for genome_keyword in genome_keywords:
                        if genome_keyword in line:
                            return genome_keyword.upper()

                # For TSV/CSV header row
                if not line.startswith("#"):
                    columns = line.split("\t") if "\t" in line else line.split(",")
                    for col in columns:
                        for genome_keyword in genome_keywords:
                            if genome_keyword in col.lower():
                                return genome_keyword.upper()
                    break  # Only check first header row
    except Exception as e:
        print(f"⚠️ Warning: Failed to read header from {filename}: {e}")

    return None

def infer_reference_genome(df=None, source_name=None, filename=None, json_path="reference_genomes.json"):
    """
    Infer reference genome using:
      1. File header
      2. Filename
      3. Column names
      4. Source name mapping
    Returns the genome string, or None if not found.
    """
    # Load JSON
    reference_genomes = load_reference_genomes(json_path)

    # Flatten species aliases
    flat_genomes = {}
    for species_dict in reference_genomes.get("species", {}).values():
        for genome, aliases in species_dict.items():
            for alias in aliases:
                flat_genomes[alias.lower()] = genome

    # Flatten source_name_mappings and lowercase keys
    source_mappings = {k.lower(): v for k, v in reference_genomes.get("source_name_mappings", {}).items()}

    # ----- Priority 1: File header -----
    if filename and os.path.exists(filename):
        header_genome = get_reference_from_file_header(filename, json_path=json_path)
        if header_genome:
            return header_genome

    # ----- Priority 2: Filename -----
    if filename:
        fname = os.path.basename(filename).lower()
        for alias, genome in flat_genomes.items():
            if alias in fname:
                return genome

    # ----- Priority 3: Column names -----
    if df is not None and isinstance(df, pd.DataFrame):
        col_string = " ".join(df.columns).lower()
        for alias, genome in flat_genomes.items():
            if alias in col_string:
                return genome

    # ----- Priority 4: Source name mapping -----
    if source_name:
        s = source_name.lower()
        for key, genome in source_mappings.items():
            if key in s:  # substring match
                return genome

    # ----- Not found -----
    return None



# -----------------------------
# 6. Ingestion workflow
# -----------------------------

# --- ADDED: helper validators ---
def _validate_json_file(path, required=False, name="file"):
    if not os.path.exists(path):
        if required:
            logging.error("%s is required but missing: %s", name, path)
            sys.exit(EXIT_ERROR)
        else:
            logging.warning("%s not found: %s", name, path)
            return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON in %s (%s): %s", name, path, e)
        traceback.print_exc()
        sys.exit(EXIT_ERROR)
    except Exception as e:
        logging.error("Failed to read %s (%s): %s", name, path, e)
        traceback.print_exc()
        sys.exit(EXIT_ERROR)

def _validate_input_file_path(path):
    if not os.path.exists(path):
        logging.error("Input file not found: %s", path)
        sys.exit(EXIT_ERROR)
    if not os.path.isfile(path):
        logging.error("Input path is not a file: %s", path)
        sys.exit(EXIT_ERROR)
    if os.path.getsize(path) == 0:
        logging.error("Input file is empty: %s", path)
        sys.exit(EXIT_ERROR)
    # Basic extension check
    if not (path.endswith(".tsv") or path.endswith(".csv") or path.endswith(".txt") or path.endswith(".vcf")):
        logging.warning("Input file does not have a standard extension (.tsv/.csv/.vcf). Proceeding but validate format: %s", path)

def _validate_mapped_df(df_mapped, original_len):
    if not isinstance(df_mapped, pd.DataFrame):
        logging.error("Mapped output is not a DataFrame.")
        sys.exit(EXIT_ERROR)
    # Ensure all required columns exist
    missing = [c for c in SV_COLUMNS if c not in df_mapped.columns]
    if missing:
        logging.error("Mapped DataFrame is missing required columns: %s", missing)
        sys.exit(EXIT_ERROR)
    # Ensure lengths match original
    if len(df_mapped) != original_len:
        logging.error("Row count mismatch: original rows=%d, mapped rows=%d", original_len, len(df_mapped))
        sys.exit(EXIT_ERROR)
    # Basic sanity checks for critical columns
    # Positions numeric and >0 if present
    for pos_col in ("Position_Start", "Position_End"):
        if pos_col in df_mapped.columns and df_mapped[pos_col] is not None:
            try:
                nums = pd.to_numeric(df_mapped[pos_col], errors="coerce")
                if nums.isnull().all():
                    logging.error("Position column %s contains no numeric values.", pos_col)
                    sys.exit(EXIT_ERROR)
                if (nums.dropna() <= 0).any():
                    logging.error("Position column %s contains non-positive values.", pos_col)
                    sys.exit(EXIT_ERROR)
            except Exception as e:
                logging.error("Failed to validate position column %s: %s", pos_col, e)
                traceback.print_exc()
                sys.exit(EXIT_ERROR)
    
# ---  -------
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
#----------------------------------------------------------
def resolve_sample_csv(sample_folder: str) -> str:
    """Return the newest sample_output_*.csv from sample_folder or exit."""
    folder = os.path.abspath(os.path.expanduser(sample_folder))
    if not os.path.isdir(folder):
        logging.error(f"Invalid sample folder: {sample_folder}")
        sys.exit(EXIT_ERROR)
    matches = glob.glob(os.path.join(folder, "sample_output_*.csv"))
    if not matches:
        logging.error(
            f"No sample_output_*.csv found in {folder}. "
            "Run sample.py first and point --sample-folder to that folder."
        )
        sys.exit(EXIT_ERROR)
    return max(matches, key=os.path.getmtime)

# ==========================================================
# 🧩 Ingestion (Structural Variant (SV)-Pipeline)
# ==========================================================

def ingest_single_file(file_path, source_name=None, sample_folder=None, sv_output=None, mapping_path=None):

    mapping_json = None
    if mapping_path:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_json = json.load(f)
            
    # --- ADDED: validate input file path before attempting read ---
    _validate_input_file_path(file_path)

    # --- Sample folder handling (non-interactive) ---
    sample_csv = resolve_sample_csv(sample_folder)
    sample_table = pd.read_csv(sample_csv)
    sample_table["external_sample_id"] = (
        sample_table["external_sample_id"].astype(str).str.strip().str.lower()
    )
    print(f"✅ Loaded {len(sample_table)} samples from: {os.path.basename(sample_csv)}")


    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    df = read_file_by_extension(file_path)

    # ---  check dataframe non-empty ---
    if df is None or df.empty:
        logging.error("Parsed DataFrame is empty for file: %s", file_path)
        sys.exit(EXIT_ERROR)
    # ---  ---

    # ---  Ensure mutation_sources.json exists and is valid before mapping (infer_mutation_origin depends on it)
    _ = _validate_json_file(MUTATION_SOURCES_FILE, required=True, name="mutation_sources.json")
    # Validate reference_genomes.json if exists (not required)
    _ = _validate_json_file("reference_genomes.json", required=False, name="reference_genomes.json")
    # ---  ---

    try:
        df_mapped = map_columns_dynamic(df, source_name, file_name=file_path, mapping_json=mapping_json)
        df_mapped = apply_transforms(df_mapped, mapping_json.get("transforms", {}) if mapping_json else {})
    except SystemExit:
        # propagate intentional exits
        raise
    except Exception as e:
        logging.error("Mapping failed for file %s: %s", file_path, e)
        traceback.print_exc()
        sys.exit(EXIT_ERROR)

    # --- ADDED: Validate mapped DataFrame sanity and consistency ---
    try:
        _validate_mapped_df(df_mapped, len(df))
    except SystemExit:
        raise
    except Exception as e:
        logging.error("Post-mapping validation unexpected error: %s", e)
        traceback.print_exc()
        sys.exit(EXIT_ERROR)
    # --- END ADDED ---

    #+++++++++++++++++++++++++

    # -----------------------------
    # 🔹 Filter and map Sample_IDs
    # -----------------------------
    # Normalize SV Sample_IDs for case-insensitive matching
    df_mapped["Sample_ID_norm"] = df_mapped["Sample_ID"].astype(str).str.strip().str.lower()

    # Normalize sample table external IDs
    sample_table["external_sample_id_norm"] = sample_table["external_sample_id"].astype(str).str.strip().str.lower()


    # Merge with sample_table on external_sample_id
    df_merged = df_mapped.merge(
        sample_table[["external_sample_id_norm", "sample_id"]],
        left_on="Sample_ID_norm",
        right_on="external_sample_id_norm",
        how="inner"  # keep only rows that have a match
    )

    # Replace Sample_ID with the corresponding sample_id from sample_table
    df_merged["Sample_ID"] = df_merged["sample_id"]
    # Drop helper columns so only SV_COLUMNS remain
    helper_cols = ["Sample_ID_norm", "external_sample_id_norm", "sample_id"]
    df_processed = df_merged.drop(columns=[c for c in helper_cols if c in df_merged.columns])
    df_output = df_processed
    #+++++++++++++++++++
    # Ensure output has exact SV_COLUMNS in the specified order; if any column is missing (None), add as nulls
    #for col in SV_COLUMNS:
        #if col not in df_output.columns:
            #df_output[col] = pd.NA
    #df_output = df_output[SV_COLUMNS]


    #++++++++++++++++++++
    # 🔹 Create unique output file names 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if sv_output:
        output_file = sv_output
    else:
        output_file = os.path.join(sample_folder, f"sv_output_{timestamp}.csv")

    try:
        df_output.to_csv(output_file, index=False, encoding="utf-8")
        
    except Exception as e:
        logging.error("Failed to write to output file %s: %s", output_file, e)
        traceback.print_exc()
        sys.exit(EXIT_ERROR)

    logging.info("Ingested %d rows from %s → %s", len(df_output), file_path, output_file)
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
            if "the following arguments are required: input_file" in message:
                print("⚠️ No input file provided.")
                print("   Usage: python sv.py <input_file>")
                sys.exit(1)
            else:
                self.print_help()
                sys.exit(1)

    parser = SmartArgumentParser(
        description="Ingest an SV file into standardized format and link to samples."
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
        "--source-name",
        default=None,
        help="Optional source name hint used for mutation/SV origin/reference inference."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path. If omitted, a timestamped file is written under --sample-folder."
    )

    parser.add_argument("--mapping", default=None, help="SV mapping JSON from recommender.")

    args = parser.parse_args()

    try:
        # ✅ Pass a single file (not a list)
        out_file = ingest_single_file(
            file_path=args.input_file,
            source_name=args.source_name,
            sample_folder=args.sample_folder,
            sv_output=args.output,
            mapping_path=args.mapping
        )

        logging.info(f"✅ SV data successfully saved to {out_file}")
    except SmartError as e:
        logging.error(f"🛑 SMARTWAY FATAL HALT: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"🛑 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc(limit=3)
        sys.exit(1)

