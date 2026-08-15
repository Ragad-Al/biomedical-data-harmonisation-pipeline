#!/usr/bin/env python
# coding: utf-8

'''
PYTHONPATH=. python onboarding/mutation.py TCGA-ACC.somaticmutation_wxs.tsv \
  --sample-folder tcga_acc1 \
  --mapping onboarding/ACC_Mutation_mapping.json

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

# ==========================================================
# 🧠 SMART ERROR GUARD SYSTEM
# ==========================================================

try:
    from onboarding.transform_library import TEMPLATES
except ImportError:
    from transform_library import TEMPLATES

def apply_transforms(df: pd.DataFrame, transforms: dict) -> pd.DataFrame:
    if not transforms:
        return df
    for target_field, spec in transforms.items():
        template = spec.get("template")
        fn = TEMPLATES.get(template)
        if not fn:
            continue
        inputs = spec.get("inputs", target_field)  # default: transform the target col itself
        params = spec.get("params", {}) or {}
        df[target_field] = fn(df, inputs, params)
    return df
# ==========================================================

class SmartError(Exception):
    """Custom fatal error for Smartway pipeline."""
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EXIT_ERROR = 1
# --- END ADDED ---

# -----------------------------
# 1. Mutation schema (new structure)
# -----------------------------
MUTATION_COLUMNS = [
    "Sample_ID", "Gene_Symbol", "Ensembl_ID", "Transcript_ID",
    "Chromosome", "Position_Start", "Position_End",
    "Reference_Allele", "Alternate_Allele", "Variant_Type",
    "Nucleotide_Change", "Amino_Acid_Change", "Consequence",
    "DNA_Variant_Allele_Frequency", "SIFT_score", "PolyPhen_score",
    "Clinical_Significance", "Disease_Association", "Mutation_Origin",
    "variant_caller_software", "reference_genome_version", "annotation_databases_used"
]


# ------------------ CONFIG ------------------ #

MUTATION_COLUMN_LOOKUP_FILE = "onboarding/column_name_lookup.json"

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

        # Name match (case-insensitive)
        if any(k in c.lower() for k in keywords):
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

def infer_variant_type(ref, alt):
    if pd.isna(ref) or pd.isna(alt):
        return "Unknown"
    ref, alt = str(ref), str(alt)

    # CNV pattern
    cnv_pattern = r"^[0-9XYM]+:[0-9]+-[0-9]+$"
    if re.match(cnv_pattern, ref) or re.match(cnv_pattern, alt):
        return "CNV"

    if ref in ["-", ""] and len(alt) > 0:
        return "INS"
    elif alt in ["-", ""] and len(ref) > 0:
        return "DEL"
    elif len(ref) == 1 and len(alt) == 1:
        return "SNP"
    elif len(ref) == len(alt) and len(ref) > 1:
        if len(ref) == 2:
            return "DNP"
        else:
            return "MNV"
    else:
        return "Unknown"

# -----------------------------
# 🔹 Load dynamic column keyword mappings
# -----------------------------

def load_mutation_column_name_lookup(filename=MUTATION_COLUMN_LOOKUP_FILE):
    """Load keyword lookup for mutation column name detection."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Mutation column name lookup file not found: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
# -----------------------------
# 3. Column mapping
# -----------------------------
def map_columns_dynamic(df, source_name=None, file_name=None, mapping_json=None):

    def _mapped_col(mapping_json, key_candidates):
        """
        mapping_json is expected to contain:
        mapping_json["mappings"][<key>]["source"]["column"]
        key_candidates: list of possible keys (snake_case or MUTATION_COLUMNS style)
        """
        if not mapping_json or "mappings" not in mapping_json:
            return None
        m = mapping_json["mappings"]
        for k in key_candidates:
            spec = m.get(k)
            if isinstance(spec, dict) and spec.get("mode") == "map":
                col = spec.get("source", {}).get("column")
                if col and col in df.columns:
                    return col
        return None

    
    mapped = {}

    # 🔹 Load dynamic keyword mappings
    try:
        column_lookup = load_mutation_column_name_lookup()
    except FileNotFoundError as e:
        logging.error("Required file missing: %s", e)
        sys.exit(EXIT_ERROR)
    except json.JSONDecodeError as e:
        logging.error("mutation_column_name_lookup.json is not valid JSON: %s", e)
        sys.exit(EXIT_ERROR)

    # Sample_ID
    sample_col = (
        _mapped_col(mapping_json, ["external_sample_id", "Sample_ID", "sample_id", "sample"])
        or detect_column_by_name(df, column_lookup.get("Sample_ID", []), expected_dtype="string")
        or detect_column_by_content(df, pattern=r"^[A-Z0-9\-]+$")
    )
    mapped["Sample_ID"] = df[sample_col] if sample_col else None

    # Gene_Symbol
    gene_col = detect_column_by_name(df, column_lookup.get("Gene_Symbol", []), expected_dtype="string") or \
               detect_column_by_content(df, pattern=r"^(?!ENSG)\w{2,10}$")
    mapped["Gene_Symbol"] = df[gene_col] if gene_col else None

    # Ensembl_ID
    ens_col = detect_column_by_name(df, column_lookup.get("Ensembl_ID", []), expected_dtype="string") or \
              detect_column_by_content(df, pattern=r"ENSG\d+")
    mapped["Ensembl_ID"] = df[ens_col] if ens_col else None

    # Transcript_ID
    tx_col = detect_column_by_name(df, column_lookup.get("Transcript_ID", []), expected_dtype="string") or \
             detect_column_by_content(df, pattern=r"ENST\d+")
    mapped["Transcript_ID"] = df[tx_col] if tx_col else None

    # Chromosome
    chr_col = (
        _mapped_col(mapping_json, ["chromosome", "Chromosome", "chrom"])
        or detect_column_by_name(df, column_lookup.get("Chromosome", []))
        or detect_column_by_content(df, pattern=r"^(chr)?[0-9XYM]+$")
    )
    mapped["Chromosome"] = df[chr_col] if chr_col else None

    # Positions — dynamic keyword detection first, fallback to numeric inference
    start_col = (
        _mapped_col(mapping_json, ["position_start", "Position_Start", "start"])
        or detect_column_by_name(df, column_lookup.get("Position_Start", []), expected_dtype="numeric")
    )
    end_col = (
        _mapped_col(mapping_json, ["position_end", "Position_End", "end"])
        or detect_column_by_name(df, column_lookup.get("Position_End", []), expected_dtype="numeric")
    )

    if start_col and end_col:
        mapped["Position_Start"] = df[start_col]
        mapped["Position_End"] = df[end_col]
    else:
        pos_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].max() > 1000]
        mapped["Position_Start"] = df[pos_candidates[0]] if len(pos_candidates) > 0 else None
        mapped["Position_End"]   = df[pos_candidates[1]] if len(pos_candidates) > 1 else None
        
    # Alleles — dynamic keyword detection first, fallback to sequence pattern detection
    ref_col = (
        _mapped_col(mapping_json, ["reference_allele", "Reference_Allele", "ref"])
        or detect_column_by_name(df, column_lookup.get("Reference_Allele", []), expected_dtype="string")
    )
    alt_col = (
        _mapped_col(mapping_json, ["alternate_allele", "Alternate_Allele", "alt"])
        or detect_column_by_name(df, column_lookup.get("Alternate_Allele", []), expected_dtype="string")
    )

    if ref_col and alt_col:
        mapped["Reference_Allele"] = df[ref_col]
        mapped["Alternate_Allele"] = df[alt_col]
    else:
        allele_candidates = [
            c for c in df.columns
            if all(re.match(r"^[ACGTNacgtn\-]+$", v) for v in df[c].dropna().astype(str).head(50))
        ]
        mapped["Reference_Allele"] = df[allele_candidates[0]] if len(allele_candidates) > 0 else None
        mapped["Alternate_Allele"] = df[allele_candidates[1]] if len(allele_candidates) > 1 else None


    # Variant_Type
    # Variant_Type (trust file first, validate, otherwise infer from alleles)
    valid_types = {"SNP", "INS", "DEL", "MNV", "DNP", "CNV"}
    vt_col = detect_column_by_name(df, column_lookup.get("Variant_Type", []), expected_dtype="string")

    if vt_col:
        vals = set(df[vt_col].dropna().astype(str).str.upper().unique())
        if vals.issubset(valid_types):
            mapped["Variant_Type"] = df[vt_col].astype(str).str.upper()
        else:
            if "Reference_Allele" in mapped and "Alternate_Allele" in mapped:
                mapped["Variant_Type"] = [
                    infer_variant_type(r, a) for r, a in zip(mapped["Reference_Allele"], mapped["Alternate_Allele"])
                ]
            else:
                mapped["Variant_Type"] = ["Unknown"] * len(df)
    else:
        if "Reference_Allele" in mapped and "Alternate_Allele" in mapped:
            mapped["Variant_Type"] = [
                infer_variant_type(r, a) for r, a in zip(mapped["Reference_Allele"], mapped["Alternate_Allele"])
            ]
        else:
            mapped["Variant_Type"] = ["Unknown"] * len(df)

    # Nucleotide_Change
    nuc_col = detect_column_by_name(df, column_lookup.get("Nucleotide_Change", [])) or \
              detect_column_by_content(df, pattern=r"c\.[0-9]+[ACGTN]+[>][ACGTN]+")
    mapped["Nucleotide_Change"] = df[nuc_col] if nuc_col else None

    # Amino_Acid_Change
    aa_col = detect_column_by_name(df, column_lookup.get("Amino_Acid_Change", [])) or \
             detect_column_by_content(df, pattern=r"^p\.[A-Z][a-z]{2}[0-9]+[A-Z][a-z]{2}$")
    mapped["Amino_Acid_Change"] = df[aa_col] if aa_col else None

    # Consequence
    cons_col = detect_column_by_name(df, column_lookup.get("Consequence", []), expected_dtype="string") or detect_column_by_content(df, pattern=r".*")
    mapped["Consequence"] = df[cons_col] if cons_col else None

    #---------------------------------------------------------------------

    # DNA Variant Allele Frequency
    vaf_col = (
        _mapped_col(mapping_json, ["dna_variant_allele_frequency", "DNA_Variant_Allele_Frequency", "dna_vaf"])
        or detect_column_by_name(df, column_lookup.get("DNA_Variant_Allele_Frequency", []), expected_dtype="numeric")
        or detect_column_by_content(df, numeric_range=(0, 1))
    )

    vaf_series = None

    # 1) Try to use an existing VAF column, but only if it's sensible
    if vaf_col is not None:
        tmp_vaf = pd.to_numeric(df[vaf_col], errors="coerce")
        non_na = tmp_vaf.dropna()

        if not non_na.empty:
            # Case A: already in 0–1 range AND not all zeros → use as-is
            if non_na.between(0.0, 1.0).all() and (non_na > 0.0).any():
                vaf_series = tmp_vaf
            # Case B: looks like 0–100% → convert to 0–1
            elif non_na.between(0.0, 100.0).all() and (non_na > 1.0).any():
                vaf_series = tmp_vaf / 100.0
            else:
                # All zeros or otherwise nonsensical → fall back to counts
                vaf_series = None

    if vaf_series is not None:
        mapped["DNA_Variant_Allele_Frequency"] = vaf_series
    else:
        # 2) Fallback: compute VAF from tumour read counts if present
        t_ref_col = None
        t_alt_col = None

        for c in df.columns:
            cl = c.lower()
            if cl in ("t_ref_count", "tumor_ref_count", "tumour_ref_count"):
                t_ref_col = c
            elif cl in ("t_alt_count", "tumor_alt_count", "tumour_alt_count"):
                t_alt_col = c

        if t_ref_col and t_alt_col:
            t_ref = pd.to_numeric(df[t_ref_col], errors="coerce")
            t_alt = pd.to_numeric(df[t_alt_col], errors="coerce")
            total = t_ref + t_alt

            vaf = t_alt / total
            # avoid division-by-zero
            vaf = vaf.where(total > 0)

            # final safety clamp to [0,1] so validator is happy
            vaf = vaf.where((vaf >= 0.0) & (vaf <= 1.0))

            mapped["DNA_Variant_Allele_Frequency"] = vaf
        else:
            # No VAF column and no counts to derive it from
            mapped["DNA_Variant_Allele_Frequency"] = None


    #--------------------------------------------------------------------------
    # SIFT_score
    sift_col = detect_column_by_name(df, column_lookup.get("SIFT_score", []))
    mapped["SIFT_score"] = df[sift_col] if sift_col else None

    # PolyPhen_score
    poly_col = detect_column_by_name(df, column_lookup.get("PolyPhen_score", []))
    mapped["PolyPhen_score"] = df[poly_col] if poly_col else None

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

    return pd.DataFrame(mapped, columns=MUTATION_COLUMNS)

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

def resolve_sample_csv(sample_folder: str) -> str:
    """Return the newest sample_output_*.csv from sample_folder or raise SmartError."""
    folder = os.path.abspath(os.path.expanduser(sample_folder))
    if not os.path.isdir(folder):
        raise SmartError(f"Invalid sample folder: {sample_folder}")
    pattern = os.path.join(folder, "sample_output_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise SmartError(
            f"No sample_output_*.csv found in {folder}. "
            "Run sample.py first and point --sample-folder to that folder."
        )
    return max(matches, key=os.path.getmtime)  # newest file


# -----------------------------
# 5. Reference genome inference
# -----------------------------
import os
import json
import pandas as pd

def load_mapping_json(mapping_path):
    if not mapping_path:
        return None
    if not os.path.exists(mapping_path):
        raise SmartError(f"Mapping JSON not found: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    missing = [c for c in MUTATION_COLUMNS if c not in df_mapped.columns]
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
    # Alleles characters
    for allele_col in ("Reference_Allele", "Alternate_Allele"):
        if allele_col in df_mapped.columns and df_mapped[allele_col] is not None:
            # Accept either nucleotides or CNV coordinates
            invalid_mask = df_mapped[allele_col].dropna().astype(str).apply(lambda x: not re.match(r"^([ACGTNacgtn\-]+|[0-9XYM]+:[0-9]+-[0-9]+)$", x))
            if invalid_mask.any():
                count = invalid_mask.sum()
                logging.error("%d invalid allele values detected in %s. Sample invalid values: %s", count, allele_col, df_mapped[allele_col].dropna().astype(str)[invalid_mask].unique()[:5].tolist())
                sys.exit(EXIT_ERROR)
    # VAF range check if present
    if "DNA_Variant_Allele_Frequency" in df_mapped.columns and df_mapped["DNA_Variant_Allele_Frequency"] is not None:
        try:
            vaf = pd.to_numeric(df_mapped["DNA_Variant_Allele_Frequency"], errors="coerce")
            invalid = vaf.dropna().apply(lambda x: not (0.0 <= x <= 1.0))
            if invalid.any():
                logging.error("DNA_Variant_Allele_Frequency contains values outside [0,1]. Invalid count: %d", invalid.sum())
                sys.exit(EXIT_ERROR)
        except Exception as e:
            logging.error("Failed to validate DNA_Variant_Allele_Frequency: %s", e)
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
# ==========================================================
# 🧩 Ingestion (mutation-Pipeline)
# ==========================================================
def ingest_single_file(file_path, source_name=None, sample_folder=None, sample_output=None, mapping_path=None):

    # --- validate input file path before attempting read ---
    _validate_input_file_path(file_path)

    # Source name & sample folder are provided via CLI (non-interactive)
    # sample_folder is REQUIRED (validated below); source_name is optional.

    # Validate the folder
    # Find newest sample_output_*.csv using the helper
    sample_csv = resolve_sample_csv(sample_folder)

    # Load + normalize IDs
    sample_table = pd.read_csv(sample_csv)
    sample_table["external_sample_id"] = (
        sample_table["external_sample_id"].astype(str).str.strip().str.lower()
    )
    ref_sample_ids = set(sample_table["external_sample_id"])
    print(f"✅ Loaded {len(sample_table)} samples from: {os.path.basename(sample_csv)}")



    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    df = read_file_by_extension(file_path)
    mapping_json = load_mapping_json(mapping_path)

    # ---  check dataframe non-empty ---
    if df is None or df.empty:
        logging.error("Parsed DataFrame is empty for file: %s", file_path)
        sys.exit(EXIT_ERROR)
    # ---  ---

    # Ensure mutation_sources.json exists and is valid before mapping (infer_mutation_origin depends on it)
    _ = _validate_json_file(MUTATION_SOURCES_FILE, required=True, name="mutation_sources.json")
    # Validate reference_genomes.json if exists (not required)
    _ = _validate_json_file("reference_genomes.json", required=False, name="reference_genomes.json")
    # ---  ---

    try:
        df_mapped = map_columns_dynamic(df, source_name, file_name=file_path, mapping_json=mapping_json)

        # ✅ execute transforms from mapping JSON
        df_mapped = apply_transforms(df_mapped, mapping_json.get("transforms", {}))

        # --- Step-4 normalisations ---
        #if "Chromosome" in df_mapped.columns:
            #df_mapped["Chromosome"] = (
                #df_mapped["Chromosome"]
                #.astype(str)
                #.str.replace(r"^chr", "", regex=True, case=False)
                #.str.strip()
            #)

        #for a in ["Reference_Allele", "Alternate_Allele"]:
            #if a in df_mapped.columns:
                #df_mapped[a] = df_mapped[a].astype(str).str.upper().str.strip()
        
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
    # Normalize mutation Sample_IDs for case-insensitive matching
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
    # Drop helper columns so only MUTATION_COLUMNS remain
    helper_cols = ["Sample_ID_norm", "external_sample_id_norm", "sample_id"]
    df_processed = df_merged.drop(columns=[c for c in helper_cols if c in df_merged.columns])
    df_output = df_processed

    #=================================================================
    # -----------------------------
    # ✅ FINAL OUTPUT CLEANUP: enforce snake_case and remove duplicates

    # -----------------------------
    def _to_snake_case(name: str) -> str:
        import re
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
        return s.replace("__", "_").lower()

    # If transform-created snake_case columns exist, drop the legacy CamelCase versions
    prefer_snake = {
        "chromosome": "Chromosome",
        "reference_allele": "Reference_Allele",
        "alternate_allele": "Alternate_Allele",
        "dna_variant_allele_frequency": "DNA_Variant_Allele_Frequency",
    }
    for snake, camel in prefer_snake.items():
        if snake in df_output.columns and camel in df_output.columns:

            df_output = df_output.drop(columns=[camel])

    # Rename all remaining columns to snake_case
    rename_map = {c: _to_snake_case(c) for c in df_output.columns}
    df_output = df_output.rename(columns=rename_map)

    # Ensure unified sample_id column name (your UUID is currently in Sample_ID)
    # After rename_map, Sample_ID becomes sample_id automatically. This line is just safety:
    if "sample_id" not in df_output.columns and "sample_id" in rename_map.values():
        pass

    # Optional: keep only canonical mutation columns in snake_case order
    canonical_cols = [_to_snake_case(c) for c in MUTATION_COLUMNS]
    canonical_cols = [c for c in canonical_cols if c in df_output.columns]

    df_output = df_output[canonical_cols]

    #+++++++++++++++++++
    # Decide output path: use explicit --output if provided, else place in --sample-folder
    if sample_output:
        output_file = sample_output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(sample_folder, f"mutation_output_{timestamp}.csv")
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
                print("   Usage: python mutation.py <input_file>")
                sys.exit(1)
            else:
                self.print_help()
                sys.exit(1)

    parser = SmartArgumentParser(
        description="Ingest a mutation file into standardized format and link to samples."
    )

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
        help="Optional source name hint used for mutation origin/reference inference."
    )

    parser.add_argument(
    "--mapping",
    default=None,
    help="Optional mutation mapping JSON produced by recommender (e.g., ACC_Mutation_mapping.json)."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path. If omitted, a timestamped file is written under --sample-folder."
    )


    args = parser.parse_args()

    try:
        out_file = ingest_single_file(
            file_path=args.input_file,
            source_name=args.source_name,
            sample_folder=args.sample_folder,
            sample_output=args.output,
            mapping_path=args.mapping
        )
        logging.info(f"✅ Mutation data successfully saved to {out_file}")
        
    except SmartError as e:
        logging.error(f"🛑 SMARTWAY FATAL HALT: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"🛑 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc(limit=3)
        sys.exit(1)

