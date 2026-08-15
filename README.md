# Biomedical Data Harmonisation Pipeline

## MSc Bioinformatics Dissertation Project

A reproducible Python and Nextflow pipeline for harmonising heterogeneous multimodal biomedical datasets into consistent, schema-compliant tables for downstream integration and analysis.

This repository presents the data-processing contribution developed for my MSc Bioinformatics dissertation at the University of Birmingham:

**Pipelines for Multimodal Biomedical Databases via Natural Language Queries**

## Project Overview

Biomedical datasets often use different file formats, identifiers, column names, and structures, making integration across studies difficult and time-consuming.

This project developed a modular data harmonisation framework that profiles incoming biomedical datasets, recommends mappings to a unified schema, applies validation and transformation rules, and produces standardised outputs suitable for downstream knowledge-graph integration.

The processing workflow supports:

- Entity / patient data
- Sample data
- Mutation data
- Structural variant data
- RNA-seq expression data
- DNA methylation data

## My Contribution

Within a wider collaborative biomedical knowledge-graph project, my work focused on the core data-processing layer.

I:

- Designed and implemented a unified data model for multimodal biomedical data
- Developed Python harmonisation modules for multiple biomedical data types
- Built profiling tools to inspect heterogeneous input datasets
- Developed mapping recommenders to assist schema alignment
- Implemented reusable transformation and validation logic
- Linked samples and molecular measurements through consistent identifiers
- Orchestrated the processing workflow using Nextflow
- Added validation, logging, error handling, and reproducible configuration
- Tested the framework across multiple publicly available cancer datasets

The Neo4j knowledge graph and natural-language query layer formed part of the wider collaborative project. This repository focuses specifically on the harmonisation and data-processing components developed as part of my dissertation work.

## Workflow

The onboarding workflow follows four main stages:

### 1. Profile

Raw input files are inspected to identify characteristics such as:

- Column names
- Data types
- Missingness
- Uniqueness
- Identifier patterns
- Genomic coordinate patterns
- Wide-matrix structures

### 2. Recommend

Module-specific recommender scripts use profiler evidence, lookup dictionaries, and the schema registry to suggest mappings between raw source columns and canonical output fields.

### 3. Review

Suggested mappings can be reviewed before processing so that important identifiers, required fields, and transformations can be checked.

### 4. Harmonise

The harmonisation modules apply approved mappings and reusable transformations before writing schema-compliant output tables.

```text
Raw biomedical data
        |
        v
     Profiler
        |
        v
 Mapping recommender
        |
        v
   Human review
        |
        v
   Harmonisation
        |
        v
Validation + standardisation
        |
        v
Schema-compliant CSV outputs
        |
        v
Downstream knowledge graph integration
```

## Supported Data Modalities

### Entity

Standardises biological-source and patient-level information such as identifiers, demographics, clinical information, organism, and survival-related data.

### Sample

Creates standardised sample records and links each sample to its corresponding entity.

### Mutation

Processes point mutations and small genomic variants, including genomic coordinates, reference and alternate alleles, gene associations, and mutation-related metadata.

### Structural Variant

Processes larger genomic alterations such as deletions, duplications, inversions, insertions, translocations, and other structural events.

### RNA-seq

Transforms gene-expression matrices into standardised long-format records linked to biological samples.

### Methylation

Processes methylation measurements and genomic annotations into standardised, sample-linked records.

## Reusable Transformations

Shared transformation utilities support operations such as:

- Removing chromosome prefixes
- Standardising text case
- Converting values to integers or floats
- Restricting values to valid numeric ranges
- Removing version suffixes from identifiers
- Selecting the first available non-null value

These transformations are defined centrally so that harmonisation behaviour can be reused across modules.

## Schema-Driven Design

The project uses a central schema registry to define:

- Canonical field names
- Expected data types
- Required and optional fields
- Field synonyms
- Input and output requirements
- Relationships between biomedical entities

This provides a consistent target structure across datasets and data modalities.

## Technologies

- Python
- pandas
- NumPy
- Nextflow
- JSON
- YAML
- Git
- GitHub
- Data validation
- ETL pipeline design
- Schema mapping
- Biomedical data processing

The wider research system used Neo4j for knowledge-graph integration and Cypher for graph querying.

## Evaluation

The pipeline was evaluated using **11 publicly available cancer genomics cohorts** from Xena and cBioPortal.

Across these datasets, the workflow processed:

- **19,551 entities**
- **23,582 samples**
- More than **1.2 million mutation records**
- More than **663,000 structural variant records**
- Approximately **43 million RNA-seq expression values**
- Approximately **54 million methylation measurements**

Individual dataset runs completed within a few minutes, with peak memory usage remaining below 1 GB during the reported evaluation.

## Repository Structure

```text
biomedical-data-harmonisation-pipeline/
├── scripts/
│   ├── profiler.py
│   ├── recommender_entity.py
│   ├── recommender_sample.py
│   ├── recommender_mutation.py
│   ├── recommender_sv.py
│   ├── recommender_rnaseq.py
│   ├── recommender_methylation.py
│   ├── entity.py
│   ├── sample.py
│   ├── mutation.py
│   ├── sv.py
│   ├── RNAseq.py
│   ├── methylation.py
│   └── transform_library.py
│
├── resources/
│   ├── schema_registry.json
│   ├── column_name_lookup.json
│   ├── column_content_lookup.json
│   ├── de_methods.json
│   ├── methylation_caller_software.json
│   ├── modification_type.json
│   ├── mutation_sources.json
│   ├── organism_map.json
│   ├── quantification_method.json
│   └── reference_genomes.json
│
├── main.nf
├── nextflow.config
├── params.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

## Data

The full biomedical datasets used during the dissertation are not included in this repository because of their size.

The evaluation used publicly available cancer genomics datasets obtained from sources including Xena and cBioPortal.

The `.gitignore` configuration prevents large raw biomedical files, temporary files, and generated workflow outputs from being accidentally committed.

## Reproducibility

Nextflow is used to coordinate the harmonisation workflow and support reproducible execution.

The modular design separates:

- Raw input data
- Profiling
- Mapping recommendations
- Configuration
- Transformation logic
- Harmonisation
- Validation
- Output generation

This structure allows compatible datasets to be processed through the same framework without rewriting the entire pipeline.

## Knowledge Graph Context

The harmonised outputs produced by this pipeline were designed for integration into a Neo4j biomedical knowledge graph.

The wider project represented biological and clinical entities and their relationships in a graph structure and supported natural-language access through translation of user questions into Cypher queries.

This repository focuses on the data harmonisation and processing layer rather than the complete collaborative knowledge-graph and natural-language-query system.

## Key Skills Demonstrated

- Python programming
- Data engineering
- ETL pipeline development
- Data cleaning and validation
- Schema design
- Data transformation
- Workflow automation
- Nextflow
- Large-scale biomedical data processing
- Reproducible research
- Technical documentation
- Git and GitHub

## Academic Context

This project was completed as part of my **MSc Bioinformatics at the University of Birmingham**, awarded with **Distinction**.

It demonstrates the application of software engineering, data engineering, bioinformatics, and reproducible workflow design to complex real-world biomedical data integration challenges.

## Author

**Ragad Alfatih**  
MSc Bioinformatics with Distinction  
Data Analyst | Business Intelligence | Bioinformatics
