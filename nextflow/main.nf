nextflow.enable.dsl=2

// Folder where JSON lookup files live, relative to repo root
def LOOKUP_DIR = "${baseDir}/../resources/lookups"

// Folder where all the Python scripts live
def SCRIPTS_DIR  = "${baseDir}/../scripts"

/*
 * SmartWay ingest pipeline (entity → sample → mutation/sv/RNAseq/methylation)
 *
 * Assumes:
 *   - All Python scripts are on PATH (or in the container image).
 *   - JSON lookup files are available in the working directory of the process
 *     (or in the container’s working directory).
 */

params.outdir   = params.outdir   ?: "results"

// Input paths (set in params.yaml; default to empty list so update_lookups doesn't warn)
params.entity_file  = params.entity_file  ?: []
params.sample_file  = params.sample_file  ?: []

params.mutation_file        = params.mutation_file
params.mutation_source      = params.mutation_source
params.sv_file              = params.sv_file
params.sv_source            = params.sv_source
params.rnaseq_file          = params.rnaseq_file
params.methylation_file     = params.methylation_file
params.methylation_manifest = params.methylation_manifest  // optional


// Optional helper args for CLI
def mutSourceArg = params.mutation_source ? "--source-name ${params.mutation_source}" : ""
def svSourceArg  = params.sv_source       ? "--source-name ${params.sv_source}"       : ""


/**********************************************************************
 * PROCESSES
 *********************************************************************/

process ENTITY {
    tag "entity"

    publishDir "${params.outdir}", mode: 'copy'
    cpus 1
    memory '4 GB'
    time '4h'

    /*
     * entity_in is a list of one or more entity files.
     * We use a fixed outdir 'entity_work' inside the process work dir.
     */
    input:
    path entity_in

    output:
        path "entity_work/*entity_output_*.csv"

    script:
    """
    # Make JSON lookup files visible to entity.py
    cp "${LOOKUP_DIR}"/*.json .

    python "${SCRIPTS_DIR}/entity.py" ${entity_in} \
        --output entity_output.csv \
        --outdir entity_work
    """
}

process SAMPLE {
    tag "sample"

    publishDir "${params.outdir}", mode: 'copy'
    cpus 1
    memory '4 GB'
    time '4h'

    /*
     * sample_in: one or more sample files
     * entity_csv: the entity_output_*.csv produced by ENTITY
     */
    input:
    path sample_in
    path entity_csv

    // sample.py writes sample_output_*.csv in the SAME folder as entity_csv,
    // but Nextflow flattens entity_csv into the current work dir for SAMPLE.
    // So the output lives at ./sample_output_*.csv in this work dir.

     output:
        path "sample_output_*.csv"

    script:
    """
    # JSONs for sample.py
    cp "${LOOKUP_DIR}"/*.json .

    entity_folder=\$(dirname "${entity_csv}")
    echo "Using entity folder: \${entity_folder}"
    python "${SCRIPTS_DIR}/sample.py" ${sample_in} --entity-folder "\${entity_folder}"
    """
}

process MUTATION {
    tag "mutation"

    publishDir "${params.outdir}", mode: 'copy'
    cpus 2
    memory '4 GB'
    time '8h'

    input:
    path mut_in
    path sample_csv

    // mutation.py writes mutation_output_*.csv in the sample folder,
    // which is the current work dir for this process.
    output:
        path "mutation_output_*.csv"

    script:
    """
    # JSONs for mutation.py
    cp "${LOOKUP_DIR}"/*.json .

    sample_folder=\$(dirname "${sample_csv}")
    echo "Using sample folder for mutation: \${sample_folder}"

    python "${SCRIPTS_DIR}/mutation.py" ${mut_in} \
    --sample-folder "\${sample_folder}" \
    ${mutSourceArg}
    """
}

process SV {
    tag "sv"

    publishDir "${params.outdir}", mode: 'copy'
    cpus 2
    memory '4 GB'
    time '8h'

    input:
    path sv_in
    path sample_csv

    // sv.py writes sv_output_*.csv in the sample folder,
    // which is the current work dir for this process.
    output:
        path "sv_output_*.csv"

    script:
    """
    # JSONs for sv.py
    cp "${LOOKUP_DIR}"/*.json .

    sample_folder=\$(dirname "${sample_csv}")
    echo "Using sample folder for SV: \${sample_folder}"

    python "${SCRIPTS_DIR}/sv.py" ${sv_in} \
    --sample-folder "\${sample_folder}" \
    ${svSourceArg}
    """
}

process RNASEQ {
    tag "rnaseq"

    publishDir "${params.outdir}", mode: 'copy'
    cpus 2
    memory '4 GB'
    time '8h'

    input:
    path rnaseq_in
    path sample_csv

    // RNAseq.py writes rnaseq_output_*.csv in the sample folder
    output:
        path "rnaseq_output_*.csv"

    script:
    """
     # JSONs for RNAseq.py
    cp "${LOOKUP_DIR}"/*.json .

    sample_folder=\$(dirname "${sample_csv}")
    echo "Using sample folder for RNAseq: \${sample_folder}"

    python "${SCRIPTS_DIR}/RNAseq.py" ${rnaseq_in} \
    --sample-folder "\${sample_folder}"
    """
}

process METHYLATION {
    tag "methylation"

    publishDir "${params.outdir}", mode: 'copy'
    cpus 2
    memory '6 GB'
    time '24h'

    /*
     * We pass:
     *   - meth_in  = methylation data file
     *   - manifest_in (optional) = manifest file
     *   - sample_csv = sample_output_*.csv
     */

      /*
     * meth_in: list of 1 or 2 files
     *   - [methylation.tsv]
     *   - [methylation.tsv, manifest.csv]
     *
     * sample_csv: sample_output_*.csv from SAMPLE
     */
     
    input:
        path meth_in
        path sample_csv

   // methylation.py writes methylation_output_*.csv in the sample folder
    output:
        path "methylation_output_*.csv"
        
    script:
    """
    # JSONs for methylation.py
    cp "${LOOKUP_DIR}"/*.json .
    
    sample_folder=\$(dirname "${sample_csv}")
    echo "Using sample folder for methylation: \${sample_folder}"

    # meth_in expands to 1 or 2 positional paths
    python "${SCRIPTS_DIR}/methylation.py" ${meth_in} \
    --sample-folder "\${sample_folder}"
    """
}

/**********************************************************************
 * WORKFLOW DEFINITION
 *********************************************************************/

workflow {

    // ---- 1) Required: ENTITY and SAMPLE ----

    // Entity input(s): one or more files
    Channel
        .fromPath(params.entity_file)
        .collect()
        .set { ch_entity_files }

    // Sample input(s): one or more files
    Channel
        .fromPath(params.sample_file)
        .collect()
        .set { ch_sample_files }

    // Run ENTITY
    ENTITY(ch_entity_files)

    // Run SAMPLE with two separate channels:
    //   - ch_sample_files -> sample_in
    //   - ENTITY.out      -> entity_csv
    SAMPLE(ch_sample_files, ENTITY.out)

    // SAMPLE.out is the sample_output_*.csv inside entity_work
    def ch_sample_csv = SAMPLE.out


    // ---- 2) Optional: MUTATION ----

    if (params.mutation_file) {
        log.info "MUTATION: running with file: ${params.mutation_file}"

        Channel
            .fromPath(params.mutation_file)
            .set { ch_mut_files }

        // Two-channel call: mutation input + sample CSV
        MUTATION(ch_mut_files, ch_sample_csv)
    }
    else {
        log.info "MUTATION: skipping (no mutation_file provided)"
    }


    // ---- 3) Optional: SV ----

    if (params.sv_file) {
        log.info "SV: running with file: ${params.sv_file}"

        Channel
            .fromPath(params.sv_file)
            .set { ch_sv_files }

        SV(ch_sv_files, ch_sample_csv)
    }
    else {
        log.info "SV: skipping (no sv_file provided)"
    }


    // ---- 4) Optional: RNASEQ ----

    if (params.rnaseq_file) {
        log.info "RNASEQ: running with file: ${params.rnaseq_file}"

        Channel
            .fromPath(params.rnaseq_file)
            .set { ch_rna_files }

        RNASEQ(ch_rna_files, ch_sample_csv)
    }
    else {
        log.info "RNASEQ: skipping (no rnaseq_file provided)"
    }


    // ---- 5) Optional: METHYLATION (1 or 2 files, like ENTITY/SAMPLE) ----

    if (params.methylation_file) {
        // Build ordered list of methylation inputs
        def meth_paths = [ params.methylation_file ]
        if (params.methylation_manifest) {
            meth_paths << params.methylation_manifest
        }

        log.info "METHYLATION: running with files: ${meth_paths.join(', ')}"

        Channel
            .fromPath(meth_paths)
            .collect()
            .set { ch_meth_files }

        // ch_meth_files -> meth_in (list of 1 or 2 files)
        // ch_sample_csv -> sample_output_*.csv
        METHYLATION(ch_meth_files, ch_sample_csv)
    }
    else {
        log.info "METHYLATION: skipping (no methylation_file provided)"
    }
}
