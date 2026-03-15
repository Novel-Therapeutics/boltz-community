#!/bin/bash
# =============================================================================
# Boltz Benchmark Setup — WSL2 Ubuntu 24.04 + RTX 4090
#
# Prerequisites:
#   - WSL2 with Ubuntu 24.04
#   - NVIDIA drivers installed on Windows (nvidia-smi works in WSL)
#   - conda or mamba
#
# Usage:
#   # Full setup (first time)
#   bash scripts/eval/setup_benchmark.sh setup
#
#   # Download eval data only
#   bash scripts/eval/setup_benchmark.sh download
#
#   # Run pilot benchmark (20 targets, quick validation)
#   bash scripts/eval/setup_benchmark.sh pilot
#
#   # Run dev benchmark (25 curated targets, fast iteration)
#   bash scripts/eval/setup_benchmark.sh dev
#
#   # Run full benchmark
#   bash scripts/eval/setup_benchmark.sh full
#
#   # Evaluate results after predictions complete
#   bash scripts/eval/setup_benchmark.sh evaluate
# =============================================================================

set -euo pipefail

# Ensure child processes are killed on exit/interrupt
trap 'kill 0 2>/dev/null; exit 1' INT TERM

# --- Configuration -----------------------------------------------------------

BENCH_DIR="${BENCH_DIR:-$HOME/boltz-benchmark}"
CONDA_ENV="${CONDA_ENV:-boltz-bench}"
OST_CONDA_ENV="${OST_CONDA_ENV:-ost-eval}"
REPO_URL="https://github.com/Novel-Therapeutics/boltz-community.git"
EVAL_DATA_ID="1JvHlYUMINOaqPTunI9wBYrfYniKgVmxf"  # Google Drive file ID

# OpenStructure version: 2.9.3 is closest available on bioconda to the 2.8.0
# used in the Boltz-1 paper. Note: lDDT-PLI scoring changed slightly in 2.9.0.
# For tracking regressions before/after our changes this is fine — we just need
# the SAME version across all our runs.
OST_VERSION="2.9.3"

# Prediction parameters (matching Boltz-1 paper)
RECYCLING_STEPS=10
SAMPLING_STEPS=200
DIFFUSION_SAMPLES=5
SEED=42
PREPROCESSING_THREADS=4  # 32 threads can OOM on WSL2

# Pilot subset size
PILOT_N=20

# Dev benchmark: curated representative set (fast iteration)
# Override via flags: --samples N --steps N --targets N --max-residues N
DEV_N="${DEV_N:-25}"
DEV_MAX_RESIDUES="${DEV_MAX_RESIDUES:-600}"
DEV_SAMPLING_STEPS="${DEV_SAMPLING_STEPS:-100}"
DEV_DIFFUSION_SAMPLES="${DEV_DIFFUSION_SAMPLES:-5}"

# --- Colors -------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Preflight checks --------------------------------------------------------

preflight() {
    info "Running preflight checks..."

    # Check WSL
    if ! grep -qi microsoft /proc/version 2>/dev/null; then
        warn "Not running in WSL — some steps may differ"
    fi

    # Check NVIDIA driver
    if ! command -v nvidia-smi &>/dev/null; then
        err "nvidia-smi not found. Install NVIDIA drivers on Windows first."
        exit 1
    fi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    ok "GPU: ${GPU_NAME} (${GPU_MEM} MiB)"

    # Check conda/mamba
    if command -v conda &>/dev/null; then
        ok "conda found: $(conda --version)"
    elif command -v mamba &>/dev/null; then
        ok "mamba found"
    else
        err "conda or mamba is required. Install miniforge: https://github.com/conda-forge/miniforge"
        exit 1
    fi
}

# --- Setup --------------------------------------------------------------------

setup_env() {
    info "Setting up benchmark environment at ${BENCH_DIR}"
    mkdir -p "${BENCH_DIR}"/{data,results,evals}

    # Clone or update repo
    if [ -d "${BENCH_DIR}/boltz-community" ]; then
        info "Updating existing repo..."
        cd "${BENCH_DIR}/boltz-community"
        git pull --ff-only || warn "git pull failed — using existing checkout"
    else
        info "Cloning boltz-community..."
        git clone "${REPO_URL}" "${BENCH_DIR}/boltz-community"
    fi

    # --- Boltz conda environment (GPU predictions) ---
    if conda env list | grep -q "^${CONDA_ENV} "; then
        info "Conda env '${CONDA_ENV}' already exists"
    else
        info "Creating conda env '${CONDA_ENV}' with Python 3.11..."
        conda create -n "${CONDA_ENV}" python=3.11 -y
    fi
    info "Installing boltz-community..."
    conda run -n "${CONDA_ENV}" pip install -e "${BENCH_DIR}/boltz-community[cuda]"
    conda run -n "${CONDA_ENV}" pip install -e "${BENCH_DIR}/boltz-community/tools/affinity-eval[dev]"
    conda run -n "${CONDA_ENV}" pip install gdown

    # Patch cuequivariance_ops GPU detection: pynvml fails on consumer GPUs
    # (e.g. RTX 4090) and silently falls back to hardcoded A6000 specs,
    # causing Triton kernels to be tuned for the wrong GPU profile.
    info "Patching cuequivariance_ops GPU detection..."
    conda run -n "${CONDA_ENV}" python "${BENCH_DIR}/boltz-community/scripts/eval/patch_cuequivariance.py"

    ok "Boltz env '${CONDA_ENV}' ready"

    # --- OpenStructure conda environment (CPU evaluation) ---
    # Separate env because OST has conflicting dependencies with PyTorch/boltz
    if conda env list | grep -q "^${OST_CONDA_ENV} "; then
        info "Conda env '${OST_CONDA_ENV}' already exists"
    else
        info "Creating conda env '${OST_CONDA_ENV}' with OpenStructure ${OST_VERSION}..."
        info "(This may take a few minutes)"
        conda create -n "${OST_CONDA_ENV}" -c conda-forge -c bioconda \
            "openstructure=${OST_VERSION}" -y
    fi
    ok "OpenStructure env '${OST_CONDA_ENV}' ready"

    # Verify OST works
    info "Verifying OpenStructure installation..."
    OST_VER=$(conda run -n "${OST_CONDA_ENV}" ost --version 2>&1 || true)
    if echo "${OST_VER}" | grep -q "OpenStructure"; then
        ok "OpenStructure: ${OST_VER}"
    else
        # Try compare-structures directly
        if conda run -n "${OST_CONDA_ENV}" ost compare-structures --help &>/dev/null; then
            ok "OpenStructure compare-structures is available"
        else
            warn "Could not verify OpenStructure — evaluation may not work"
        fi
    fi

    ok "Setup complete!"
    echo ""
    info "Next steps:"
    echo "  1. bash scripts/eval/setup_benchmark.sh download"
    echo "  2. bash scripts/eval/setup_benchmark.sh pilot"
}

# --- Download eval data -------------------------------------------------------

download_data() {
    info "Downloading Boltz-1 evaluation data from Google Drive..."

    ARCHIVE="${BENCH_DIR}/data/boltz_results_final.zip"

    if [ -d "${BENCH_DIR}/data/boltz_results_final" ]; then
        ok "Eval data already downloaded"
        return
    fi

    if [ -f "${ARCHIVE}" ]; then
        info "Archive already downloaded, extracting..."
    else
        info "Downloading (~2-5 GB, may take a while)..."
        conda run -n "${CONDA_ENV}" gdown \
            "https://drive.google.com/uc?id=${EVAL_DATA_ID}" -O "${ARCHIVE}" --fuzzy
    fi

    info "Extracting..."
    cd "${BENCH_DIR}/data"
    python3 -c "import zipfile; zipfile.ZipFile('${ARCHIVE}').extractall()"
    ok "Eval data ready at ${BENCH_DIR}/data/boltz_results_final/"

    # Show what we got
    echo ""
    info "Data contents:"
    ls -la "${BENCH_DIR}/data/boltz_results_final/"

    N_TEST=$(ls "${BENCH_DIR}/data/boltz_results_final/targets/test/" 2>/dev/null | wc -l)
    N_CASP=$(ls "${BENCH_DIR}/data/boltz_results_final/targets/casp15/" 2>/dev/null | wc -l)
    info "PDB test targets: ${N_TEST}"
    info "CASP15 targets: ${N_CASP}"
}

# --- Run predictions ----------------------------------------------------------

_run_boltz() {
    # Suppress noisy warnings from PyTorch, Lightning, and cuequivariance
    export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning:pytorch_lightning,ignore::UserWarning:cuequivariance_ops_torch,ignore::UserWarning:torch"
    export PL_DISABLE_TIPS=1
    conda run --no-capture-output -n "${CONDA_ENV}" "$@"
}

_run_ost() {
    conda run --no-capture-output -n "${OST_CONDA_ENV}" "$@"
}

_kill_orphans() {
    # Kill any leftover boltz/python GPU processes from previous runs
    local ORPHANS
    ORPHANS=$(ps aux | grep -E '[b]oltz predict|[p]ython.*boltz' | grep -v "$$" | awk '{print $2}' || true)
    if [ -n "${ORPHANS}" ]; then
        for PID in ${ORPHANS}; do
            warn "Killing orphaned process ${PID}"
            kill "${PID}" 2>/dev/null || true
        done
        sleep 1
    fi
}

_check_triton() {
    # Check for Triton quality issue on 4090 (upstream #391)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if echo "${GPU_NAME}" | grep -qi "4090"; then
        warn "RTX 4090 detected. Triton kernels may degrade prediction quality (upstream #391)."
        warn "Disabling Triton for benchmark accuracy: BOLTZ_USE_TRITON=0"
        export BOLTZ_USE_TRITON=0
    fi
}

run_pilot() {
    _kill_orphans
    info "Running pilot benchmark (${PILOT_N} targets)..."

    EVAL_DATA="${BENCH_DIR}/data/boltz_results_final"
    if [ ! -d "${EVAL_DATA}/targets/test" ]; then
        err "Eval data not found. Run: bash scripts/eval/setup_benchmark.sh download"
        exit 1
    fi

    _check_triton

    # Prepare pilot inputs from eval data
    PILOT_DIR="${BENCH_DIR}/data/pilot_inputs"

    # The eval data has: inputs/test/boltz/{queries/*.yaml, msa/*.csv}
    # The YAML files contain hardcoded MSA paths from the Boltz team's server.
    # We rewrite them to point to the local msa/ directory.
    INPUT_DIR="${EVAL_DATA}/inputs/test/boltz"
    if [ ! -d "${INPUT_DIR}/queries" ]; then
        err "Boltz input files not found at ${INPUT_DIR}/queries"
        info "Available contents:"
        ls "${INPUT_DIR}/" 2>/dev/null || echo "  (none)"
        exit 1
    fi

    if [ ! -d "${PILOT_DIR}/queries" ]; then
        info "Preparing pilot inputs (${PILOT_N} targets)..."
        mkdir -p "${PILOT_DIR}/queries" "${PILOT_DIR}/msa"

        # Select first N YAML files
        SELECTED=$(ls "${INPUT_DIR}/queries/" | head -n "${PILOT_N}")

        for YAML_FILE in ${SELECTED}; do
            STEM="${YAML_FILE%.yaml}"

            # Copy YAML and rewrite MSA paths to point to local msa/ directory
            sed "s|/data/rbg/users/[^ ]*/msa/|../msa/|g" \
                "${INPUT_DIR}/queries/${YAML_FILE}" > "${PILOT_DIR}/queries/${YAML_FILE}"

            # Copy corresponding MSA files
            for MSA_FILE in "${INPUT_DIR}"/msa/${STEM}*; do
                [ -f "${MSA_FILE}" ] && cp "${MSA_FILE}" "${PILOT_DIR}/msa/"
            done
        done
        ok "Prepared ${PILOT_N} pilot inputs with local MSA paths"
    else
        info "Pilot inputs already prepared"
    fi

    ACTUAL_N=$(ls "${PILOT_DIR}/queries/" | wc -l)
    info "Running predictions on ${ACTUAL_N} targets..."

    # Run Boltz-2 (default model)
    info "=== Boltz-2 predictions ==="
    _run_boltz boltz predict "${PILOT_DIR}/queries" \
        --out_dir "${BENCH_DIR}/results/pilot_boltz2" \
        --recycling_steps "${RECYCLING_STEPS}" \
        --sampling_steps "${SAMPLING_STEPS}" \
        --diffusion_samples "${DIFFUSION_SAMPLES}" \
        --seed "${SEED}" \
        --skip_bad_inputs \
        --preprocessing-threads "${PREPROCESSING_THREADS}" \
        2>&1 | tee "${BENCH_DIR}/results/pilot_boltz2.log"

    ok "Pilot Boltz-2 predictions complete"

    # Run Boltz-1
    info "=== Boltz-1 predictions ==="
    _run_boltz boltz predict "${PILOT_DIR}/queries" \
        --out_dir "${BENCH_DIR}/results/pilot_boltz1" \
        --recycling_steps "${RECYCLING_STEPS}" \
        --sampling_steps "${SAMPLING_STEPS}" \
        --diffusion_samples "${DIFFUSION_SAMPLES}" \
        --seed "${SEED}" \
        --skip_bad_inputs \
        --preprocessing-threads "${PREPROCESSING_THREADS}" \
        --model boltz1 \
        2>&1 | tee "${BENCH_DIR}/results/pilot_boltz1.log"

    ok "Pilot Boltz-1 predictions complete"

    echo ""
    info "Pilot predictions saved to:"
    echo "  Boltz-2: ${BENCH_DIR}/results/pilot_boltz2/"
    echo "  Boltz-1: ${BENCH_DIR}/results/pilot_boltz1/"
    echo ""
    info "Next: bash scripts/eval/setup_benchmark.sh evaluate pilot"
}


# --- Dev benchmark (curated representative set) ------------------------------

run_dev() {
    _kill_orphans
    info "Running DEV benchmark (${DEV_N} curated targets)..."

    EVAL_DATA="${BENCH_DIR}/data/boltz_results_final"
    INPUT_DIR="${EVAL_DATA}/inputs/test/boltz"
    DEV_DIR="${BENCH_DIR}/data/dev_inputs"

    if [ ! -d "${INPUT_DIR}/queries" ]; then
        err "Eval data not found. Run: bash scripts/eval/setup_benchmark.sh download"
        exit 1
    fi

    _check_triton

    if [ ! -d "${DEV_DIR}/queries" ]; then
        info "Curating dev target set..."

        # Generate target list using curation script
        REPO="${BENCH_DIR}/boltz-community"
        DEV_LIST="${DEV_DIR}/targets.txt"
        mkdir -p "${DEV_DIR}"

        _run_boltz python "${REPO}/scripts/eval/curate_dev_set.py" \
            "${INPUT_DIR}/queries/" \
            --num-targets "${DEV_N}" \
            --max-residues "${DEV_MAX_RESIDUES}" \
            --seed "${SEED}" \
            --output "${DEV_LIST}"

        # Copy selected targets with rewritten MSA paths
        mkdir -p "${DEV_DIR}/queries" "${DEV_DIR}/msa"

        while IFS= read -r TARGET_NAME; do
            YAML_FILE="${TARGET_NAME}.yaml"

            if [ ! -f "${INPUT_DIR}/queries/${YAML_FILE}" ]; then
                warn "Target YAML not found: ${YAML_FILE}, skipping"
                continue
            fi

            # Copy YAML and rewrite MSA paths
            sed "s|/data/rbg/users/[^ ]*/msa/|../msa/|g" \
                "${INPUT_DIR}/queries/${YAML_FILE}" > "${DEV_DIR}/queries/${YAML_FILE}"

            # Copy corresponding MSA files
            for MSA_FILE in "${INPUT_DIR}"/msa/${TARGET_NAME}*; do
                [ -f "${MSA_FILE}" ] && cp "${MSA_FILE}" "${DEV_DIR}/msa/"
            done
        done < "${DEV_LIST}"

        ok "Prepared $(ls "${DEV_DIR}/queries/" | wc -l) dev inputs"
    else
        info "Dev inputs already prepared"
    fi

    ACTUAL_N=$(ls "${DEV_DIR}/queries/" | wc -l)
    info "Running predictions on ${ACTUAL_N} curated targets..."

    info "Dev parameters: ${DEV_SAMPLING_STEPS} sampling steps, ${DEV_DIFFUSION_SAMPLES} sample(s)"

    # Run Boltz-2
    info "=== Boltz-2 predictions ==="
    _run_boltz boltz predict "${DEV_DIR}/queries" \
        --out_dir "${BENCH_DIR}/results/dev_boltz2" \
        --recycling_steps "${RECYCLING_STEPS}" \
        --sampling_steps "${DEV_SAMPLING_STEPS}" \
        --diffusion_samples "${DEV_DIFFUSION_SAMPLES}" \
        --seed "${SEED}" \
        --skip_bad_inputs \
        --preprocessing-threads "${PREPROCESSING_THREADS}" \
        2>&1 | tee "${BENCH_DIR}/results/dev_boltz2.log"

    ok "Dev Boltz-2 predictions complete"

    # Run Boltz-1
    info "=== Boltz-1 predictions ==="
    _run_boltz boltz predict "${DEV_DIR}/queries" \
        --out_dir "${BENCH_DIR}/results/dev_boltz1" \
        --recycling_steps "${RECYCLING_STEPS}" \
        --sampling_steps "${DEV_SAMPLING_STEPS}" \
        --diffusion_samples "${DEV_DIFFUSION_SAMPLES}" \
        --seed "${SEED}" \
        --skip_bad_inputs \
        --preprocessing-threads "${PREPROCESSING_THREADS}" \
        --model boltz1 \
        2>&1 | tee "${BENCH_DIR}/results/dev_boltz1.log"

    ok "Dev Boltz-1 predictions complete"

    echo ""
    info "Dev predictions saved to:"
    echo "  Boltz-2: ${BENCH_DIR}/results/dev_boltz2/"
    echo "  Boltz-1: ${BENCH_DIR}/results/dev_boltz1/"
    echo ""
    info "Next: bash scripts/eval/setup_benchmark.sh evaluate dev"
}


_prepare_inputs() {
    # Prepare inputs for a dataset by rewriting MSA paths to local paths
    local SRC_DIR="$1"
    local DST_DIR="$2"

    if [ -d "${DST_DIR}/queries" ]; then
        info "Inputs already prepared at ${DST_DIR}"
        return
    fi

    if [ ! -d "${SRC_DIR}/queries" ]; then
        err "Source queries not found at ${SRC_DIR}/queries"
        return 1
    fi

    mkdir -p "${DST_DIR}/queries" "${DST_DIR}/msa"

    info "Rewriting MSA paths for $(ls "${SRC_DIR}/queries/" | wc -l) inputs..."

    for YAML_FILE in "${SRC_DIR}"/queries/*.yaml; do
        BASENAME=$(basename "${YAML_FILE}")
        STEM="${BASENAME%.yaml}"

        # Rewrite MSA paths to relative ../msa/ paths
        sed "s|/data/rbg/users/[^ ]*/msa/|../msa/|g" \
            "${YAML_FILE}" > "${DST_DIR}/queries/${BASENAME}"

        # Copy corresponding MSA files
        for MSA_FILE in "${SRC_DIR}"/msa/${STEM}*; do
            [ -f "${MSA_FILE}" ] && cp "${MSA_FILE}" "${DST_DIR}/msa/"
        done
    done

    ok "Prepared inputs at ${DST_DIR}"
}

run_full() {
    _kill_orphans
    info "Running FULL benchmark (this will take many hours)..."

    EVAL_DATA="${BENCH_DIR}/data/boltz_results_final"
    if [ ! -d "${EVAL_DATA}/targets/test" ]; then
        err "Eval data not found. Run: bash scripts/eval/setup_benchmark.sh download"
        exit 1
    fi

    _check_triton

    for MODEL in boltz2 boltz1; do
        for DATASET in test casp15; do
            SRC_INPUT="${EVAL_DATA}/inputs/${DATASET}/boltz"
            PREPARED_INPUT="${BENCH_DIR}/data/full_inputs_${DATASET}"

            if [ ! -d "${SRC_INPUT}/queries" ]; then
                warn "Input dir not found: ${SRC_INPUT}/queries, skipping ${DATASET}"
                continue
            fi

            # Prepare inputs with local MSA paths
            _prepare_inputs "${SRC_INPUT}" "${PREPARED_INPUT}"

            RESULT_DIR="${BENCH_DIR}/results/full_${MODEL}_${DATASET}"

            if [ -d "${RESULT_DIR}" ]; then
                warn "Results already exist at ${RESULT_DIR}, skipping"
                warn "Delete the directory to re-run"
                continue
            fi

            N_INPUTS=$(ls "${PREPARED_INPUT}/queries/" | wc -l)
            info "=== ${MODEL} on ${DATASET} (${N_INPUTS} targets) ==="

            MODEL_FLAG=""
            if [ "${MODEL}" = "boltz1" ]; then
                MODEL_FLAG="--model boltz1"
            fi

            _run_boltz boltz predict "${PREPARED_INPUT}/queries" \
                --out_dir "${RESULT_DIR}" \
                --recycling_steps "${RECYCLING_STEPS}" \
                --sampling_steps "${SAMPLING_STEPS}" \
                --diffusion_samples "${DIFFUSION_SAMPLES}" \
                --seed "${SEED}" \
                --skip_bad_inputs \
        --preprocessing-threads "${PREPROCESSING_THREADS}" \
                ${MODEL_FLAG} \
                2>&1 | tee "${RESULT_DIR}.log"

            ok "${MODEL} on ${DATASET} complete"
        done
    done

    ok "Full benchmark predictions complete!"
    info "Next: bash scripts/eval/setup_benchmark.sh evaluate full"
}

# --- Evaluate results ---------------------------------------------------------

run_evaluate() {
    local SCOPE="${1:-pilot}"
    EVAL_DATA="${BENCH_DIR}/data/boltz_results_final"
    REPO="${BENCH_DIR}/boltz-community"

    info "Running OpenStructure evaluation (${SCOPE})..."
    info "Using OpenStructure ${OST_VERSION} via conda env '${OST_CONDA_ENV}'"

    for MODEL in boltz2 boltz1; do
        for DATASET in test casp15; do
            if [ "${SCOPE}" != "full" ] && [ "${DATASET}" = "casp15" ]; then
                continue
            fi

            if [ "${SCOPE}" = "full" ]; then
                PRED_DIR="${BENCH_DIR}/results/full_${MODEL}_${DATASET}"
            else
                PRED_DIR="${BENCH_DIR}/results/${SCOPE}_${MODEL}"
            fi

            # Find the predictions subdirectory (boltz creates boltz_results_<input_dir>/predictions/)
            # Use the deepest match that actually contains prediction subdirectories
            PRED_SUBDIR=""
            for CANDIDATE in $(find "${PRED_DIR}" -name "predictions" -type d 2>/dev/null); do
                if [ -n "$(ls -A "${CANDIDATE}" 2>/dev/null)" ]; then
                    PRED_SUBDIR="${CANDIDATE}"
                    break
                fi
            done
            if [ -z "${PRED_SUBDIR}" ]; then
                warn "No predictions found in ${PRED_DIR}, skipping"
                continue
            fi

            REF_DIR="${EVAL_DATA}/targets/${DATASET}"
            if [ ! -d "${REF_DIR}" ]; then
                warn "Reference dir not found: ${REF_DIR}, skipping"
                continue
            fi

            EVAL_DIR="${BENCH_DIR}/evals/${SCOPE}_${MODEL}_${DATASET}"
            mkdir -p "${EVAL_DIR}"

            info "=== Evaluating ${MODEL} on ${DATASET} ==="
            info "  Predictions: ${PRED_SUBDIR}"
            info "  References:  ${REF_DIR}"
            info "  Output:      ${EVAL_DIR}"

            # Use fewer samples for dev scope
            if [ "${SCOPE}" = "dev" ]; then
                NUM_SAMPLES="${DEV_DIFFUSION_SAMPLES}"
            else
                NUM_SAMPLES="${DIFFUSION_SAMPLES}"
            fi

            _run_ost python "${REPO}/scripts/eval/run_boltz_eval.py" \
                "${PRED_SUBDIR}" \
                "${REF_DIR}" \
                "${EVAL_DIR}" \
                --num-samples "${NUM_SAMPLES}"

            # Aggregate (runs in boltz env since it needs pandas/numpy)
            info "Aggregating results..."
            _run_boltz python "${REPO}/scripts/eval/aggregate_boltz_eval.py" \
                "${PRED_SUBDIR}" \
                "${EVAL_DIR}" \
                --output "${BENCH_DIR}/results/${SCOPE}_${MODEL}_${DATASET}_results.csv" \
                --num-samples "${NUM_SAMPLES}"

            ok "${MODEL} on ${DATASET} evaluation complete"
        done
    done

    # Print all summary files
    echo ""
    info "=========================================="
    info "  BENCHMARK RESULTS"
    info "=========================================="
    for SUMMARY in "${BENCH_DIR}"/results/*_summary.csv; do
        if [ -f "${SUMMARY}" ]; then
            echo ""
            info "$(basename "${SUMMARY}"):"
            cat "${SUMMARY}"
        fi
    done

    # Auto-clean after evaluation
    run_clean
}


# --- Clean up -----------------------------------------------------------------

run_clean() {
    info "Cleaning up..."

    # Remove processed/featurized data (regenerated each run)
    CLEANED=0
    for PROC_DIR in "${BENCH_DIR}"/results/*/processed/; do
        if [ -d "${PROC_DIR}" ]; then
            SIZE=$(du -sh "${PROC_DIR}" 2>/dev/null | cut -f1)
            rm -rf "${PROC_DIR}"
            info "Removed ${PROC_DIR} (${SIZE})"
            CLEANED=$((CLEANED + 1))
        fi
    done

    # Kill orphaned boltz/python GPU processes
    ORPHANS=$(ps aux | grep -E '[b]oltz predict|[p]ython.*boltz' | grep -v "$$" | awk '{print $2}')
    if [ -n "${ORPHANS}" ]; then
        for PID in ${ORPHANS}; do
            info "Killing orphaned process ${PID}"
            kill "${PID}" 2>/dev/null || true
        done
    fi

    if [ "${CLEANED}" -eq 0 ]; then
        info "Nothing to clean"
    else
        ok "Cleaned ${CLEANED} processed directories"
    fi

    # Report remaining disk usage
    if [ -d "${BENCH_DIR}" ]; then
        echo ""
        info "Disk usage:"
        du -sh "${BENCH_DIR}"/data/ 2>/dev/null | sed 's/^/  /'
        du -sh "${BENCH_DIR}"/results/ 2>/dev/null | sed 's/^/  /'
        du -sh "${BENCH_DIR}"/evals/ 2>/dev/null | sed 's/^/  /'
    fi
}


# --- Status -------------------------------------------------------------------

status() {
    info "Benchmark status:"
    echo ""

    # GPU
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
        || echo "  (not available)"
    echo ""

    # Conda envs
    echo "Conda environments:"
    if conda env list | grep -q "^${CONDA_ENV} "; then
        ok "Boltz env '${CONDA_ENV}' exists"
    else
        warn "Boltz env '${CONDA_ENV}' not created"
    fi
    if conda env list | grep -q "^${OST_CONDA_ENV} "; then
        ok "OpenStructure env '${OST_CONDA_ENV}' exists"
    else
        warn "OpenStructure env '${OST_CONDA_ENV}' not created"
    fi
    echo ""

    # Data
    echo "Data:"
    if [ -d "${BENCH_DIR}/data/boltz_results_final" ]; then
        N_TEST=$(ls "${BENCH_DIR}/data/boltz_results_final/targets/test/" 2>/dev/null | wc -l)
        N_CASP=$(ls "${BENCH_DIR}/data/boltz_results_final/targets/casp15/" 2>/dev/null | wc -l)
        ok "Eval data: ${N_TEST} test targets, ${N_CASP} CASP15 targets"
    else
        warn "Eval data not downloaded"
    fi
    echo ""

    # Results
    echo "Results:"
    for DIR in "${BENCH_DIR}"/results/*/; do
        if [ -d "${DIR}" ]; then
            N=$(find "${DIR}" -name "*.cif" 2>/dev/null | wc -l)
            echo "  $(basename "${DIR}"): ${N} CIF files"
        fi
    done

    # Summaries
    for SUMMARY in "${BENCH_DIR}"/results/*_summary.csv; do
        if [ -f "${SUMMARY}" ]; then
            echo "  $(basename "${SUMMARY}")"
        fi
    done
}

# --- Parse flags --------------------------------------------------------------

COMMAND="${1:-help}"
shift || true

while [ $# -gt 0 ]; do
    case "$1" in
        --samples)      DEV_DIFFUSION_SAMPLES="$2"; shift 2 ;;
        --steps)        DEV_SAMPLING_STEPS="$2"; shift 2 ;;
        --targets)      DEV_N="$2"; shift 2 ;;
        --max-residues) DEV_MAX_RESIDUES="$2"; shift 2 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# --- Main dispatch ------------------------------------------------------------

case "${COMMAND}" in
    setup)
        preflight
        setup_env
        ;;
    download)
        download_data
        ;;
    pilot)
        run_pilot
        ;;
    dev)
        run_dev
        ;;
    full)
        run_full
        ;;
    evaluate)
        run_evaluate "${EXTRA_ARGS[0]:-pilot}"
        ;;
    clean)
        run_clean
        ;;
    status)
        status
        ;;
    preflight)
        preflight
        ;;
    *)
        echo "Boltz Benchmark Runner"
        echo ""
        echo "Usage: bash scripts/eval/setup_benchmark.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup      Install dependencies, create conda envs"
        echo "  download   Download Boltz-1 evaluation data from Google Drive"
        echo "  pilot      Run pilot benchmark (${PILOT_N} targets, ~1-2 hours on 4090)"
        echo "  dev        Run dev benchmark (${DEV_N} curated targets)"
        echo "  full       Run full benchmark (all targets, ~1-3 days on 4090)"
        echo "  evaluate [pilot|dev|full]  Run OpenStructure eval on predictions"
        echo "  clean      Remove processed data and orphaned processes"
        echo "  status     Show current benchmark status"
        echo "  preflight  Check system prerequisites"
        echo ""
        echo "Dev flags (use with 'dev' command):"
        echo "  --samples N       Diffusion samples per target (default: ${DEV_DIFFUSION_SAMPLES})"
        echo "  --steps N         Sampling steps (default: ${DEV_SAMPLING_STEPS})"
        echo "  --targets N       Number of targets (default: ${DEV_N})"
        echo "  --max-residues N  Max residues per target (default: ${DEV_MAX_RESIDUES})"
        echo ""
        echo "Environment variables:"
        echo "  BENCH_DIR       Benchmark working directory (default: ~/boltz-benchmark)"
        echo "  CONDA_ENV       Boltz conda environment (default: boltz-bench)"
        echo "  OST_CONDA_ENV   OpenStructure conda environment (default: ost-eval)"
        ;;
esac
