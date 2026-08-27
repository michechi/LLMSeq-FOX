#!/bin/bash
#
# Render Experiment A and B figures from the anchor JSONs, inside the arm64
# PyTorch container (same env used for training). Ensures matplotlib is in
# the shared userbase first.
#
# Submit:
#   sbatch scripts/slurm/hpc_templates/plot_experiments.slurm
#
# Figures land in analysis/experiment_{A,B}/figures/.
#
#SBATCH --job-name=plot_experiments
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=00:20:00
#SBATCH --partition=accel
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/plot_experiments_%j.out
#SBATCH --error=logs/plot_experiments_%j.err

set -o errexit
set -o nounset

export PYTHONUNBUFFERED=1
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
export HTTP_PROXY=http://10.63.2.48:3128/
export HTTPS_PROXY=http://10.63.2.48:3128/

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

REPO_ROOT="${REPO_ROOT:-${REPRO_ROOT:-$PWD}}"
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}
BERT_PKG_DIR="${SCRATCH:-/tmp}/python_packages/arm64_pt2506"

mkdir -p logs

CONTAINER_ENV="PYTHONUSERBASE=$BERT_PKG_DIR,PYTHONUNBUFFERED=1"
INSTALL_ENV="$CONTAINER_ENV,http_proxy=$http_proxy,https_proxy=$https_proxy"

BINDS="--bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw"
BINDS="$BINDS --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw"

cd "$REPO_ROOT"

# Install matplotlib into $BERT_PKG_DIR if missing.
if ! apptainer exec $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
        python3 -c "import matplotlib" 2>/dev/null; then
    echo "[prep] matplotlib missing, installing into $BERT_PKG_DIR"
    apptainer exec $BINDS --env "$INSTALL_ENV" "$CONTAINER" \
        pip install --user --quiet --no-cache-dir matplotlib
fi

echo "=== Plot Experiment A ==="
srun apptainer exec $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
    python3 analysis/experiment_A/scripts/plot_auc_vs_ell.py \
        --results_dir "$RESULTS_DIR" \
        --output_dir  analysis/experiment_A/figures \
        --csv_out     analysis/experiment_A/figures/experiment_A_aggregate.csv

echo "=== Plot Experiment B ==="
srun apptainer exec $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
    python3 analysis/experiment_B/scripts/plot_auc_vs_k.py \
        --results_dir "$RESULTS_DIR" \
        --output_dir  analysis/experiment_B/figures \
        --csv_out     analysis/experiment_B/figures/experiment_B_aggregate.csv

# Experiment C uses a different results directory.
RESULTS_DIR_C="${RESULTS_DIR_C:-${SCRATCH:-/tmp}/results/parity_streaming}"
if [ -d "$RESULTS_DIR_C" ] && ls "$RESULTS_DIR_C"/expC_*.json >/dev/null 2>&1; then
    echo "=== Plot Experiment C ==="
    srun apptainer exec $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
        python3 analysis/experiment_C/scripts/summarize_regime.py \
            --results_dir "$RESULTS_DIR_C" \
            --output_dir  analysis/experiment_C/figures \
            --csv_out     analysis/experiment_C/figures/experiment_C_aggregate.csv
else
    echo "=== Skipping Experiment C (no JSONs at $RESULTS_DIR_C) ==="
fi

echo "=== DONE ==="
ls -l analysis/experiment_A/figures/ analysis/experiment_B/figures/ analysis/experiment_C/figures/ 2>/dev/null
