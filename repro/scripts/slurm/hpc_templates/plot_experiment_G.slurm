#!/bin/bash
#
# Aggregate Experiment G JSONs (subset-size sweep + length-difficulty curve at
# the balanced parity regime) into CSVs, figures, and LaTeX tables.
#
# Submit:  sbatch scripts/slurm/hpc_templates/plot_experiment_G.slurm
#
#SBATCH --job-name=plot_expG
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/plot_expG_%j.out
#SBATCH --error=logs/plot_expG_%j.err

set -o errexit
set -o nounset

export PYTHONUNBUFFERED=1
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

REPO_ROOT="${REPO_ROOT:-${REPRO_ROOT:-$PWD}}"
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}
BERT_PKG_DIR="${SCRATCH:-/tmp}/python_packages/arm64_pt2506"

mkdir -p logs

CONTAINER_ENV="PYTHONUSERBASE=$BERT_PKG_DIR,PYTHONUNBUFFERED=1"

BINDS="--bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw"
BINDS="$BINDS --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw"

cd "$REPO_ROOT"

srun apptainer exec $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
    python3 analysis/experiment_G/scripts/summarize_subset_length.py \
        --results_dir "$RESULTS_DIR" \
        --output_dir  analysis/experiment_G/figures
