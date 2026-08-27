#!/bin/bash
#
# Aggregate fewshot_llm_eval JSONs into a CSV + figure.
#
# Submit:  sbatch scripts/slurm/hpc_templates/plot_fewshot.slurm
#
#SBATCH --job-name=plot_fewshot
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/plot_fewshot_%j.out
#SBATCH --error=logs/plot_fewshot_%j.err

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
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH:-/tmp}/results/fewshot_llm}"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}
BERT_PKG_DIR="${SCRATCH:-/tmp}/python_packages/arm64_pt2506"

mkdir -p logs

CONTAINER_ENV="PYTHONUSERBASE=$BERT_PKG_DIR,PYTHONUNBUFFERED=1"

BINDS="--bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw"
BINDS="$BINDS --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw"

cd "$REPO_ROOT"

srun apptainer exec $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
    python3 analysis/fewshot_llm/scripts/summarize_fewshot.py \
        --results_dir "$RESULTS_DIR" \
        --output_dir  analysis/fewshot_llm/figures
