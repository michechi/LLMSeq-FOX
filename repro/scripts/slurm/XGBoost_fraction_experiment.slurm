#!/bin/bash
#SBATCH --job-name=XGB_alph_666_tfidf
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=20:00:00
#SBATCH --partition=accel
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1

set -o errexit
set -o nounset

# ============== PARTE 1: SETUP CACHE E TOKEN ==============
export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE"

export HF_HOME="$SCRATCH_CACHE"
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN environment variable before submitting}"

echo "=== JOB CONFIGURATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Scratch Cache: $SCRATCH_CACHE"
echo "Free space in SCRATCH: $(df -h $SCRATCH | tail -1 | awk '{print $4}')"

# ============== PARTE 2: CARICA MODULI ==============
module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster

if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "ERROR: Apptainer/Singularity not found. Trying to load module..."
    module load Apptainer
fi

# ============== PARTE 3: PREPARA DIRECTORY ==============
mkdir -p ${SCRATCH:-/tmp}/cache/embeddings/

# ============== PARTE 4: COMANDO DI ESECUZIONE ==============
CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"

echo "=== STARTING CONTAINER ==="
echo "Container: ${SCRATCH:-/tmp}/extended-pytorch/extended-pytorch.sif"
echo "Environment: $CONTAINER_ENV"

# CHECK VERSIONE XGBOOST
echo "=== CHECKING XGBOOST VERSION ==="
srun apptainer exec --nv \
--bind "$SCRATCH_CACHE:$SCRATCH_CACHE" \
--bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp} \
--env "$CONTAINER_ENV" \
${SCRATCH:-/tmp}/extended-pytorch/extended-pytorch.sif \
python3 -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"

# Use $HOME for the local cache directory
srun apptainer exec --nv \
--bind "$SCRATCH_CACHE:$SCRATCH_CACHE" \
--bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp} \
--env "$CONTAINER_ENV" \
${SCRATCH:-/tmp}/extended-pytorch/extended-pytorch.sif \
python3 -u ./src/experiments/XGBoost_fraction_experiment.py \
--csv_to_use 'alph' \
--path_csv ${REPRO_ROOT:-$PWD}/data/simulation/tested/ \
--output_dir results_XGB \
--model_name meta-llama/Llama-3.1-8B \
--seed 666 \
--run_tfidf