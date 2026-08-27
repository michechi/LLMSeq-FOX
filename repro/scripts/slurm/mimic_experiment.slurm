#!/bin/bash
# ==============================================================
# MIMIC CKD->ESRD ordered vs shuffled  -  parametric SLURM template
# ==============================================================
# Submit via submit_all_mimic.sh, or manually:
#
#   sbatch --job-name=mimic_transformer_ordered \
#          --output=./logs/mimic_transformer_ordered_%j.out \
#          --time=04:00:00 --mem=32GB \
#          --export=ALL,MODEL=transformer,DATA=ordered,EXTRA_ARGS="" \
#          scripts/slurm/mimic_experiment.slurm
#
# Required env (export before sbatch, or rely on inheritance):
#   REPRO_ROOT  : path to the repro repo root
#   CONTAINER   : path to your apptainer / singularity image (Python 3.10+,
#                 torch 2.x, transformers, peft, scikit-learn, xgboost)
#   HF_TOKEN    : HuggingFace token (only needed for gated models, e.g. Llama)
# ==============================================================

#SBATCH --job-name=mimic_experiment
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=accel
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/mimic_experiment_%j.out

set -o errexit
set -o nounset

: "${MODEL:?ERROR: MODEL not set (transformer|lstm|bilstm|bert|llama|xgboost|logreg)}"
: "${DATA:?ERROR: DATA not set (ordered|shuffled)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

REPO=${REPRO_ROOT:?ERROR: export REPRO_ROOT=/path/to/repro}
CONTAINER=${CONTAINER:?ERROR: export CONTAINER=/path/to/your/pytorch.sif}

# ============== CACHE ==============
export PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"
mkdir -p "$PERMANENT_CACHE"

export HF_HOME="$PERMANENT_CACHE"
export HF_DATASETS_CACHE="$PERMANENT_CACHE"
export XDG_CACHE_HOME="$PERMANENT_CACHE"
export PYTHONUNBUFFERED=1

echo "=== JOB CONFIGURATION ==="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Repo:     $REPO"
echo "Model:    $MODEL"
echo "Data:     $DATA"
echo "Extra:    $EXTRA_ARGS"
echo "Cache:    $PERMANENT_CACHE"

# ============== MODULES ==============
module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster

if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "Apptainer/Singularity not on PATH; trying module load Apptainer..."
    module load Apptainer
fi

cd "$REPO"
mkdir -p logs results/mimic

# ============== CONTAINER ENV ==============
CONTAINER_ENV="PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,REPRO_ROOT=$REPO"
if [[ -n "${HF_TOKEN:-}" ]]; then
    CONTAINER_ENV="$CONTAINER_ENV,HUGGING_FACE_HUB_TOKEN=$HF_TOKEN,HF_TOKEN=$HF_TOKEN"
fi

# ============== RUN ==============
srun apptainer exec --nv \
    --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
    --bind "$REPO:$REPO" \
    --pwd "$REPO" \
    --env "$CONTAINER_ENV" \
    "$CONTAINER" \
    python3 -u -m src.mimic.train_mimic \
        --model "$MODEL" \
        --data "$DATA" \
        $EXTRA_ARGS

echo "=== JOB DONE ==="
