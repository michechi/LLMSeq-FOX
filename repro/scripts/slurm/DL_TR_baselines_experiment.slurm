#!/bin/bash
#SBATCH --job-name=DL_TR_RNNTransformerNaive_exp
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=15:00:00
#SBATCH --partition=accel
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/Transformers_Naive_%j.out
#SBATCH --error=slurm_logs/Transformers_Naive_%j.err

set -o errexit
set -o nounset

# ============== PARTE 1: SETUP CACHE ==============
export PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"

export HF_HOME="$PERMANENT_CACHE"
export HF_DATASETS_CACHE="$PERMANENT_CACHE"
export XDG_CACHE_HOME="$PERMANENT_CACHE"
export PYTHONUNBUFFERED=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN environment variable before submitting}"

echo "=== JOB CONFIGURATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Permanent Cache: $PERMANENT_CACHE"

# ============== PARTE 2: CARICA MODULI ==============
module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster

if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "ERROR: Apptainer/Singularity not found. Trying to load module..."
    module load Apptainer
fi

# ============== PARTE 3: COMANDO DI ESECUZIONE ==============
CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"

echo "=== STARTING CONTAINER ==="
echo "Container: ${SCRATCH:-/tmp}/extended-pytorch/extended-pytorch.sif"
echo "Environment: $CONTAINER_ENV"

srun apptainer exec --nv \
    --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
    --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw \
    --env "$CONTAINER_ENV" \
    ${SCRATCH:-/tmp}/extended-pytorch/extended-pytorch.sif \
    python3 -u src/experiments/DL_TR_baselines_experiment.py \
        ${@}  # Pass all arguments from command line

# ============== PARTE 4: NESSUNA PULIZIA ==============
JOB_EXIT_CODE=$?
echo "=== JOB FINISHED ==="
echo "Exit code: $JOB_EXIT_CODE"
echo "Cache permanente mantenuta in: $PERMANENT_CACHE"

exit $JOB_EXIT_CODE