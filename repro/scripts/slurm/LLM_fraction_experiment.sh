#!/bin/bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=1-24:00:00
#SBATCH --partition=accel
#SBATCH --mem=90GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1

set -o errexit
set -o nounset

echo "=== RUN: ${RUN_NAME:?ERROR: RUN_NAME not set} ==="

# ============== EXPERIMENT PARAMETERS (from wrapper or defaults) ==============
MODEL_NAME="${EXP_MODEL:-meta-llama/Llama-3.2-1B}"
SEED="${EXP_SEED:-8888}"
BATCH_SIZE="${EXP_BATCH_SIZE:-64}"
MAX_LENGTH="${EXP_MAX_LENGTH:-50}"
NUMBER_TO_USE="${EXP_NUMBER:-9}"
FRACTIONS="${EXP_FRACTIONS:-1.0}"
PEFT="${EXP_PEFT:-true}"
QUANTIZATION="${EXP_QUANTIZATION:-false}"
LR="${EXP_LR:-2e-5}"
EPOCHS="${EXP_EPOCHS:-20}"
PATIENCE="${EXP_PATIENCE:-3}"

echo "=== EXPERIMENT PARAMETERS ==="
echo "Model: $MODEL_NAME"
echo "Seed: $SEED"
echo "Batch size: $BATCH_SIZE"
echo "Max length: $MAX_LENGTH"
echo "Dataset: $NUMBER_TO_USE"
echo "Fractions: $FRACTIONS"
echo "PEFT: $PEFT"
echo "Quantization: $QUANTIZATION"
echo "LR: $LR"
echo "Epochs: $EPOCHS"
echo "Patience: $PATIENCE"

# ============== SETUP CACHE E TOKEN ==============
export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE"

export HF_HOME="$SCRATCH_CACHE"
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN environment variable before submitting}"

# ============== PROXY PER INTERNET ==============
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
export HTTP_PROXY=http://10.63.2.48:3128/
export HTTPS_PROXY=http://10.63.2.48:3128/

echo "=== JOB CONFIGURATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Scratch Cache: $SCRATCH_CACHE"
echo "Free space in SCRATCH: $(df -h $SCRATCH | tail -1 | awk '{print $4}')"

# ============== CARICA MODULI ==============
module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster

if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "ERROR: Apptainer/Singularity not found. Trying to load module..."
    module load Apptainer
fi

# ============== PREPARA DIRECTORY ==============
mkdir -p ${SCRATCH:-/tmp}/results/llm_fraction/

# ============== BUILD PYTHON COMMAND ==============
CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,http_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,https_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,HTTP_PROXY=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,HTTPS_PROXY=http://10.63.2.48:3128/"

# Build optional flags
OPTIONAL_FLAGS=""
if [ "$PEFT" = "true" ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --peft"
fi
if [ "$QUANTIZATION" = "true" ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --use_quantization"
fi

echo "=== STARTING CONTAINER ==="
echo "Container: ${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}
echo "Optional flags: $OPTIONAL_FLAGS"

# Run the LLM fraction experiment
srun apptainer exec --nv \
--bind "$SCRATCH_CACHE:$SCRATCH_CACHE" \
--bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp} \
--bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD} \
--env "$CONTAINER_ENV" \
${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif} \
python3 -u ./src/experiments/LLM_fraction_experiment.py \
--number_to_use "$NUMBER_TO_USE" \
--path_csv ${REPRO_ROOT:-$PWD}/data/simulation/tested/ \
--cache_dir "${SCRATCH:-/tmp}/cache/" \
--output_dir ${SCRATCH:-/tmp}/results/llm_fraction/ \
--model_name "$MODEL_NAME" \
--batch_size "$BATCH_SIZE" \
--max_length "$MAX_LENGTH" \
--epochs "$EPOCHS" \
--patience "$PATIENCE" \
--lr "$LR" \
--seed "$SEED" \
--fractions "$FRACTIONS" \
$OPTIONAL_FLAGS
