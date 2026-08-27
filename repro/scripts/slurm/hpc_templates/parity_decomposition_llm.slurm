#!/bin/bash
#
# Parity decomposition — LLM sweep (Llama-3.2-1B + LoRA) on the HPC cluster.
#
# 3 variants × 2 sizes × 3 seeds = 18 runs. Submit with:
#
#   cd scripts/slurm/hpc_templates && sbatch parity_decomposition_llm.slurm
#
# Outputs land in ${SCRATCH:-/tmp}/results/parity_decomp/.
#
#SBATCH --job-name=parity_decomp_llm
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=10:00:00
#SBATCH --partition=accel
#SBATCH --mem=90G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-17
#SBATCH --output=logs/parity_decomp_llm_%A_%a.out
#SBATCH --error=logs/parity_decomp_llm_%A_%a.err

set -o errexit
set -o nounset

# ============== CACHE + TOKEN ==============
export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE" logs

export HF_HOME="$SCRATCH_CACHE"
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"
export PYTHONUNBUFFERED=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN environment variable before submitting}"

# Proxy required for HuggingFace hub access on the HPC cluster compute nodes
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
export HTTP_PROXY=http://10.63.2.48:3128/
export HTTPS_PROXY=http://10.63.2.48:3128/

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${REPRO_ROOT:-$PWD}}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/simulation/parity_decomp}"
BASE_DATA_DIR="${BASE_DATA_DIR:-$REPO_ROOT/data/simulation/tested}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}

mkdir -p "$OUTPUT_DIR" "$DATA_DIR" "$PERMANENT_CACHE"

CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,http_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,https_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,HTTP_PROXY=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,HTTPS_PROXY=http://10.63.2.48:3128/"

# Build variant CSVs only on task 0; other tasks wait.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ ! -f "$DATA_DIR/X_train_parity_raw_400K.csv" ]; then
    echo "[rank $SLURM_ARRAY_TASK_ID] building full variant CSVs in $DATA_DIR"
    srun apptainer exec --nv \
        --bind "$SCRATCH_CACHE:$SCRATCH_CACHE:rw" \
        --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
        --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw \
        --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw \
        --env "$CONTAINER_ENV" \
        "$CONTAINER" \
        python3 -m src.data.parity_variants --build_full \
            --data_dir "$BASE_DATA_DIR" \
            --output_dir "$DATA_DIR"
else
    for i in $(seq 1 60); do
        [ -f "$DATA_DIR/X_train_parity_raw_400K.csv" ] && break
        echo "[rank $SLURM_ARRAY_TASK_ID] waiting for variant CSVs... ($i/60)"
        sleep 15
    done
fi

# ============== GRID ==============
VARIANTS=("raw" "masked" "bitonly")
SIZES=("40K" "400K")
SEEDS=(0 1 2)

N_V=${#VARIANTS[@]}
N_SIZES=${#SIZES[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL=$((N_V * N_SIZES * N_SEEDS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID >= TOTAL $TOTAL — exiting."
    exit 0
fi

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( IDX % N_SEEDS ))
IDX=$(( IDX / N_SEEDS ))
SIZE_IDX=$(( IDX % N_SIZES ))
VARIANT_IDX=$(( IDX / N_SIZES ))

VARIANT=${VARIANTS[$VARIANT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: variant=$VARIANT size=$SIZE seed=$SEED"
echo "Container: $CONTAINER"

cd "$REPO_ROOT"

srun apptainer exec --nv \
    --bind "$SCRATCH_CACHE:$SCRATCH_CACHE:rw" \
    --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
    --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw \
    --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw \
    --env "$CONTAINER_ENV" \
    "$CONTAINER" \
    python3 -u -m src.experiments.parity_decomposition_llm \
        --variant "$VARIANT" \
        --size "$SIZE" \
        --seed "$SEED" \
        --model_name meta-llama/Llama-3.2-1B \
        --peft \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --cache_dir "$SCRATCH_CACHE" \
        --batch_size 64 \
        --max_length 128 \
        --epochs 20 \
        --patience 3 \
        --lr 2e-5 \
        --early loss

echo "=== JOB FINISHED ==="
