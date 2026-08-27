#!/bin/bash
#
# Parity decomposition — DL sweep (Transformer, LSTM) on the HPC cluster.
#
# 3 variants × 2 models × 2 sizes × 3 seeds = 36 runs. Submit with:
#
#   cd scripts/slurm/hpc_templates && sbatch parity_decomposition_dl.slurm
#
# Outputs land in ${SCRATCH:-/tmp}/results/parity_decomp/.
#
#SBATCH --job-name=parity_decomp_dl
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=04:00:00
#SBATCH --partition=accel
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-35
#SBATCH --output=logs/parity_decomp_dl_%A_%a.out
#SBATCH --error=logs/parity_decomp_dl_%A_%a.err

set -o errexit
set -o nounset

# ============== CACHE + TOKEN ==============
export PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"
export HF_HOME="$PERMANENT_CACHE"
export HF_DATASETS_CACHE="$PERMANENT_CACHE"
export XDG_CACHE_HOME="$PERMANENT_CACHE"
export PYTHONUNBUFFERED=1

mkdir -p logs "$PERMANENT_CACHE"

# ============== MODULES ==============
module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${REPRO_ROOT:-$PWD}}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/simulation/parity_decomp}"
BASE_DATA_DIR="${BASE_DATA_DIR:-$REPO_ROOT/data/simulation/tested}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
CONTAINER="${SCRATCH:-/tmp}/extended-pytorch/extended-pytorch.sif"

mkdir -p "$OUTPUT_DIR" "$DATA_DIR"

CONTAINER_ENV="HF_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"

# Build variant CSVs only on task 0 (other tasks wait for the files to appear).
if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ ! -f "$DATA_DIR/X_train_parity_raw_400K.csv" ]; then
    echo "[rank $SLURM_ARRAY_TASK_ID] building full variant CSVs in $DATA_DIR"
    srun apptainer exec --nv \
        --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
        --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw \
        --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw \
        --env "$CONTAINER_ENV" \
        "$CONTAINER" \
        python3 -m src.data.parity_variants --build_full \
            --data_dir "$BASE_DATA_DIR" \
            --output_dir "$DATA_DIR"
else
    # Wait for task 0 to finish building the CSVs.
    for i in $(seq 1 60); do
        [ -f "$DATA_DIR/X_train_parity_raw_400K.csv" ] && break
        echo "[rank $SLURM_ARRAY_TASK_ID] waiting for variant CSVs... ($i/60)"
        sleep 15
    done
fi

# ============== GRID ==============
VARIANTS=("raw" "masked" "bitonly")
MODELS=("Transformer" "LSTM")
SIZES=("40K" "400K")
SEEDS=(0 1 2)

N_V=${#VARIANTS[@]}
N_M=${#MODELS[@]}
N_SIZES=${#SIZES[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL=$((N_V * N_M * N_SIZES * N_SEEDS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID >= TOTAL $TOTAL — exiting."
    exit 0
fi

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( IDX % N_SEEDS ))
IDX=$(( IDX / N_SEEDS ))
SIZE_IDX=$(( IDX % N_SIZES ))
IDX=$(( IDX / N_SIZES ))
MODEL_IDX=$(( IDX % N_M ))
VARIANT_IDX=$(( IDX / N_M ))

VARIANT=${VARIANTS[$VARIANT_IDX]}
MODEL=${MODELS[$MODEL_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: variant=$VARIANT model=$MODEL size=$SIZE seed=$SEED"
echo "Container: $CONTAINER"

cd "$REPO_ROOT"

srun apptainer exec --nv \
    --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
    --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw \
    --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw \
    --env "$CONTAINER_ENV" \
    "$CONTAINER" \
    python3 -u -m src.experiments.parity_decomposition_dl \
        --variant "$VARIANT" \
        --model "$MODEL" \
        --size "$SIZE" \
        --seed "$SEED" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 30 \
        --patience 3 \
        --batch_size 64

echo "=== JOB FINISHED ==="
