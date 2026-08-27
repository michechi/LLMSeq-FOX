#!/bin/bash
#SBATCH --job-name=sensitivity_DL
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=2-24:00:00
#SBATCH --partition=accel
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/sensitivity_DL_%j.out
#SBATCH --error=logs/sensitivity_DL_%j.err

set -o errexit
set -o nounset

# ============== SETUP ==============
export PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"
export HF_HOME="$PERMANENT_CACHE"
export HF_DATASETS_CACHE="$PERMANENT_CACHE"
export XDG_CACHE_HOME="$PERMANENT_CACHE"
export PYTHONUNBUFFERED=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN environment variable before submitting}"

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

# ============== CONFIG ==============
DATA_DIR="${REPRO_ROOT:-$PWD}/data/simulation/sensitivity/"
OUTPUT_DIR="${SCRATCH:-/tmp}/results/sensitivity/"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}

CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"

# All 21 unique dataset IDs
DATASETS=(100 101 102 103 110 112 120 121 122 123 124 130 131 132 133 134 140 141 142 143 144)

mkdir -p "$OUTPUT_DIR"

# ============== RUN ==============
echo "=== Sensitivity: Transformer + LSTM on ${#DATASETS[@]} datasets ==="

for ID in "${DATASETS[@]}"; do
    echo ""
    echo "========== Dataset $ID =========="
    srun apptainer exec --nv \
        --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
        --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw \
        --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw \
        --env "$CONTAINER_ENV" \
        "$CONTAINER" \
        python3 -u ../../src/experiments/DL_TR_baselines_experiment.py \
            --number_to_use "$ID" \
            --path_csv "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --models Transformer,LSTM,RNNTransformer \
            --label_column Outcome \
            --optimal \
            --fractions "1.0" \
            --seed 9550
done

echo "=== ALL DONE ==="
