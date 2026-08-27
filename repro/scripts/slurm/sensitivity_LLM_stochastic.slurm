#!/bin/bash
#SBATCH --job-name=sensitivity_stochastic_Llama1B
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=5-00:00:00
#SBATCH --partition=accel
#SBATCH --mem=90G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/sensitivity_stochastic_Llama1B_%j.out
#SBATCH --error=logs/sensitivity_stochastic_Llama1B_%j.err

set -o errexit
set -o nounset

# ============== SETUP ==============
export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE"

export HF_HOME="$SCRATCH_CACHE"
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"
export PYTHONUNBUFFERED=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN environment variable before submitting}"

# Proxy
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
export HTTP_PROXY=http://10.63.2.48:3128/
export HTTPS_PROXY=http://10.63.2.48:3128/

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

# ============== CONFIG ==============
DATA_DIR="${REPRO_ROOT:-$PWD}/data/simulation/sensitivity_stochastic/"
OUTPUT_DIR="${SCRATCH:-/tmp}/results/sensitivity_stochastic/"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}

CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,http_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,https_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,HTTP_PROXY=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,HTTPS_PROXY=http://10.63.2.48:3128/"

DATASETS=(100 101 102 103 110 112 120 121 122 123 124 130 131 132 133 134 140 141 142 143 144)

mkdir -p "$OUTPUT_DIR"

# ============== RUN ==============
echo "=== Sensitivity: Llama-3.2-1B on ${#DATASETS[@]} datasets ==="

for ID in "${DATASETS[@]}"; do
    echo ""
    echo "========== Dataset $ID (Llama-1B) =========="
    srun apptainer exec --nv \
        --bind "$SCRATCH_CACHE:$SCRATCH_CACHE" \
        --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp} \
        --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD} \
        --env "$CONTAINER_ENV" \
        "$CONTAINER" \
        python3 -u ../../src/experiments/LLM_fraction_experiment.py \
            --number_to_use "$ID" \
            --path_csv "$DATA_DIR" \
            --cache_dir "${SCRATCH:-/tmp}/cache/" \
            --output_dir "$OUTPUT_DIR" \
            --model_name meta-llama/Llama-3.2-1B \
            --peft \
            --batch_size 64 \
            --max_length 100 \
            --epochs 20 \
            --patience 3 \
            --lr 2e-5 \
            --seed 9550 \
            --fractions "1.0"
done

echo "=== ALL DONE ==="
