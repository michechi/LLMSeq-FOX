#!/bin/bash
#SBATCH --job-name=attn_bert
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=02:00:00
#SBATCH --partition=accel
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/attn_bert_%j.out
#SBATCH --error=logs/attn_bert_%j.err

set -o errexit
set -o nounset

export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE"
export HF_HOME="$SCRATCH_CACHE"
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"
export PYTHONUNBUFFERED=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN}"

export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
export HTTP_PROXY=http://10.63.2.48:3128/
export HTTPS_PROXY=http://10.63.2.48:3128/

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}

CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,http_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,https_proxy=http://10.63.2.48:3128/"

echo "=== BERT Attention Analysis ==="

srun apptainer exec --nv \
    --bind "$SCRATCH_CACHE:$SCRATCH_CACHE" \
    --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp} \
    --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD} \
    --env "$CONTAINER_ENV" \
    "$CONTAINER" \
    python3 -u -m src.analysis.attention_analysis \
        --model_type bert \
        --path_csv ${REPRO_ROOT:-$PWD}/data/simulation/tested/ \
        --number_to_use 9 \
        --output_dir ${SCRATCH:-/tmp}/results/attention_bert/ \
        --cache_dir "$SCRATCH_CACHE" \
        --batch_size 32 \
        --max_length 128 \
        --epochs 10 \
        --patience 3 \
        --n_samples 200 \
        --n_heatmaps 3

echo "=== DONE ==="
