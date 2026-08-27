#!/bin/bash
#
# Experiment F, Stage 2 (BERT): warm-started training on main-paper parity at n=20.
#
# Grid: BERT x seed in {0,1,2} = 3 tasks.
# Warm-starts from matched-seed Stage 1 checkpoint in $CKPT_DIR.
#
# Submit with dependency on Stage 1 BERT jobid (see experiment_F_launch.sh).
#
#SBATCH --job-name=expF_stage2_bert
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=04:00:00
#SBATCH --partition=accel
#SBATCH --mem=90G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --output=logs/expF_stage2_bert_%A_%a.out
#SBATCH --error=logs/expF_stage2_bert_%A_%a.err

set -o errexit
set -o nounset

export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE" logs

export HF_HOME="$SCRATCH_CACHE"
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"
export PYTHONUNBUFFERED=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN before submitting}"

export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
export HTTP_PROXY=http://10.63.2.48:3128/
export HTTPS_PROXY=http://10.63.2.48:3128/

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

REPO_ROOT="${REPO_ROOT:-${REPRO_ROOT:-$PWD}}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/simulation/curriculum}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
CKPT_DIR="${CKPT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp/ckpt_expF}"
PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}
BERT_PKG_DIR="${SCRATCH:-/tmp}/python_packages/arm64_pt2506"

mkdir -p "$OUTPUT_DIR" "$PERMANENT_CACHE"

PKG_SENTINEL="$BERT_PKG_DIR/.installed_ok"
if [ ! -f "$PKG_SENTINEL" ]; then
    echo "ERROR: $PKG_SENTINEL not found. Run install_bert_deps.slurm first."
    exit 1
fi

CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUSERBASE=$BERT_PKG_DIR"
CONTAINER_ENV="$CONTAINER_ENV,http_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,https_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,HTTP_PROXY=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,HTTPS_PROXY=http://10.63.2.48:3128/"

BINDS="--bind $SCRATCH_CACHE:$SCRATCH_CACHE:rw"
BINDS="$BINDS --bind $PERMANENT_CACHE:$PERMANENT_CACHE:rw"
BINDS="$BINDS --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw"
BINDS="$BINDS --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw"

SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CKPT="$CKPT_DIR/stage1_BERT_seed${SEED}.pt"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Stage 1 BERT checkpoint $CKPT missing. Run experiment_F_stage1_bert.slurm first."
    exit 1
fi

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Config: model=BERT seed=$SEED stage=2 (n=20, warm from n=10)"
echo "Init checkpoint: $CKPT"

cd "$REPO_ROOT"

srun apptainer exec --nv $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
    python3 -u -m src.experiments.parity_decomposition_bert \
        --variant raw \
        --size main_n20 \
        --seed "$SEED" \
        --model_name google-bert/bert-base-uncased \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --cache_dir "$SCRATCH_CACHE" \
        --batch_size 32 \
        --max_length 128 \
        --epochs 20 \
        --patience 3 \
        --lr 2e-5 \
        --early loss \
        --init_checkpoint "$CKPT"

echo "=== JOB FINISHED ==="
