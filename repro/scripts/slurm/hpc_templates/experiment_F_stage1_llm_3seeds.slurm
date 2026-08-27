#!/bin/bash
#
# Experiment F, Stage 1 (Llama-3.2-1B + LoRA) on main-paper parity, n=10.
# 3 seeds. Saves trainable state_dict (LoRA + classification head) for warm-start.
# RoPE is stateless so no positional-embedding transfer is needed for Stage 2.
#
#SBATCH --job-name=expF_s1_llm_3s
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=06:00:00
#SBATCH --partition=accel
#SBATCH --mem=90G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --output=logs/expF_s1_llm_3s_%A_%a.out
#SBATCH --error=logs/expF_s1_llm_3s_%A_%a.err

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
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp_llm_curriculum}"
CKPT_DIR="${CKPT_DIR:-$OUTPUT_DIR/ckpt}"
PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}
BERT_PKG_DIR="${SCRATCH:-/tmp}/python_packages/arm64_pt2506"

mkdir -p "$OUTPUT_DIR" "$CKPT_DIR" "$PERMANENT_CACHE"

PKG_SENTINEL="$BERT_PKG_DIR/.installed_ok"
if [ ! -f "$PKG_SENTINEL" ]; then
    echo "ERROR: $PKG_SENTINEL not found."
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

# Sentinel: curriculum CSVs must already exist (built by experiment_F_stage1_dl.slurm task 0).
N10_SENTINEL="$DATA_DIR/X_train_main_parity_n10.csv"
if [ ! -f "$N10_SENTINEL" ]; then
    echo "ERROR: $N10_SENTINEL missing. Run experiment_F_stage1_dl.slurm task 0 first."
    exit 1
fi

SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== JOB CONFIGURATION ==="
echo "Stage 1 Llama-3.2-1B + LoRA  seed=$SEED  n=10"

cd "$REPO_ROOT"

srun apptainer exec --nv $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
    python3 -u -m src.experiments.parity_decomposition_llm \
        --variant raw \
        --size main_n10 \
        --seed "$SEED" \
        --model_name meta-llama/Llama-3.2-1B \
        --peft \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --cache_dir "$SCRATCH_CACHE" \
        --batch_size 64 \
        --max_length 64 \
        --epochs 30 \
        --patience 5 \
        --lr 2e-5 \
        --early auc \
        --save_checkpoint "$CKPT_DIR/stage1_Llama32-1B_seed${SEED}.pt"

echo "=== JOB FINISHED ==="
