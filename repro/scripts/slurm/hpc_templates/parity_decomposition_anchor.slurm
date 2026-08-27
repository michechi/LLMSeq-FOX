#!/bin/bash
#
# Parity decomposition — binary anchor sweep on the HPC cluster.
#
# DL tasks (Transformer, LSTM):  --array=0-11  (2 ell × 2 models × 3 seeds)
# LLM tasks (Llama-3.2-1B+LoRA): --array=12-17 (2 ell × 1 model × 3 seeds)
# Use --array=0-5 to limit to ell=2 (mandatory sanity check).
#
# Submit with:
#   cd scripts/slurm/hpc_templates && sbatch parity_decomposition_anchor.slurm
#
#SBATCH --job-name=parity_anchor
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=08:00:00
#SBATCH --partition=accel
#SBATCH --mem=90G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-17
#SBATCH --output=logs/parity_anchor_%A_%a.out
#SBATCH --error=logs/parity_anchor_%A_%a.err

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
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"
DL_CONTAINER="${SCRATCH:-/tmp}/extended-pytorch/extended-pytorch.sif"
LLM_CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}

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

# Ensure anchor CSVs exist (built once by task 0 for each ell).
for ELL in 2 4; do
    K=1
    [ "$ELL" = "4" ] && K=2
    if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ ! -f "$DATA_DIR/X_train_anchor_l${ELL}_k${K}.csv" ]; then
        echo "[rank $SLURM_ARRAY_TASK_ID] building anchor ell=$ELL k=$K"
        srun apptainer exec --nv \
            --bind "$SCRATCH_CACHE:$SCRATCH_CACHE:rw" \
            --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
            --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw \
            --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw \
            --env "$CONTAINER_ENV" \
            "$DL_CONTAINER" \
            python3 -m src.data.parity_variants --build_anchor \
                --anchor_ell "$ELL" --anchor_k "$K" --output_dir "$DATA_DIR"
    fi
done

# Non-zero tasks wait for the anchor CSVs.
if [ "$SLURM_ARRAY_TASK_ID" != "0" ]; then
    for i in $(seq 1 60); do
        [ -f "$DATA_DIR/X_train_anchor_l2_k1.csv" ] && [ -f "$DATA_DIR/X_train_anchor_l4_k2.csv" ] && break
        echo "[rank $SLURM_ARRAY_TASK_ID] waiting for anchor CSVs... ($i/60)"
        sleep 15
    done
fi

# ============== GRID ==============
ELLS=(2 4)
DL_MODELS=("Transformer" "LSTM")
SEEDS=(0 1 2)

N_ELL=${#ELLS[@]}
N_DL=${#DL_MODELS[@]}
N_SEEDS=${#SEEDS[@]}

DL_TOTAL=$(( N_ELL * N_DL * N_SEEDS ))        # 12
LLM_TOTAL=$(( N_ELL * N_SEEDS ))              # 6

IDX=$SLURM_ARRAY_TASK_ID
cd "$REPO_ROOT"

if [ "$IDX" -lt "$DL_TOTAL" ]; then
    # DL job
    SEED_IDX=$(( IDX % N_SEEDS ))
    IDX=$(( IDX / N_SEEDS ))
    MODEL_IDX=$(( IDX % N_DL ))
    ELL_IDX=$(( IDX / N_DL ))
    ELL=${ELLS[$ELL_IDX]}
    MODEL=${DL_MODELS[$MODEL_IDX]}
    SEED=${SEEDS[$SEED_IDX]}

    echo "=== DL anchor ell=$ELL model=$MODEL seed=$SEED ==="
    srun apptainer exec --nv \
        --bind "$SCRATCH_CACHE:$SCRATCH_CACHE:rw" \
        --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
        --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw \
        --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw \
        --env "$CONTAINER_ENV" \
        "$DL_CONTAINER" \
        python3 -u -m src.experiments.parity_decomposition_dl \
            --variant bitonly \
            --model "$MODEL" \
            --size "binary${ELL}" \
            --seed "$SEED" \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --epochs 30 --patience 3 --batch_size 64
else
    # LLM job
    LLM_IDX=$(( IDX - DL_TOTAL ))
    if [ "$LLM_IDX" -ge "$LLM_TOTAL" ]; then
        echo "IDX $SLURM_ARRAY_TASK_ID out of range — exiting."
        exit 0
    fi
    SEED_IDX=$(( LLM_IDX % N_SEEDS ))
    ELL_IDX=$(( LLM_IDX / N_SEEDS ))
    ELL=${ELLS[$ELL_IDX]}
    SEED=${SEEDS[$SEED_IDX]}

    echo "=== LLM anchor ell=$ELL seed=$SEED ==="
    srun apptainer exec --nv \
        --bind "$SCRATCH_CACHE:$SCRATCH_CACHE:rw" \
        --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
        --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw \
        --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw \
        --env "$CONTAINER_ENV" \
        "$LLM_CONTAINER" \
        python3 -u -m src.experiments.parity_decomposition_llm \
            --variant bitonly \
            --size "binary${ELL}" \
            --seed "$SEED" \
            --model_name meta-llama/Llama-3.2-1B \
            --peft \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --cache_dir "$SCRATCH_CACHE" \
            --batch_size 64 --max_length 128 \
            --epochs 20 --patience 3 --lr 2e-5 --early loss
fi

echo "=== JOB FINISHED ==="
