#!/bin/bash
#
# Few-shot / in-context LLM evaluation on Tricky Det, Tricky Rnd, Parity.
#
# Grid: 3 models x 3 tasks x 3 shot counts x 3 seeds = 81 tasks.
# Uses the arm64 pytorch container + shared BERT_PKG_DIR (transformers is
# already pip-installed there by install_bert_deps.slurm).
#
# Submit (requires HF_TOKEN for gated Llama checkpoints):
#   HF_TOKEN=... sbatch scripts/slurm/hpc_templates/fewshot_llm.slurm
#
# Outputs one JSON per cell at $OUTPUT_DIR; the eval script skips cells whose
# JSON already exists, so resubmission is idempotent.
#
#SBATCH --job-name=fewshot_llm
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=02:00:00
#SBATCH --partition=accel
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-80
#SBATCH --output=logs/fewshot_llm_%A_%a.out
#SBATCH --error=logs/fewshot_llm_%A_%a.err

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
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/simulation/tested}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/fewshot_llm}"
PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}
BERT_PKG_DIR="${SCRATCH:-/tmp}/python_packages/arm64_pt2506"

mkdir -p "$OUTPUT_DIR" "$PERMANENT_CACHE"

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

PKG_SENTINEL="$BERT_PKG_DIR/.installed_ok"
if [ ! -f "$PKG_SENTINEL" ]; then
    echo "ERROR: $PKG_SENTINEL not found. Run install_bert_deps.slurm first."
    exit 1
fi

# ============== GRID ==============
# Outer to inner: MODEL, TASK, SHOTS, SEED.
MODELS=("meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-14B")
TASKS=("tricky_det" "tricky_rnd" "parity")
SHOTS=(0 5 20)
SEEDS=(0 1 2)

# Per-model batch size (smaller for 14B to fit 20-shot prompts).
BATCH_SIZES_1B=16
BATCH_SIZES_8B=8
BATCH_SIZES_14B=4

N_M=${#MODELS[@]}
N_T=${#TASKS[@]}
N_K=${#SHOTS[@]}
N_S=${#SEEDS[@]}
TOTAL=$((N_M * N_T * N_K * N_S))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID >= TOTAL $TOTAL, exiting."
    exit 0
fi

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( IDX % N_S ));  IDX=$(( IDX / N_S ))
SHOT_IDX=$(( IDX % N_K ));  IDX=$(( IDX / N_K ))
TASK_IDX=$(( IDX % N_T ));  IDX=$(( IDX / N_T ))
MODEL_IDX=$(( IDX % N_M ))

MODEL=${MODELS[$MODEL_IDX]}
TASK=${TASKS[$TASK_IDX]}
SHOT=${SHOTS[$SHOT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# 0-shot is deterministic — only run seed 0.
if [ "$SHOT" = "0" ] && [ "$SEED" != "0" ]; then
    echo "0-shot is seed-invariant; skipping seed=$SEED."
    exit 0
fi

case "$MODEL" in
    *1B*)  BATCH=$BATCH_SIZES_1B ;;
    *8B*)  BATCH=$BATCH_SIZES_8B ;;
    *14B*) BATCH=$BATCH_SIZES_14B ;;
    *)     BATCH=4 ;;
esac

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: model=$MODEL task=$TASK shots=$SHOT seed=$SEED batch=$BATCH"
echo "Container: $CONTAINER"

cd "$REPO_ROOT"

srun apptainer exec --nv \
    $BINDS \
    --env "$CONTAINER_ENV" \
    "$CONTAINER" \
    python3 -u -m src.experiments.fewshot_llm_eval \
        --model_name "$MODEL" \
        --task "$TASK" \
        --n_shots "$SHOT" \
        --seed "$SEED" \
        --n_test 1000 \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --cache_dir "$SCRATCH_CACHE" \
        --batch_size "$BATCH" \
        --max_length 3072

echo "=== JOB FINISHED ==="
