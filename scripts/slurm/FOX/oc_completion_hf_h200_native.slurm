#!/bin/bash -l
# Full OC matched-completion Hugging Face training on a FOX H200.
#
# Each array element is exactly one (arm, task, seed) run and therefore gets
# its own 24-hour allocation.  Fifteen minutes before the limit, the batch
# shell requeues that element; train_hf resumes from its last completed epoch.
# One epoch must fit inside an allocation.
#
# Primary array (0-13):
#   0-5    BERT LoRA,       2 tasks x 3 seeds
#   6-11   Llama-1B LoRA,   2 tasks x 3 seeds
#   12-13  Llama-1B full,   2 tasks x seed 9550
# Optional full-FT extension (14-17): seeds 9551 and 9552.
#
# Prefer the companion submit wrapper instead of calling sbatch directly:
#   bash scripts/slurm/FOX/submit_oc_completion_hf_h200_native.sh primary
#
#SBATCH --job-name=oc_hf_train
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-13%1
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_hf_train_%A_%a.out

set -o errexit
set -o nounset
set -o pipefail

# ============== ARRAY MAP ==============
CONFIGS=(
    "bert_lora ocdet 9550"
    "bert_lora ocdet 9551"
    "bert_lora ocdet 9552"
    "bert_lora ocnoisy 9550"
    "bert_lora ocnoisy 9551"
    "bert_lora ocnoisy 9552"
    "llama_lora ocdet 9550"
    "llama_lora ocdet 9551"
    "llama_lora ocdet 9552"
    "llama_lora ocnoisy 9550"
    "llama_lora ocnoisy 9551"
    "llama_lora ocnoisy 9552"
    "llama_full ocdet 9550"
    "llama_full ocnoisy 9550"
    "llama_full ocdet 9551"
    "llama_full ocnoisy 9551"
    "llama_full ocdet 9552"
    "llama_full ocnoisy 9552"
)

IDX="$SLURM_ARRAY_TASK_ID"
if (( IDX < 0 || IDX >= ${#CONFIGS[@]} )); then
    echo "ERROR: unsupported array index $IDX" >&2
    exit 2
fi
read -r ARM TASK SEED <<< "${CONFIGS[$IDX]}"

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
if [[ ! -d "$REPO_ROOT/repro/src/oc_completion" ]]; then
    echo "ERROR: REPO_ROOT is not an LLMSeq checkout: $REPO_ROOT" >&2
    exit 2
fi

: "${OC_RUN_ROOT:?ERROR: export OC_RUN_ROOT to a project-storage directory}"
if [[ "$OC_RUN_ROOT" != /* ]]; then
    echo "ERROR: OC_RUN_ROOT must be absolute: $OC_RUN_ROOT" >&2
    exit 2
fi
mkdir -p "$OC_RUN_ROOT"
OC_RUN_ROOT="$(realpath -e -- "$OC_RUN_ROOT")"

HOME_ROOT="$(realpath -e -- "$HOME")"
case "$OC_RUN_ROOT/" in
    "$HOME_ROOT/"*)
        echo "ERROR: full-run outputs must not use quota-limited home storage." >&2
        echo "Set OC_RUN_ROOT under /fp/projects01 (current: $OC_RUN_ROOT)." >&2
        exit 2
        ;;
esac

export DATA_DIR="$REPO_ROOT/data"
export LLMSEQ_ROOT="$OC_RUN_ROOT"
export HF_HOME="${HF_HOME:-$OC_RUN_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [[ "$HF_HOME" != /* || "$HF_HUB_CACHE" != /* ]]; then
    echo "ERROR: HF_HOME and HF_HUB_CACHE must be absolute paths" >&2
    exit 2
fi

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$OC_RUN_ROOT/checkpoints/oc_completion" \
    "$OC_RUN_ROOT/results/matched_completion"
HF_HOME="$(realpath -e -- "$HF_HOME")"
HF_HUB_CACHE="$(realpath -e -- "$HF_HUB_CACHE")"
export HF_HOME HF_HUB_CACHE
for cache_path in "$HF_HOME" "$HF_HUB_CACHE"; do
    case "$cache_path/" in
        "$HOME_ROOT/"*)
            echo "ERROR: Hugging Face cache must use project storage: $cache_path" >&2
            exit 2
            ;;
    esac
done

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required full-run input is missing: $1" >&2
        exit 2
    fi
}

for data_task in ocdet ocnoisy; do
    for split in train val test; do
        require_file "$DATA_DIR/simulation/oc_completion/X_${split}_${data_task}.csv"
        require_file "$DATA_DIR/simulation/oc_completion/y_${split}_${data_task}.csv"
    done
done
for family in one_hole two_hole; do
    for background in clean heldout; do
        for split in val test; do
            require_file "$DATA_DIR/simulation/oc_completion/pairs/pairs_${family}_${background}_${split}.csv"
        done
    done
done

TASK_DIR="oc_deterministic"
if [[ "$TASK" == "ocnoisy" ]]; then
    TASK_DIR="oc_noisy"
fi
RUN_DIR="$OC_RUN_ROOT/checkpoints/oc_completion/$TASK_DIR/$ARM/seed_$SEED"
DONE_MARKER="$RUN_DIR/done.json"
BEST_CHECKPOINT="$RUN_DIR/best.pt"
mkdir -p "$RUN_DIR"

# Prevent accidental duplicate submissions for the same run.
exec 9>"$RUN_DIR/.train.lock"
if ! flock -n 9; then
    echo "ERROR: another job already owns $ARM/$TASK/seed_$SEED" >&2
    exit 75
fi

OC_MAX_REQUEUES="${OC_MAX_REQUEUES:-3}"
if [[ ! "$OC_MAX_REQUEUES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: OC_MAX_REQUEUES must be a non-negative integer" >&2
    exit 2
fi
OC_PRUNE_COMPLETED="${OC_PRUNE_COMPLETED:-1}"
if [[ "$OC_PRUNE_COMPLETED" != 0 && "$OC_PRUNE_COMPLETED" != 1 ]]; then
    echo "ERROR: OC_PRUNE_COMPLETED must be 0 or 1" >&2
    exit 2
fi
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
ACTIVE_PID=""

handle_time_limit() {
    trap - USR1
    echo "=== WALLTIME SIGNAL ===" >&2
    echo "Completed epochs are safe in last.pt; the active epoch will restart." >&2
    if (( RESTART_COUNT >= OC_MAX_REQUEUES )); then
        echo "ERROR: reached OC_MAX_REQUEUES=$OC_MAX_REQUEUES; resubmit after review." >&2
        if [[ -n "$ACTIVE_PID" ]]; then
            kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        fi
        exit 99
    fi
    echo "Requeueing job $SLURM_JOB_ID (restart $((RESTART_COUNT + 1)))." >&2
    if [[ -n "$ACTIVE_PID" ]]; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
    fi
    if scontrol requeue "$SLURM_JOB_ID"; then
        exit 0
    fi
    echo "ERROR: automatic requeue failed; resubmit this array element manually." >&2
    exit 99
}
trap handle_time_limit USR1

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID  array: $SLURM_ARRAY_TASK_ID  restart: $RESTART_COUNT"
echo "Node: $(hostname)"
echo "Run: $ARM / $TASK / seed $SEED"
echo "Repository: $REPO_ROOT"
echo "Data: $DATA_DIR"
echo "Run outputs: $OC_RUN_ROOT"
echo "Hugging Face cache: $HF_HUB_CACHE"
echo "Checkpoint cleanup after success: $OC_PRUNE_COMPLETED"
module list 2>&1

if [[ "$ARM" == llama_* ]]; then
    TOKEN_PATH="${HF_TOKEN_PATH:-$HF_HOME/token}"
    LLAMA_CACHE="$HF_HUB_CACHE/models--meta-llama--Llama-3.2-1B"
    if [[ -z "${HF_TOKEN:-}" && ! -s "$TOKEN_PATH" && ! -d "$LLAMA_CACHE" ]]; then
        echo "ERROR: Llama is neither cached nor authenticated." >&2
        echo "Accept model access and export HF_TOKEN before submission." >&2
        exit 2
    fi
fi

python3 -u - <<'PY'
import accelerate
import peft
import torch
import transformers

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see the allocated H200")
gpu = torch.cuda.get_device_properties(0)
print(
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"peft={peft.__version__} accelerate={accelerate.__version__} "
    f"gpu={gpu.name} capability={gpu.major}.{gpu.minor}"
)
PY

cd "$REPO_ROOT/repro"

if [[ -f "$DONE_MARKER" && -f "$BEST_CHECKPOINT" ]]; then
    echo "Training already complete; keeping the existing best checkpoint."
else
    python3 -u -m src.oc_completion.train_hf \
        --arm "$ARM" \
        --task "$TASK" \
        --seed "$SEED" \
        --resume \
        --max_epochs "${OC_MAX_EPOCHS:-20}" \
        --patience "${OC_PATIENCE:-3}" \
        --max_length 64 \
        --threads "$SLURM_CPUS_PER_TASK" \
        --diag_pairs 200 \
        --eval_batch 64 &
    ACTIVE_PID=$!
    if wait "$ACTIVE_PID"; then
        TRAIN_STATUS=0
    else
        TRAIN_STATUS=$?
    fi
    ACTIVE_PID=""
    if (( TRAIN_STATUS != 0 )); then
        echo "ERROR: training exited with status $TRAIN_STATUS" >&2
        exit "$TRAIN_STATUS"
    fi
fi

require_file "$DONE_MARKER"
require_file "$BEST_CHECKPOINT"

# The protocol evaluates best.pt.  Once done.json exists, last.pt (optimizer
# resume state) and final.pt are redundant and would exceed the ec12 quota.
case "$OC_PRUNE_COMPLETED" in
    0) echo "Keeping last.pt and final.pt because OC_PRUNE_COMPLETED=0." ;;
    1)
        echo "Removing completed-run resume/final checkpoints; retaining best.pt."
        rm -f -- \
            "$RUN_DIR/last.pt" \
            "$RUN_DIR/final.pt" \
            "$RUN_DIR/.last.pt.tmp" \
            "$RUN_DIR/.best.pt.tmp" \
            "$RUN_DIR/.final.pt.tmp"
        ;;
esac

echo "=== TRAINING COMPLETE: $ARM / $TASK / seed $SEED ==="
