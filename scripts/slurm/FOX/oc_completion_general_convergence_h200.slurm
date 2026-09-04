#!/bin/bash -l
# Generic, one-seed, checkpoint-free OC convergence experiment on one FOX H200.
#
# The same launcher supports OC-Det and OC-Noisy, encoder classifiers and
# causal-LM classifiers, and LoRA or full fine-tuning.  It keeps the best
# trainable state only in RAM; the Python process must finish its final
# evaluation before the allocation ends.
#
# Required environment variables:
#   OC_TASK=ocdet|ocnoisy
#   OC_MODEL_NAME=<Hugging Face model ID>
#   OC_RUN_ROOT=<fresh absolute directory under /fp/projects01>
#
# Typical LoRA submission from the LITE repository root:
#   mkdir -p logs
#   export REPO_ROOT="$PWD"
#   export HF_HOME=/fp/projects01/ec12/ec-michechi/huggingface
#   export HF_HUB_CACHE="$HF_HOME/hub"
#   export OC_TASK=ocdet
#   export OC_MODEL_NAME=meta-llama/Llama-3.2-1B
#   export OC_MODEL_KIND=causal
#   export OC_LORA=1
#   export OC_RUN_ROOT=/fp/projects01/ec12/ec-michechi/oc_llama32_1b_det_s9550
#   sbatch scripts/slurm/FOX/oc_completion_general_convergence_h200.slurm

#SBATCH --job-name=oc_general_conv
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
# Four hours are reserved for graceful final evaluation of the best in-memory
# state.  The trap below forwards this warning from the batch shell to Python.
#SBATCH --signal=B:USR1@14400
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_general_convergence_%j.out

set -o errexit
set -o nounset
set -o pipefail

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
if [[ ! -f "$REPO_ROOT/repro/src/oc_completion/train_general_convergence.py" ]]; then
    echo "ERROR: generic convergence trainer is missing from REPO_ROOT=$REPO_ROOT" >&2
    exit 2
fi

: "${OC_TASK:?ERROR: export OC_TASK=ocdet or OC_TASK=ocnoisy}"
: "${OC_MODEL_NAME:?ERROR: export OC_MODEL_NAME to a Hugging Face model ID}"
: "${OC_RUN_ROOT:?ERROR: export OC_RUN_ROOT to a fresh project-storage directory}"

OC_MODEL_KIND="${OC_MODEL_KIND:-causal}"
OC_LORA="${OC_LORA:-1}"
OC_MAX_EPOCHS="${OC_MAX_EPOCHS:-80}"
OC_PATIENCE="${OC_PATIENCE:-3}"
OC_BATCH_SIZE="${OC_BATCH_SIZE:-8}"
OC_GRAD_ACCUM="${OC_GRAD_ACCUM:-2}"
OC_LR="${OC_LR:-2e-5}"
OC_WARMUP_RATIO="${OC_WARMUP_RATIO:-0.06}"
OC_SEED="${OC_SEED:-9550}"
OC_MAX_LENGTH="${OC_MAX_LENGTH:-64}"
OC_EVAL_BATCH="${OC_EVAL_BATCH:-64}"
OC_GRADIENT_CHECKPOINTING="${OC_GRADIENT_CHECKPOINTING:-0}"
OC_TRUST_REMOTE_CODE="${OC_TRUST_REMOTE_CODE:-0}"
OC_WORKERS="${OC_WORKERS:-4}"
OC_DIAG_PAIRS="${OC_DIAG_PAIRS:-2000}"
OC_MODEL_REVISION="${OC_MODEL_REVISION:-main}"
OC_LORA_R="${OC_LORA_R:-}"
OC_LORA_ALPHA="${OC_LORA_ALPHA:-}"
OC_LORA_DROPOUT="${OC_LORA_DROPOUT:-}"
OC_LORA_TARGETS="${OC_LORA_TARGETS:-}"

case "$OC_TASK" in
    ocdet|ocnoisy) ;;
    *)
        echo "ERROR: OC_TASK must be ocdet or ocnoisy: $OC_TASK" >&2
        exit 2
        ;;
esac
case "$OC_MODEL_KIND" in
    bert|causal) ;;
    *)
        echo "ERROR: OC_MODEL_KIND must be bert or causal: $OC_MODEL_KIND" >&2
        exit 2
        ;;
esac
for binary_name in OC_LORA OC_GRADIENT_CHECKPOINTING OC_TRUST_REMOTE_CODE; do
    binary_value="${!binary_name}"
    if [[ "$binary_value" != 0 && "$binary_value" != 1 ]]; then
        echo "ERROR: $binary_name must be 0 or 1: $binary_value" >&2
        exit 2
    fi
done

for integer_name in \
        OC_SEED OC_MAX_EPOCHS OC_PATIENCE OC_BATCH_SIZE OC_GRAD_ACCUM \
        OC_MAX_LENGTH OC_EVAL_BATCH OC_WORKERS OC_DIAG_PAIRS; do
    integer_value="${!integer_name}"
    if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $integer_name must be a non-negative integer: $integer_value" >&2
        exit 2
    fi
done
for positive_name in \
        OC_MAX_EPOCHS OC_PATIENCE OC_BATCH_SIZE OC_GRAD_ACCUM \
        OC_MAX_LENGTH OC_EVAL_BATCH OC_DIAG_PAIRS; do
    if (( ${!positive_name} < 1 )); then
        echo "ERROR: $positive_name must be positive: ${!positive_name}" >&2
        exit 2
    fi
done

if [[ "$OC_MODEL_NAME" == -* || "$OC_MODEL_NAME" =~ [[:space:]] ]]; then
    echo "ERROR: OC_MODEL_NAME cannot start with '-' or contain whitespace" >&2
    exit 2
fi
if [[ -z "$OC_MODEL_REVISION" || "$OC_MODEL_REVISION" == -* || \
      "$OC_MODEL_REVISION" =~ [[:space:]] ]]; then
    echo "ERROR: OC_MODEL_REVISION cannot be empty, start with '-', or contain whitespace" >&2
    exit 2
fi
if [[ -n "$OC_LORA_TARGETS" && \
      ! "$OC_LORA_TARGETS" =~ ^[A-Za-z0-9_.]+(,[A-Za-z0-9_.]+)*$ ]]; then
    echo "ERROR: OC_LORA_TARGETS must be a comma-separated module-name list" >&2
    exit 2
fi
if [[ "$OC_LORA" == 0 && \
      ( -n "$OC_LORA_R" || -n "$OC_LORA_ALPHA" || \
        -n "$OC_LORA_DROPOUT" || -n "$OC_LORA_TARGETS" ) ]]; then
    echo "ERROR: OC_LORA_* overrides were supplied while OC_LORA=0" >&2
    exit 2
fi

python3 - "$OC_LR" "$OC_WARMUP_RATIO" \
        "$OC_LORA_R" "$OC_LORA_ALPHA" "$OC_LORA_DROPOUT" <<'PY'
import math
import sys

names = ("OC_LR", "OC_WARMUP_RATIO", "OC_LORA_R", "OC_LORA_ALPHA", "OC_LORA_DROPOUT")
values = dict(zip(names, sys.argv[1:]))

def finite_float(name):
    try:
        value = float(values[name])
    except ValueError as exc:
        raise SystemExit(f"ERROR: {name} must be numeric: {values[name]}") from exc
    if not math.isfinite(value):
        raise SystemExit(f"ERROR: {name} must be finite: {values[name]}")
    return value

lr = finite_float("OC_LR")
warmup = finite_float("OC_WARMUP_RATIO")
if lr <= 0:
    raise SystemExit("ERROR: OC_LR must be positive")
if not 0.0 <= warmup < 1.0:
    raise SystemExit("ERROR: OC_WARMUP_RATIO must be in [0, 1)")

if values["OC_LORA_R"]:
    try:
        rank = int(values["OC_LORA_R"])
    except ValueError as exc:
        raise SystemExit("ERROR: OC_LORA_R must be an integer") from exc
    if rank < 1:
        raise SystemExit("ERROR: OC_LORA_R must be positive")
if values["OC_LORA_ALPHA"]:
    try:
        alpha = int(values["OC_LORA_ALPHA"])
    except ValueError as exc:
        raise SystemExit("ERROR: OC_LORA_ALPHA must be an integer") from exc
    if alpha < 1:
        raise SystemExit("ERROR: OC_LORA_ALPHA must be positive")
if values["OC_LORA_DROPOUT"]:
    dropout = finite_float("OC_LORA_DROPOUT")
    if not 0.0 <= dropout < 1.0:
        raise SystemExit("ERROR: OC_LORA_DROPOUT must be in [0, 1)")
PY

if [[ "$OC_RUN_ROOT" != /* ]]; then
    echo "ERROR: OC_RUN_ROOT must be absolute: $OC_RUN_ROOT" >&2
    exit 2
fi
OC_RUN_ROOT="$(realpath -m -- "$OC_RUN_ROOT")"
case "$OC_RUN_ROOT/" in
    /fp/projects01/*) ;;
    *)
        echo "ERROR: OC_RUN_ROOT must be under /fp/projects01: $OC_RUN_ROOT" >&2
        exit 2
        ;;
esac
if [[ "$OC_RUN_ROOT" == /fp/projects01 ]]; then
    echo "ERROR: OC_RUN_ROOT cannot be the project-filesystem root" >&2
    exit 2
fi
mkdir -p "$OC_RUN_ROOT"
OC_RUN_ROOT="$(realpath -e -- "$OC_RUN_ROOT")"
case "$OC_RUN_ROOT/" in
    /fp/projects01/*) ;;
    *)
        echo "ERROR: OC_RUN_ROOT must resolve under /fp/projects01: $OC_RUN_ROOT" >&2
        exit 2
        ;;
esac
if [[ "$OC_RUN_ROOT" == /fp/projects01 ]]; then
    echo "ERROR: OC_RUN_ROOT cannot be the project-filesystem root" >&2
    exit 2
fi
if find "$OC_RUN_ROOT" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: OC_RUN_ROOT must be fresh and empty so outputs cannot collide:" >&2
    echo "       $OC_RUN_ROOT" >&2
    exit 2
fi

HOME_ROOT="$(realpath -e -- "$HOME")"
case "$OC_RUN_ROOT/" in
    "$HOME_ROOT/"*)
        echo "ERROR: OC_RUN_ROOT must use project storage, not home" >&2
        exit 2
        ;;
esac

MODEL_SLUG="${OC_MODEL_NAME//\//_}"
MODEL_SLUG="${MODEL_SLUG//:/_}"
MODEL_SLUG="${MODEL_SLUG//+/_}"
MODEL_SLUG="${MODEL_SLUG:0:80}"
LORA_TAG=full
if [[ "$OC_LORA" == 1 ]]; then
    LORA_TAG=lora
fi
DEFAULT_RUN_NAME="${MODEL_SLUG}_${OC_MODEL_KIND}_${LORA_TAG}_${OC_TASK}_s${OC_SEED}_e${OC_MAX_EPOCHS}_p${OC_PATIENCE}_bs${OC_BATCH_SIZE}x${OC_GRAD_ACCUM}_lr${OC_LR}_wu${OC_WARMUP_RATIO}_j${SLURM_JOB_ID}_nockpt"
OC_RUN_NAME="${OC_RUN_NAME:-$DEFAULT_RUN_NAME}"
if [[ ! "$OC_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ || ${#OC_RUN_NAME} -gt 240 ]]; then
    echo "ERROR: OC_RUN_NAME must use [A-Za-z0-9._-] and be at most 240 characters" >&2
    exit 2
fi
OUTPUT_DIR="$OC_RUN_ROOT/results/matched_completion/convergence/$OC_RUN_NAME"
mkdir -p "$OUTPUT_DIR"

export DATA_DIR="$REPO_ROOT/data"
export LLMSEQ_ROOT="$OC_RUN_ROOT"
export HF_HOME="${HF_HOME:-$OC_RUN_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [[ "$HF_HOME" != /* || "$HF_HUB_CACHE" != /* ]]; then
    echo "ERROR: HF_HOME and HF_HUB_CACHE must be absolute" >&2
    exit 2
fi
HF_HOME="$(realpath -m -- "$HF_HOME")"
HF_HUB_CACHE="$(realpath -m -- "$HF_HUB_CACHE")"
for cache_path in "$HF_HOME" "$HF_HUB_CACHE"; do
    case "$cache_path/" in
        /fp/projects01/*) ;;
        *)
            echo "ERROR: Hugging Face cache must be under /fp/projects01: $cache_path" >&2
            exit 2
            ;;
    esac
done
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
HF_HOME="$(realpath -e -- "$HF_HOME")"
HF_HUB_CACHE="$(realpath -e -- "$HF_HUB_CACHE")"
export HF_HOME HF_HUB_CACHE
for cache_path in "$HF_HOME" "$HF_HUB_CACHE"; do
    case "$cache_path/" in
        /fp/projects01/*) ;;
        *)
            echo "ERROR: Hugging Face cache must resolve under /fp/projects01: $cache_path" >&2
            exit 2
            ;;
    esac
    case "$cache_path/" in
        "$HOME_ROOT/"*)
            echo "ERROR: Hugging Face cache must not use home: $cache_path" >&2
            exit 2
            ;;
    esac
done

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required input is missing: $1" >&2
        exit 2
    fi
}

for split in train val test; do
    require_file "$DATA_DIR/simulation/oc_completion/X_${split}_${OC_TASK}.csv"
    require_file "$DATA_DIR/simulation/oc_completion/y_${split}_${OC_TASK}.csv"
done
for family in one_hole two_hole; do
    for background in clean heldout; do
        for split in val test; do
            require_file \
                "$DATA_DIR/simulation/oc_completion/pairs/pairs_${family}_${background}_${split}.csv"
        done
    done
done

python3 -u - <<'PY'
import peft
import torch
import transformers

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see the allocated H200")
print(
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"peft={peft.__version__} gpu={torch.cuda.get_device_name(0)}"
)
PY

echo "=== FOX GENERAL CHECKPOINT-FREE CONVERGENCE CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repository: $REPO_ROOT"
echo "Data: $DATA_DIR"
echo "Results root: $OC_RUN_ROOT"
echo "Run output: $OUTPUT_DIR"
echo "HF cache: $HF_HUB_CACHE"
echo "Task: $OC_TASK"
echo "Model: $OC_MODEL_NAME (revision $OC_MODEL_REVISION; kind $OC_MODEL_KIND)"
echo "LoRA: $OC_LORA"
echo "Seed / maximum epochs / patience: $OC_SEED / $OC_MAX_EPOCHS / $OC_PATIENCE"
echo "Micro-batch / accumulation / effective batch: $OC_BATCH_SIZE / $OC_GRAD_ACCUM / $((OC_BATCH_SIZE * OC_GRAD_ACCUM))"
echo "Learning rate / warmup ratio: $OC_LR / $OC_WARMUP_RATIO"
echo "Maximum length / evaluation batch: $OC_MAX_LENGTH / $OC_EVAL_BATCH"
echo "Gradient checkpointing: $OC_GRADIENT_CHECKPOINTING"
echo "Trust remote code: $OC_TRUST_REMOTE_CODE"
echo "Checkpoint policy: NONE"
echo "Requeue policy: disabled"
module list 2>&1

TRAIN_ARGS=(
    --task "$OC_TASK"
    --model_name "$OC_MODEL_NAME"
    --model_kind "$OC_MODEL_KIND"
    --model_revision "$OC_MODEL_REVISION"
    --seed "$OC_SEED"
    --max_epochs "$OC_MAX_EPOCHS"
    --patience "$OC_PATIENCE"
    --batch_size "$OC_BATCH_SIZE"
    --grad_accum "$OC_GRAD_ACCUM"
    --lr "$OC_LR"
    --warmup_ratio "$OC_WARMUP_RATIO"
    --max_length "$OC_MAX_LENGTH"
    --threads "$SLURM_CPUS_PER_TASK"
    --workers "$OC_WORKERS"
    --diag_pairs "$OC_DIAG_PAIRS"
    --eval_batch "$OC_EVAL_BATCH"
    --output_dir "$OUTPUT_DIR"
    --require_cuda
)
if [[ "$OC_LORA" == 1 ]]; then
    TRAIN_ARGS+=(--lora)
    [[ -z "$OC_LORA_R" ]] || TRAIN_ARGS+=(--lora_r "$OC_LORA_R")
    [[ -z "$OC_LORA_ALPHA" ]] || TRAIN_ARGS+=(--lora_alpha "$OC_LORA_ALPHA")
    [[ -z "$OC_LORA_DROPOUT" ]] || TRAIN_ARGS+=(--lora_dropout "$OC_LORA_DROPOUT")
    [[ -z "$OC_LORA_TARGETS" ]] || TRAIN_ARGS+=(--lora_targets "$OC_LORA_TARGETS")
else
    TRAIN_ARGS+=(--no-lora)
fi
if [[ "$OC_GRADIENT_CHECKPOINTING" == 1 ]]; then
    TRAIN_ARGS+=(--gradient_checkpointing)
else
    TRAIN_ARGS+=(--no-gradient_checkpointing)
fi
if [[ "$OC_TRUST_REMOTE_CODE" == 1 ]]; then
    TRAIN_ARGS+=(--trust_remote_code)
else
    TRAIN_ARGS+=(--no-trust_remote_code)
fi

ACTIVE_PID=""
forward_walltime_signal() {
    trap - USR1
    echo "=== FOX WALLTIME WARNING: requesting graceful in-memory finalization ===" >&2
    if [[ -n "$ACTIVE_PID" ]]; then
        kill -USR1 "$ACTIVE_PID" 2>/dev/null || true
    fi
}
trap forward_walltime_signal USR1

cd "$REPO_ROOT/repro"
python3 -u -m src.oc_completion.train_general_convergence "${TRAIN_ARGS[@]}" &
ACTIVE_PID=$!

while true; do
    if wait "$ACTIVE_PID"; then
        STATUS=0
        break
    else
        STATUS=$?
    fi
    # USR1 interrupts bash's wait while Python continues finalizing.  Keep the
    # allocation alive until the child exits or Slurm reaches the hard limit.
    if kill -0 "$ACTIVE_PID" 2>/dev/null; then
        continue
    fi
    break
done
ACTIVE_PID=""
if (( STATUS != 0 )); then
    echo "ERROR: convergence trainer exited with status $STATUS" >&2
    exit "$STATUS"
fi

require_file "$OUTPUT_DIR/summary.json"
python3 - "$OUTPUT_DIR/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("checkpoint_saved") is not False:
    raise SystemExit("ERROR: summary does not confirm checkpoint_saved=false")
print(
    "termination=" + str(summary.get("termination_reason")),
    "converged_by_patience=" + str(summary.get("converged_by_patience")),
    "epochs=" + str(summary.get("epochs_completed")),
    "best_epoch=" + str(summary.get("best_epoch")),
)
PY

if find "$OUTPUT_DIR" \
        \( -type d -name 'checkpoint-*' -o \
           -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o \
                       -name '*.safetensors' \) \) \
        -print -quit | grep -q .; then
    echo "ERROR: a forbidden model checkpoint was written under $OUTPUT_DIR" >&2
    exit 2
fi

echo "=== GENERAL OC CONVERGENCE JOB FINISHED ==="
