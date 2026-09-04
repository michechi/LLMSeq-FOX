#!/bin/bash -l
# General single-configuration Ordered Compliance hole-audit fine-tuning on FOX.
#
# The launcher accepts trainer-style command-line arguments as well as
# environment variables (command-line values win).  It uses adapter training
# by default and supports ordinary LoRA, 8-bit LoRA, and 4-bit QLoRA for
# Hugging Face encoder- and decoder-only text models.  HF_TOKEN is read only
# from the submitted environment; it is never placed on the command line,
# hashed into a derived tag, or printed.
#
# Required before submission:
#   mkdir -p logs/oc_hole
#   export OC_MODEL_NAME=<Hugging Face model ID or local model directory>
#   export HF_HOME=/cluster/work/projects/ec12/$USER/huggingface
#
# Optional storage overrides:
#   export REPO_ROOT=$PWD
#   export OC_DATA_ROOT=$PWD/data/simulation/oc_hole_audit
#   export OC_CHECKPOINT_ROOT=/cluster/work/projects/ec12/$USER/checkpoints/oc_hole_audit
#   export OC_HF_CACHE=$HF_HOME/hub       # HF_HUB_CACHE is also accepted
#   export OC_RUN_DIR=/absolute/path/to/one/run
#   export HF_TOKEN=...                   # only for gated/private models
#
# Qwen2.5-14B, BF16 LoRA on one H200 NVL:
#   export OC_MODEL_NAME=Qwen/Qwen2.5-14B
#   export OC_MODEL_KIND=decoder
#   export OC_MODEL_TAG=qwen25_14b_bf16_lora
#   export OC_PEFT=1 OC_QUANTIZATION=none
#   export OC_DTYPE=bf16 OC_COMPUTE_DTYPE=bf16
#   export OC_MICRO_BATCH=4 OC_GRADIENT_ACCUMULATION=4 OC_EVAL_BATCH=8
#   export OC_GRADIENT_CHECKPOINTING=1
#   sbatch scripts/slurm/FOX/oc_hole_train_hf_general.slurm
#
# Qwen2.5-14B, 4-bit NF4 QLoRA (the trainer selects NF4/double quantization):
#   export OC_MODEL_NAME=Qwen/Qwen2.5-14B
#   export OC_MODEL_KIND=decoder
#   export OC_MODEL_TAG=qwen25_14b_qlora4
#   export OC_PEFT=1 OC_QUANTIZATION=4bit
#   export OC_DTYPE=bf16 OC_COMPUTE_DTYPE=bf16
#   export OC_MICRO_BATCH=1 OC_GRADIENT_ACCUMULATION=16 OC_EVAL_BATCH=2
#   export OC_GRADIENT_CHECKPOINTING=1
#   sbatch scripts/slurm/FOX/oc_hole_train_hf_general.slurm

#SBATCH --job-name=oc_hole_general
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=nvidia_h200_nvl:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_hole/train_hf_general_%j.out

set -o errexit
set -o nounset
set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch scripts/slurm/FOX/oc_hole_train_hf_general.slurm [OPTIONS]
  bash scripts/slurm/FOX/oc_hole_train_hf_general.slurm --help

Command-line options override matching OC_* environment variables.

Model recipe:
  --model-name ID_OR_PATH             Required (OC_MODEL_NAME)
  --model-kind encoder|decoder        Default: decoder
  --model-tag TAG                     Safe explicit run/report tag
  --model-revision REVISION           Default: main
  --tokenizer-name ID_OR_PATH         Default: use the model tokenizer
  --tokenizer-revision REVISION       Default: use the model revision
  --peft | --no-peft                  Default: --peft
  --quantization none|8bit|4bit       Default: none
  --dtype auto|bf16|fp16|fp32         Default: bf16
  --compute-dtype auto|bf16|fp16|fp32 Default: bf16
  --gradient-checkpointing | --no-gradient-checkpointing
                                      Default: enabled
  --trust-remote-code | --no-trust-remote-code
  --local-files-only | --no-local-files-only

LoRA:
  --lora-r N                          Default: 8
  --lora-alpha N                      Default: 16
  --lora-dropout FLOAT                Default: 0.1
  --lora-targets NAMES                Comma-separated module names

Training:
  --pi 0.0|0.1|0.2|0.3                Default: 0.3
  --model-seed N                      Default: 9950
  --max-epochs N                      Default: 20
  --patience N                        Default: 3
  --micro-batch N                     Default: 1
  --gradient-accumulation N           Default: 16
  --eval-batch N                      Default: 2
  --workers N                         Default: allocated CPU count
  --threads N                         Default: allocated CPU count
  --audit-batch-size N                Default: 4096
  --max-train-rows N                  Default: 0 (unlimited)
  --resume | --no-resume              Default: --resume
  --smoke | --no-smoke                Default: --no-smoke

Storage:
  --repo-root PATH                    Default: REPO_ROOT/SLURM_SUBMIT_DIR/PWD
  --data-root PATH                    Ordered-data root
  --checkpoint-root PATH              Checkpoint root
  --run-dir PATH                      Exact run directory
  --hf-home PATH                      Required persistent Hugging Face home
  --hf-cache PATH                     Default: HF_HUB_CACHE or HF_HOME/hub

Other:
  -h, --help                          Show this help and exit

When --model-tag/OC_MODEL_TAG is omitted, the launcher derives a readable tag
with a short SHA-256 suffix over the non-secret model recipe.
HF_TOKEN remains environment-only for gated/private models.
EOF
}

need_value() {
    if (( $# < 2 )); then
        echo "ERROR: $1 requires a value" >&2
        exit 2
    fi
}

while (( $# > 0 )); do
    if [[ "$1" == --*=* ]]; then
        option_name="${1%%=*}"
        option_value="${1#*=}"
        shift
        set -- "$option_name" "$option_value" "$@"
    fi
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --model-name)
            need_value "$@"; OC_MODEL_NAME="$2"; shift 2 ;;
        --model-kind)
            need_value "$@"; OC_MODEL_KIND="$2"; shift 2 ;;
        --model-tag)
            need_value "$@"; OC_MODEL_TAG="$2"; shift 2 ;;
        --model-revision)
            need_value "$@"; OC_MODEL_REVISION="$2"; shift 2 ;;
        --tokenizer-name)
            need_value "$@"; OC_TOKENIZER_NAME="$2"; shift 2 ;;
        --tokenizer-revision)
            need_value "$@"; OC_TOKENIZER_REVISION="$2"; shift 2 ;;
        --peft)
            OC_PEFT=1; shift ;;
        --no-peft)
            OC_PEFT=0; shift ;;
        --quantization)
            need_value "$@"; OC_QUANTIZATION="$2"; shift 2 ;;
        --dtype)
            need_value "$@"; OC_DTYPE="$2"; shift 2 ;;
        --compute-dtype)
            need_value "$@"; OC_COMPUTE_DTYPE="$2"; shift 2 ;;
        --gradient-checkpointing)
            OC_GRADIENT_CHECKPOINTING=1; shift ;;
        --no-gradient-checkpointing)
            OC_GRADIENT_CHECKPOINTING=0; shift ;;
        --trust-remote-code)
            OC_TRUST_REMOTE_CODE=1; shift ;;
        --no-trust-remote-code)
            OC_TRUST_REMOTE_CODE=0; shift ;;
        --local-files-only)
            OC_LOCAL_FILES_ONLY=1; shift ;;
        --no-local-files-only)
            OC_LOCAL_FILES_ONLY=0; shift ;;
        --lora-r)
            need_value "$@"; OC_LORA_R="$2"; shift 2 ;;
        --lora-alpha)
            need_value "$@"; OC_LORA_ALPHA="$2"; shift 2 ;;
        --lora-dropout)
            need_value "$@"; OC_LORA_DROPOUT="$2"; shift 2 ;;
        --lora-targets)
            need_value "$@"; OC_LORA_TARGETS="$2"; shift 2 ;;
        --pi)
            need_value "$@"; OC_PI="$2"; shift 2 ;;
        --model-seed)
            need_value "$@"; OC_MODEL_SEED="$2"; shift 2 ;;
        --max-epochs)
            need_value "$@"; OC_MAX_EPOCHS="$2"; shift 2 ;;
        --patience)
            need_value "$@"; OC_PATIENCE="$2"; shift 2 ;;
        --micro-batch)
            need_value "$@"; OC_MICRO_BATCH="$2"; shift 2 ;;
        --gradient-accumulation)
            need_value "$@"; OC_GRADIENT_ACCUMULATION="$2"; shift 2 ;;
        --eval-batch)
            need_value "$@"; OC_EVAL_BATCH="$2"; shift 2 ;;
        --workers)
            need_value "$@"; OC_WORKERS="$2"; shift 2 ;;
        --threads)
            need_value "$@"; OC_THREADS="$2"; shift 2 ;;
        --audit-batch-size)
            need_value "$@"; OC_AUDIT_BATCH_SIZE="$2"; shift 2 ;;
        --max-train-rows)
            need_value "$@"; OC_MAX_TRAIN_ROWS="$2"; shift 2 ;;
        --resume)
            OC_RESUME=1; shift ;;
        --no-resume)
            OC_RESUME=0; shift ;;
        --smoke)
            OC_SMOKE=1; shift ;;
        --no-smoke)
            OC_SMOKE=0; shift ;;
        --repo-root)
            need_value "$@"; REPO_ROOT="$2"; shift 2 ;;
        --data-root)
            need_value "$@"; OC_DATA_ROOT="$2"; shift 2 ;;
        --checkpoint-root)
            need_value "$@"; OC_CHECKPOINT_ROOT="$2"; shift 2 ;;
        --run-dir)
            need_value "$@"; OC_RUN_DIR="$2"; shift 2 ;;
        --hf-home)
            need_value "$@"; HF_HOME="$2"; shift 2 ;;
        --hf-cache)
            need_value "$@"; OC_HF_CACHE="$2"; shift 2 ;;
        --)
            shift
            if (( $# > 0 )); then
                echo "ERROR: unexpected positional argument: $1" >&2
                exit 2
            fi
            ;;
        *)
            echo "ERROR: unknown option or positional argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
if [[ ! -f "$REPO_ROOT/repro/src/oc_completion/ordered_train_hf.py" ]]; then
    echo "ERROR: invalid REPO_ROOT or missing ordered HF trainer: $REPO_ROOT" >&2
    exit 2
fi

: "${OC_MODEL_NAME:?ERROR: export OC_MODEL_NAME to a Hugging Face model ID or local directory}"
: "${HF_HOME:?ERROR: set HF_HOME to a persistent project-storage directory}"

OC_MODEL_KIND="${OC_MODEL_KIND:-decoder}"
OC_MODEL_REVISION="${OC_MODEL_REVISION:-main}"
OC_TOKENIZER_NAME="${OC_TOKENIZER_NAME:-}"
OC_TOKENIZER_REVISION="${OC_TOKENIZER_REVISION:-$OC_MODEL_REVISION}"
OC_PEFT="${OC_PEFT:-1}"
OC_QUANTIZATION="${OC_QUANTIZATION:-none}"
OC_DTYPE="${OC_DTYPE:-bf16}"
OC_COMPUTE_DTYPE="${OC_COMPUTE_DTYPE:-bf16}"
OC_GRADIENT_CHECKPOINTING="${OC_GRADIENT_CHECKPOINTING:-1}"
OC_TRUST_REMOTE_CODE="${OC_TRUST_REMOTE_CODE:-0}"
OC_LOCAL_FILES_ONLY="${OC_LOCAL_FILES_ONLY:-0}"
OC_RESUME="${OC_RESUME:-1}"
OC_SMOKE="${OC_SMOKE:-0}"

OC_PI="${OC_PI:-0.3}"
OC_MODEL_SEED="${OC_MODEL_SEED:-9950}"
OC_MAX_EPOCHS="${OC_MAX_EPOCHS:-20}"
OC_PATIENCE="${OC_PATIENCE:-3}"
OC_MICRO_BATCH="${OC_MICRO_BATCH:-1}"
OC_GRADIENT_ACCUMULATION="${OC_GRADIENT_ACCUMULATION:-16}"
OC_EVAL_BATCH="${OC_EVAL_BATCH:-2}"
OC_WORKERS="${OC_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
OC_THREADS="${OC_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
OC_AUDIT_BATCH_SIZE="${OC_AUDIT_BATCH_SIZE:-4096}"
OC_MAX_TRAIN_ROWS="${OC_MAX_TRAIN_ROWS:-0}"

OC_LORA_R="${OC_LORA_R:-8}"
OC_LORA_ALPHA="${OC_LORA_ALPHA:-16}"
OC_LORA_DROPOUT="${OC_LORA_DROPOUT:-0.1}"
OC_LORA_TARGETS="${OC_LORA_TARGETS:-}"

case "$OC_MODEL_KIND" in
    encoder|decoder) ;;
    *)
        echo "ERROR: OC_MODEL_KIND must be encoder or decoder: $OC_MODEL_KIND" >&2
        exit 2
        ;;
esac
case "$OC_QUANTIZATION" in
    none|8bit|4bit) ;;
    *)
        echo "ERROR: OC_QUANTIZATION must be none, 8bit, or 4bit: $OC_QUANTIZATION" >&2
        exit 2
        ;;
esac
for dtype_name in OC_DTYPE OC_COMPUTE_DTYPE; do
    case "${!dtype_name}" in
        auto|bf16|fp16|fp32) ;;
        *)
            echo "ERROR: $dtype_name must be auto, bf16, fp16, or fp32: ${!dtype_name}" >&2
            exit 2
            ;;
    esac
done
for binary_name in \
        OC_PEFT OC_GRADIENT_CHECKPOINTING OC_TRUST_REMOTE_CODE \
        OC_LOCAL_FILES_ONLY OC_RESUME OC_SMOKE; do
    binary_value="${!binary_name}"
    if [[ "$binary_value" != 0 && "$binary_value" != 1 ]]; then
        echo "ERROR: $binary_name must be 0 or 1: $binary_value" >&2
        exit 2
    fi
done
if [[ "$OC_QUANTIZATION" != none && "$OC_PEFT" != 1 ]]; then
    echo "ERROR: 8-bit/4-bit training requires OC_PEFT=1" >&2
    exit 2
fi

if [[ "$OC_MODEL_NAME" == -* || "$OC_MODEL_NAME" =~ [[:space:]] ]]; then
    echo "ERROR: OC_MODEL_NAME cannot start with '-' or contain whitespace" >&2
    exit 2
fi
if [[ -z "$OC_MODEL_REVISION" || "$OC_MODEL_REVISION" == -* || \
      "$OC_MODEL_REVISION" =~ [[:space:]] ]]; then
    echo "ERROR: OC_MODEL_REVISION cannot be empty, start with '-', or contain whitespace" >&2
    exit 2
fi
if [[ -n "$OC_TOKENIZER_NAME" && \
      ( "$OC_TOKENIZER_NAME" == -* || "$OC_TOKENIZER_NAME" =~ [[:space:]] ) ]]; then
    echo "ERROR: OC_TOKENIZER_NAME cannot start with '-' or contain whitespace" >&2
    exit 2
fi
if [[ -z "$OC_TOKENIZER_REVISION" || "$OC_TOKENIZER_REVISION" == -* || \
      "$OC_TOKENIZER_REVISION" =~ [[:space:]] ]]; then
    echo "ERROR: OC_TOKENIZER_REVISION cannot be empty, start with '-', or contain whitespace" >&2
    exit 2
fi
if [[ -n "$OC_LORA_TARGETS" && \
      ! "$OC_LORA_TARGETS" =~ ^[A-Za-z0-9_.]+(,[A-Za-z0-9_.]+)*$ ]]; then
    echo "ERROR: OC_LORA_TARGETS must be a comma-separated module-name list" >&2
    exit 2
fi

case "$OC_PI" in
    0.0) PI_TAG=0p0 ;;
    0.1) PI_TAG=0p1 ;;
    0.2) PI_TAG=0p2 ;;
    0.3) PI_TAG=0p3 ;;
    *)
        echo "ERROR: OC_PI must be one of 0.0, 0.1, 0.2, or 0.3: $OC_PI" >&2
        exit 2
        ;;
esac
for integer_name in \
        OC_MODEL_SEED OC_MAX_EPOCHS OC_PATIENCE OC_MICRO_BATCH \
        OC_GRADIENT_ACCUMULATION OC_EVAL_BATCH OC_WORKERS OC_THREADS \
        OC_AUDIT_BATCH_SIZE OC_MAX_TRAIN_ROWS OC_LORA_R OC_LORA_ALPHA; do
    integer_value="${!integer_name}"
    if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $integer_name must be a non-negative integer: $integer_value" >&2
        exit 2
    fi
done
for positive_name in \
        OC_MAX_EPOCHS OC_PATIENCE OC_MICRO_BATCH OC_GRADIENT_ACCUMULATION \
        OC_EVAL_BATCH OC_THREADS OC_AUDIT_BATCH_SIZE OC_LORA_R OC_LORA_ALPHA; do
    if (( ${!positive_name} < 1 )); then
        echo "ERROR: $positive_name must be positive: ${!positive_name}" >&2
        exit 2
    fi
done

python3 - "$OC_LORA_DROPOUT" <<'PY'
import math
import sys

try:
    dropout = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"ERROR: OC_LORA_DROPOUT must be numeric: {sys.argv[1]}") from exc
if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
    raise SystemExit("ERROR: OC_LORA_DROPOUT must be finite and in [0, 1)")
PY

PEFT_TAG=full
if [[ "$OC_PEFT" == 1 ]]; then
    PEFT_TAG=lora
fi
PRECISION_TAG="$OC_QUANTIZATION"
if [[ "$OC_QUANTIZATION" == none ]]; then
    PRECISION_TAG="$OC_DTYPE"
else
    PRECISION_TAG="${OC_QUANTIZATION}_${OC_COMPUTE_DTYPE}compute"
fi
MODEL_SLUG="${OC_MODEL_NAME//\//_}"
MODEL_SLUG="${MODEL_SLUG//[!A-Za-z0-9._-]/_}"
if [[ ! "$MODEL_SLUG" =~ ^[A-Za-z0-9] ]]; then
    MODEL_SLUG="model_$MODEL_SLUG"
fi
MODEL_SLUG="${MODEL_SLUG:0:56}"
RECIPE_HASH="$(
    python3 - \
        "model_name=$OC_MODEL_NAME" \
        "model_kind=$OC_MODEL_KIND" \
        "model_revision=$OC_MODEL_REVISION" \
        "tokenizer_name=$OC_TOKENIZER_NAME" \
        "tokenizer_revision=$OC_TOKENIZER_REVISION" \
        "peft=$OC_PEFT" \
        "quantization=$OC_QUANTIZATION" \
        "dtype=$OC_DTYPE" \
        "compute_dtype=$OC_COMPUTE_DTYPE" \
        "gradient_checkpointing=$OC_GRADIENT_CHECKPOINTING" \
        "trust_remote_code=$OC_TRUST_REMOTE_CODE" \
        "local_files_only=$OC_LOCAL_FILES_ONLY" \
        "lora_r=$OC_LORA_R" \
        "lora_alpha=$OC_LORA_ALPHA" \
        "lora_dropout=$OC_LORA_DROPOUT" \
        "lora_targets=$OC_LORA_TARGETS" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
for field in sys.argv[1:]:
    digest.update(field.encode("utf-8"))
    digest.update(b"\0")
print(digest.hexdigest()[:12])
PY
)"
OC_MODEL_TAG="${OC_MODEL_TAG:-${MODEL_SLUG}_${OC_MODEL_KIND}_${PEFT_TAG}_${PRECISION_TAG}_${RECIPE_HASH}}"
if [[ ! "$OC_MODEL_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || \
      ${#OC_MODEL_TAG} -gt 100 || "$OC_MODEL_TAG" == "." || \
      "$OC_MODEL_TAG" == ".." ]]; then
    echo "ERROR: OC_MODEL_TAG must be a safe path segment using [A-Za-z0-9._-]" >&2
    echo "       (starts alphanumeric, maximum 100 characters): $OC_MODEL_TAG" >&2
    exit 2
fi

OC_DATA_ROOT="${OC_DATA_ROOT:-$REPO_ROOT/data/simulation/oc_hole_audit}"
OC_CHECKPOINT_ROOT="${OC_CHECKPOINT_ROOT:-/cluster/work/projects/ec12/$USER/checkpoints/oc_hole_audit}"
OC_HF_CACHE="${OC_HF_CACHE:-${HF_HUB_CACHE:-$HF_HOME/hub}}"
for path_name in OC_DATA_ROOT OC_CHECKPOINT_ROOT HF_HOME OC_HF_CACHE; do
    path_value="${!path_name}"
    if [[ "$path_value" != /* ]]; then
        echo "ERROR: $path_name must be an absolute path: $path_value" >&2
        exit 2
    fi
done
OC_DATA_ROOT="$(realpath -e -- "$OC_DATA_ROOT")"
mkdir -p "$OC_CHECKPOINT_ROOT" "$HF_HOME" "$OC_HF_CACHE"
OC_CHECKPOINT_ROOT="$(realpath -e -- "$OC_CHECKPOINT_ROOT")"
HF_HOME="$(realpath -e -- "$HF_HOME")"
OC_HF_CACHE="$(realpath -e -- "$OC_HF_CACHE")"
HF_HUB_CACHE="$OC_HF_CACHE"

RUN_VARIANT_TAG="seed_$OC_MODEL_SEED"
if [[ "$OC_SMOKE" == 1 ]]; then
    RUN_VARIANT_TAG="${RUN_VARIANT_TAG}_smoke"
fi
OC_RUN_DIR="${OC_RUN_DIR:-$OC_CHECKPOINT_ROOT/$OC_MODEL_TAG/pi_$PI_TAG/$RUN_VARIANT_TAG}"
if [[ "$OC_RUN_DIR" != /* || "$OC_RUN_DIR" == / ]]; then
    echo "ERROR: OC_RUN_DIR must be an absolute, non-root path: $OC_RUN_DIR" >&2
    exit 2
fi
OC_RUN_DIR="$(realpath -m -- "$OC_RUN_DIR")"
mkdir -p "$OC_RUN_DIR"
OC_RUN_DIR="$(realpath -e -- "$OC_RUN_DIR")"

for required in \
        "$OC_DATA_ROOT/splits/train.parquet" \
        "$OC_DATA_ROOT/splits/val.parquet" \
        "$OC_DATA_ROOT/splits/test.parquet" \
        "$OC_DATA_ROOT/noise/train.parquet" \
        "$OC_DATA_ROOT/noise/val.parquet" \
        "$OC_DATA_ROOT/noise/test.parquet"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: missing ordered-data input: $required" >&2
        exit 2
    fi
done

export REPO_ROOT HF_HOME HF_HUB_CACHE OC_CHECKPOINT_ROOT
export LLMSEQ_ROOT="$REPO_ROOT"
export PYTHONHASHSEED="$OC_MODEL_SEED"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$OC_THREADS"
export OPENBLAS_NUM_THREADS="$OC_THREADS"
export MKL_NUM_THREADS="$OC_THREADS"
export NUMEXPR_NUM_THREADS="$OC_THREADS"

python3 - "$OC_PEFT" "$OC_QUANTIZATION" <<'PY'
import importlib
import sys

import accelerate
import torch
import transformers

peft_enabled = sys.argv[1] == "1"
quantization = sys.argv[2]
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see the allocated H200 GPU")

versions = [
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"accelerate={accelerate.__version__}",
]
if peft_enabled:
    peft = importlib.import_module("peft")
    versions.append(f"peft={peft.__version__}")
if quantization != "none":
    try:
        bitsandbytes = importlib.import_module("bitsandbytes")
        from transformers.utils import is_bitsandbytes_available
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "ERROR: quantized training requires a CUDA-compatible bitsandbytes "
            "installation in the active FOX Python environment"
        ) from exc
    if not is_bitsandbytes_available():
        raise SystemExit(
            "ERROR: Transformers reports that bitsandbytes is unavailable; "
            "check its CUDA/Python compatibility"
        )
    versions.append(f"bitsandbytes={bitsandbytes.__version__}")
versions.append(f"gpu={torch.cuda.get_device_name(0)}")
print(" ".join(versions))
PY

if [[ "$OC_LOCAL_FILES_ONLY" == 0 && -z "${HF_TOKEN:-}" ]]; then
    echo "NOTE: HF_TOKEN is unset; public/cached models work, but gated models may not." >&2
fi

echo "=== FOX GENERAL OC-HOLE TRAINING CONFIGURATION ==="
echo "Job / node: ${SLURM_JOB_ID:-unknown} / $(hostname)"
echo "Repository: $REPO_ROOT"
echo "Data: $OC_DATA_ROOT"
echo "Checkpoint root: $OC_CHECKPOINT_ROOT"
echo "Run directory: $OC_RUN_DIR"
echo "HF cache: $HF_HUB_CACHE"
echo "Model: $OC_MODEL_NAME (revision $OC_MODEL_REVISION; kind $OC_MODEL_KIND)"
echo "Tokenizer: ${OC_TOKENIZER_NAME:-model default} (revision $OC_TOKENIZER_REVISION)"
echo "Model tag: $OC_MODEL_TAG"
echo "PEFT / quantization: $OC_PEFT / $OC_QUANTIZATION"
echo "Model dtype / quantized compute dtype: $OC_DTYPE / $OC_COMPUTE_DTYPE"
echo "Noise pi / model seed: $OC_PI / $OC_MODEL_SEED"
echo "Micro-batch / accumulation / effective batch: $OC_MICRO_BATCH / $OC_GRADIENT_ACCUMULATION / $((OC_MICRO_BATCH * OC_GRADIENT_ACCUMULATION))"
echo "Evaluation batch: $OC_EVAL_BATCH"
echo "Audit batch / maximum training rows: $OC_AUDIT_BATCH_SIZE / $OC_MAX_TRAIN_ROWS"
echo "Gradient checkpointing: $OC_GRADIENT_CHECKPOINTING"
echo "Trust remote code / local files only: $OC_TRUST_REMOTE_CODE / $OC_LOCAL_FILES_ONLY"
echo "Resume / smoke: $OC_RESUME / $OC_SMOKE"

TRAIN_ARGS=(
    --model-name "$OC_MODEL_NAME"
    --model-kind "$OC_MODEL_KIND"
    --model-tag "$OC_MODEL_TAG"
    --model-revision "$OC_MODEL_REVISION"
    --tokenizer-revision "$OC_TOKENIZER_REVISION"
    --quantization "$OC_QUANTIZATION"
    --dtype "$OC_DTYPE"
    --compute-dtype "$OC_COMPUTE_DTYPE"
    --pi "$OC_PI"
    --model-seed "$OC_MODEL_SEED"
    --data-root "$OC_DATA_ROOT"
    --checkpoint-root "$OC_CHECKPOINT_ROOT"
    --run-dir "$OC_RUN_DIR"
    --hf-cache "$OC_HF_CACHE"
    --max-epochs "$OC_MAX_EPOCHS"
    --patience "$OC_PATIENCE"
    --micro-batch "$OC_MICRO_BATCH"
    --gradient-accumulation "$OC_GRADIENT_ACCUMULATION"
    --eval-batch "$OC_EVAL_BATCH"
    --workers "$OC_WORKERS"
    --threads "$OC_THREADS"
    --audit-batch-size "$OC_AUDIT_BATCH_SIZE"
    --max-train-rows "$OC_MAX_TRAIN_ROWS"
)
[[ -z "$OC_TOKENIZER_NAME" ]] || TRAIN_ARGS+=(--tokenizer-name "$OC_TOKENIZER_NAME")
if [[ "$OC_PEFT" == 1 ]]; then
    TRAIN_ARGS+=(
        --peft
        --lora-r "$OC_LORA_R"
        --lora-alpha "$OC_LORA_ALPHA"
        --lora-dropout "$OC_LORA_DROPOUT"
    )
    [[ -z "$OC_LORA_TARGETS" ]] || TRAIN_ARGS+=(--lora-targets "$OC_LORA_TARGETS")
else
    TRAIN_ARGS+=(--no-peft)
fi
if [[ "$OC_GRADIENT_CHECKPOINTING" == 1 ]]; then
    TRAIN_ARGS+=(--gradient-checkpointing)
else
    TRAIN_ARGS+=(--no-gradient-checkpointing)
fi
if [[ "$OC_TRUST_REMOTE_CODE" == 1 ]]; then
    TRAIN_ARGS+=(--trust-remote-code)
fi
if [[ "$OC_LOCAL_FILES_ONLY" == 1 ]]; then
    TRAIN_ARGS+=(--local-files-only)
fi
if [[ "$OC_RESUME" == 1 ]]; then
    TRAIN_ARGS+=(--resume)
fi
if [[ "$OC_SMOKE" == 1 ]]; then
    TRAIN_ARGS+=(--smoke)
fi

cd "$REPO_ROOT/repro"
python3 -u -m src.oc_completion.ordered_train_hf "${TRAIN_ARGS[@]}"

python3 - "$OC_RUN_DIR/done.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    done = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"ERROR: trainer did not write a valid {path}") from exc
if done.get("status") != "complete":
    raise SystemExit(f"ERROR: run is not marked complete in {path}")
print(f"complete: {path.parent}")
PY
