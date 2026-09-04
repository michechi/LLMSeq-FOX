#!/bin/bash -l
# One-model, one-task, one-seed checkpoint-free DL convergence run on FOX.
#
# Required environment variables:
#   OC_MODEL=LSTM|Transformer|RNNTransformer
#   OC_TASK=ocdet|ocnoisy
#   OC_RUN_ROOT=<fresh absolute directory under /fp/projects01>
#
# The learning rate is constant.  Leave OC_LR unset to use the model's
# established RECIPES value, or export a positive OC_LR to override it.
# Create the repository's logs directory before calling sbatch because Slurm
# opens the output file before this script starts.
#
# Example from the LITE repository root:
#   mkdir -p logs
#   export REPO_ROOT="$PWD"
#   export OC_MODEL=LSTM
#   export OC_TASK=ocdet
#   export OC_RUN_ROOT=/fp/projects01/ec12/ec-michechi/oc_lstm_det_s9550
#   sbatch scripts/slurm/FOX/oc_completion_dl_convergence_h200.slurm

#SBATCH --job-name=oc_dl_conv
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
# Four hours remain after USR1 for restoring the best in-memory state and
# evaluating the ordinary and eight matched-pair datasets.
#SBATCH --signal=B:USR1@14400
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_dl_convergence_%j.out

set -o errexit
set -o nounset
set -o pipefail

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
TRAINER="$REPO_ROOT/repro/src/oc_completion/train_dl_convergence.py"
if [[ ! -f "$TRAINER" ]]; then
    echo "ERROR: DL convergence trainer is missing: $TRAINER" >&2
    exit 2
fi

: "${OC_MODEL:?ERROR: export OC_MODEL=LSTM, Transformer, or RNNTransformer}"
: "${OC_TASK:?ERROR: export OC_TASK=ocdet or OC_TASK=ocnoisy}"
: "${OC_RUN_ROOT:?ERROR: export OC_RUN_ROOT to a fresh project directory}"

OC_SEED="${OC_SEED:-9550}"
OC_MAX_EPOCHS="${OC_MAX_EPOCHS:-100}"
OC_PATIENCE="${OC_PATIENCE:-3}"
OC_BATCH_SIZE="${OC_BATCH_SIZE:-64}"
OC_LR="${OC_LR:-}"
OC_THREADS="${OC_THREADS:-$SLURM_CPUS_PER_TASK}"
OC_DIAG_PAIRS="${OC_DIAG_PAIRS:-2000}"
OC_EVAL_BATCH="${OC_EVAL_BATCH:-1024}"

case "$OC_MODEL" in
    LSTM|Transformer|RNNTransformer) ;;
    *)
        echo "ERROR: OC_MODEL must be LSTM, Transformer, or RNNTransformer" >&2
        exit 2
        ;;
esac
case "$OC_TASK" in
    ocdet|ocnoisy) ;;
    *)
        echo "ERROR: OC_TASK must be ocdet or ocnoisy: $OC_TASK" >&2
        exit 2
        ;;
esac

for integer_name in \
        OC_SEED OC_MAX_EPOCHS OC_PATIENCE OC_BATCH_SIZE OC_THREADS \
        OC_DIAG_PAIRS OC_EVAL_BATCH; do
    integer_value="${!integer_name}"
    if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $integer_name must be a non-negative integer" >&2
        exit 2
    fi
done
for positive_name in \
        OC_MAX_EPOCHS OC_PATIENCE OC_BATCH_SIZE OC_THREADS OC_DIAG_PAIRS \
        OC_EVAL_BATCH; do
    if (( ${!positive_name} < 1 )); then
        echo "ERROR: $positive_name must be positive" >&2
        exit 2
    fi
done
if (( OC_THREADS > SLURM_CPUS_PER_TASK )); then
    echo "ERROR: OC_THREADS=$OC_THREADS exceeds allocated CPUs=$SLURM_CPUS_PER_TASK" >&2
    exit 2
fi

if [[ -n "$OC_LR" ]]; then
    python3 - "$OC_LR" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("ERROR: OC_LR must be numeric") from exc
if not math.isfinite(value) or value <= 0:
    raise SystemExit("ERROR: OC_LR must be finite and positive")
PY
fi

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
        echo "ERROR: OC_RUN_ROOT must resolve under /fp/projects01" >&2
        exit 2
        ;;
esac
if find "$OC_RUN_ROOT" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: OC_RUN_ROOT must be fresh and empty:" >&2
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

LR_TAG="${OC_LR:-recipe}"
DEFAULT_RUN_NAME="${OC_MODEL}_${OC_TASK}_s${OC_SEED}_e${OC_MAX_EPOCHS}_p${OC_PATIENCE}_bs${OC_BATCH_SIZE}_lr${LR_TAG}_j${SLURM_JOB_ID}_nockpt"
OC_RUN_NAME="${OC_RUN_NAME:-$DEFAULT_RUN_NAME}"
if [[ ! "$OC_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ || ${#OC_RUN_NAME} -gt 240 ]]; then
    echo "ERROR: OC_RUN_NAME must use [A-Za-z0-9._-] and be <=240 chars" >&2
    exit 2
fi

OUTPUT_DIR="$OC_RUN_ROOT/results/matched_completion/convergence/$OC_RUN_NAME"
mkdir -p "$OUTPUT_DIR"

export DATA_DIR="$REPO_ROOT/data"
export LLMSEQ_ROOT="$OC_RUN_ROOT"
export PYTHONHASHSEED="$OC_SEED"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$OC_THREADS"
export OPENBLAS_NUM_THREADS="$OC_THREADS"
export MKL_NUM_THREADS="$OC_THREADS"
export NUMEXPR_NUM_THREADS="$OC_THREADS"

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
import pandas
import sklearn
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see the allocated H200")
print(
    f"torch={torch.__version__} sklearn={sklearn.__version__} "
    f"pandas={pandas.__version__} gpu={torch.cuda.get_device_name(0)}"
)
PY

echo "=== FOX CHECKPOINT-FREE DL CONVERGENCE CONFIGURATION ==="
echo "Job / node: $SLURM_JOB_ID / $(hostname)"
echo "Repository: $REPO_ROOT"
echo "Data: $DATA_DIR"
echo "Run root: $OC_RUN_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Model / task / seed: $OC_MODEL / $OC_TASK / $OC_SEED"
echo "Maximum epochs / patience: $OC_MAX_EPOCHS / $OC_PATIENCE"
echo "Train / evaluation batch: $OC_BATCH_SIZE / $OC_EVAL_BATCH"
if [[ -n "$OC_LR" ]]; then
    echo "Constant learning rate: $OC_LR (explicit override)"
else
    echo "Constant learning rate: model RECIPES default"
fi
echo "Threads / diagnostic pairs: $OC_THREADS / $OC_DIAG_PAIRS"
echo "Checkpoint policy: NONE"
echo "Requeue policy: disabled"
module list 2>&1

TRAIN_ARGS=(
    --model "$OC_MODEL"
    --task "$OC_TASK"
    --seed "$OC_SEED"
    --max_epochs "$OC_MAX_EPOCHS"
    --patience "$OC_PATIENCE"
    --batch_size "$OC_BATCH_SIZE"
    --threads "$OC_THREADS"
    --diag_pairs "$OC_DIAG_PAIRS"
    --eval_batch "$OC_EVAL_BATCH"
    --output_dir "$OUTPUT_DIR"
    --require_cuda
)
if [[ -n "$OC_LR" ]]; then
    TRAIN_ARGS+=(--lr "$OC_LR")
fi

ACTIVE_PID=""
forward_walltime_signal() {
    trap - USR1
    echo "=== FOX WALLTIME WARNING: requesting in-memory finalization ===" >&2
    if [[ -n "$ACTIVE_PID" ]]; then
        kill -USR1 "$ACTIVE_PID" 2>/dev/null || true
    fi
}
trap forward_walltime_signal USR1

cd "$REPO_ROOT/repro"
python3 - "$OC_MODEL" <<'PY'
import sys

import torch

from src.oc_completion.train_dl import (
    OC_ENCODING,
    PAD_ID,
    RECIPES,
    create_model,
    encode_oc_sequence,
)

model_name = sys.argv[1]
encoded = encode_oc_sequence("AZ" + "B" * 18).unsqueeze(0)
model = create_model(model_name, RECIPES[model_name]["config"])
if OC_ENCODING["name"] != "oc_pad0_a1_z26":
    raise SystemExit(f"ERROR: unexpected OC encoding: {OC_ENCODING}")
if encoded.shape != (1, 20) or encoded[0, 0].item() != 1:
    raise SystemExit("ERROR: scratch OC encoder does not map A to 1 at width 20")
if encoded[0, 1].item() != 26 or encoded.eq(PAD_ID).any():
    raise SystemExit("ERROR: scratch OC encoder collides with padding")
if model.embedding.num_embeddings != 27 or model.embedding.padding_idx != 0:
    raise SystemExit("ERROR: scratch OC model does not use vocab=27/PAD=0")
print(
    f"scratch_encoding={OC_ENCODING['name']} shape={tuple(encoded.shape)} "
    f"A={encoded[0, 0].item()} Z={encoded[0, 1].item()} "
    f"padding_positions={encoded.eq(PAD_ID).sum().item()}"
)
PY
python3 -u -m src.oc_completion.train_dl_convergence "${TRAIN_ARGS[@]}" &
ACTIVE_PID=$!

STATUS=0
while true; do
    if wait "$ACTIVE_PID"; then
        STATUS=0
        break
    else
        STATUS=$?
    fi
    # Bash's wait is interrupted by USR1 while Python continues finalizing.
    if kill -0 "$ACTIVE_PID" 2>/dev/null; then
        continue
    fi
    break
done
ACTIVE_PID=""
if (( STATUS != 0 )); then
    echo "ERROR: DL convergence trainer exited with status $STATUS" >&2
    exit "$STATUS"
fi

require_file "$OUTPUT_DIR/config.json"
require_file "$OUTPUT_DIR/history.json"
require_file "$OUTPUT_DIR/summary.json"
if ! find "$OUTPUT_DIR" -maxdepth 1 -type f \
        \( -name 'validation_predictions.parquet' -o \
           -name 'validation_predictions.csv.gz' \) -print -quit | grep -q .; then
    echo "ERROR: validation predictions are missing" >&2
    exit 2
fi
if ! find "$OUTPUT_DIR" -maxdepth 1 -type f \
        \( -name 'test_predictions.parquet' -o \
           -name 'test_predictions.csv.gz' \) -print -quit | grep -q .; then
    echo "ERROR: test predictions are missing" >&2
    exit 2
fi

python3 - "$OUTPUT_DIR/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("status") != "complete":
    raise SystemExit(f"ERROR: incomplete summary status: {summary.get('status')}")
if summary.get("checkpoint_saved") is not False:
    raise SystemExit("ERROR: summary does not confirm checkpoint_saved=false")
if summary.get("encoding", {}).get("name") != "oc_pad0_a1_z26":
    raise SystemExit("ERROR: summary does not confirm the corrected OC encoding")
print(
    "termination=" + str(summary.get("termination_reason")),
    "converged_by_patience=" + str(summary.get("converged_by_patience")),
    "epochs=" + str(summary.get("epochs_completed")),
    "best_epoch=" + str(summary.get("best_epoch")),
    "lr=" + str(summary.get("learning_rate")),
)
PY

if find "$OC_RUN_ROOT" \
        \( -type d \( -name 'checkpoints' -o -name 'checkpoint-*' \) -o \
           -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o \
                       -name '*.safetensors' \) \) \
        -print -quit | grep -q .; then
    echo "ERROR: a forbidden checkpoint was written under $OC_RUN_ROOT" >&2
    exit 2
fi

echo "=== DL CONVERGENCE JOB FINISHED ==="
echo "Results: $OUTPUT_DIR"
