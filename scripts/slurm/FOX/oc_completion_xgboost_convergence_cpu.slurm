#!/bin/bash -l
# One-task, one-seed, checkpoint-free OC XGBoost convergence job on FOX CPUs.
#
# XGBoost uses the canonical all-offset pair-count features.  Boosting rounds
# are the epoch analogue, and patience is measured in rounds without an
# observed-validation-logloss improvement.  This job intentionally requests
# no GPU and does not serialize the fitted booster.
#
# Before submission, create the Slurm log directory from the LITE repo root:
#   mkdir -p logs
#
# Required environment variables:
#   OC_TASK=ocdet|ocnoisy
#   OC_RUN_ROOT=<fresh absolute directory under /fp/projects01>
#
# Example:
#   export REPO_ROOT="$PWD"
#   export OC_TASK=ocdet
#   export OC_RUN_ROOT=/fp/projects01/ec12/ec-michechi/xgb_ocdet_s9550
#   export OC_MAX_ROUNDS=2000 OC_PATIENCE=50 OC_LR=0.05
#   sbatch scripts/slurm/FOX/oc_completion_xgboost_convergence_cpu.slurm

#SBATCH --job-name=oc_xgb_conv
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The default FOX CPU partition is intentional: do not add accel/GPU options.
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Four hours are reserved for ordinary and matched-pair evaluation after the
# Python callback stops training at a boosting-round boundary.
#SBATCH --signal=B:USR1@14400
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_xgb_convergence_%j.out

set -o errexit
set -o nounset
set -o pipefail

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
TRAINER="$REPO_ROOT/repro/src/oc_completion/train_xgb_convergence.py"
if [[ ! -f "$TRAINER" ]]; then
    echo "ERROR: XGBoost convergence trainer is missing: $TRAINER" >&2
    exit 2
fi

: "${OC_TASK:?ERROR: export OC_TASK=ocdet or OC_TASK=ocnoisy}"
: "${OC_RUN_ROOT:?ERROR: export OC_RUN_ROOT to a fresh project-storage directory}"

OC_SEED="${OC_SEED:-9550}"
OC_MAX_ROUNDS="${OC_MAX_ROUNDS:-2000}"
OC_PATIENCE="${OC_PATIENCE:-50}"
OC_LR="${OC_LR:-0.05}"
OC_MAX_DEPTH="${OC_MAX_DEPTH:-6}"
OC_SUBSAMPLE="${OC_SUBSAMPLE:-0.8}"
OC_THREADS="${OC_THREADS:-$SLURM_CPUS_PER_TASK}"

case "$OC_TASK" in
    ocdet|ocnoisy) ;;
    *)
        echo "ERROR: OC_TASK must be ocdet or ocnoisy: $OC_TASK" >&2
        exit 2
        ;;
esac

for integer_name in \
        OC_SEED OC_MAX_ROUNDS OC_PATIENCE OC_MAX_DEPTH OC_THREADS; do
    integer_value="${!integer_name}"
    if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $integer_name must be a non-negative integer: $integer_value" >&2
        exit 2
    fi
done
for positive_name in OC_MAX_ROUNDS OC_PATIENCE OC_MAX_DEPTH OC_THREADS; do
    if (( ${!positive_name} < 1 )); then
        echo "ERROR: $positive_name must be positive: ${!positive_name}" >&2
        exit 2
    fi
done
if (( OC_SEED > 2147483647 )); then
    echo "ERROR: OC_SEED must be at most 2147483647: $OC_SEED" >&2
    exit 2
fi
if (( OC_THREADS > SLURM_CPUS_PER_TASK )); then
    echo "ERROR: OC_THREADS=$OC_THREADS exceeds allocated CPUs=$SLURM_CPUS_PER_TASK" >&2
    exit 2
fi

python3 - "$OC_LR" "$OC_SUBSAMPLE" <<'PY'
import math
import sys

names = ("OC_LR", "OC_SUBSAMPLE")
values = dict(zip(names, sys.argv[1:]))

def finite_float(name):
    try:
        value = float(values[name])
    except ValueError as exc:
        raise SystemExit(f"ERROR: {name} must be numeric: {values[name]}") from exc
    if not math.isfinite(value):
        raise SystemExit(f"ERROR: {name} must be finite: {values[name]}")
    return value

learning_rate = finite_float("OC_LR")
subsample = finite_float("OC_SUBSAMPLE")
if learning_rate <= 0:
    raise SystemExit("ERROR: OC_LR must be positive")
if not 0.0 < subsample <= 1.0:
    raise SystemExit("ERROR: OC_SUBSAMPLE must be in (0, 1]")
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
if [[ -n "$(find "$OC_RUN_ROOT" -mindepth 1 -print -quit)" ]]; then
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

DEFAULT_RUN_NAME="xgb_all_offset_${OC_TASK}_s${OC_SEED}_r${OC_MAX_ROUNDS}_p${OC_PATIENCE}_lr${OC_LR}_d${OC_MAX_DEPTH}_sub${OC_SUBSAMPLE}_j${SLURM_JOB_ID}_nockpt"
OC_RUN_NAME="${OC_RUN_NAME:-$DEFAULT_RUN_NAME}"
if [[ ! "$OC_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ || ${#OC_RUN_NAME} -gt 240 ]]; then
    echo "ERROR: OC_RUN_NAME must use [A-Za-z0-9._-] and be at most 240 characters" >&2
    exit 2
fi
OUTPUT_DIR="$OC_RUN_ROOT/results/matched_completion/convergence/$OC_RUN_NAME"
mkdir -p "$OUTPUT_DIR"

export DATA_DIR="$REPO_ROOT/data"
export LLMSEQ_ROOT="$OC_RUN_ROOT"
export PYTHONHASHSEED="$OC_SEED"
export PYTHONUNBUFFERED=1
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
import sklearn
import xgboost
from xgboost import XGBClassifier

major = int(xgboost.__version__.split(".", 1)[0])
if major < 2:
    raise SystemExit(
        f"ERROR: xgboost>=2 is required for explicit device='cpu'; found {xgboost.__version__}"
    )
probe = XGBClassifier(tree_method="hist", device="cpu", n_estimators=1)
if probe.get_params().get("device") != "cpu":
    raise SystemExit("ERROR: this XGBoost build did not accept device='cpu'")
print(f"xgboost={xgboost.__version__} sklearn={sklearn.__version__} device=cpu")
PY

echo "=== FOX XGBOOST CHECKPOINT-FREE CONVERGENCE CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repository: $REPO_ROOT"
echo "Data: $DATA_DIR"
echo "Results root: $OC_RUN_ROOT"
echo "Run output: $OUTPUT_DIR"
echo "Task / seed: $OC_TASK / $OC_SEED"
echo "Maximum rounds / patience: $OC_MAX_ROUNDS / $OC_PATIENCE"
echo "Learning rate / depth / subsample: $OC_LR / $OC_MAX_DEPTH / $OC_SUBSAMPLE"
echo "Threads / memory: $OC_THREADS / 128G"
echo "Tree method / device: hist / cpu"
echo "Checkpoint policy: NONE"
echo "Requeue policy: disabled"
module list 2>&1

TRAIN_ARGS=(
    --task "$OC_TASK"
    --seed "$OC_SEED"
    --max_rounds "$OC_MAX_ROUNDS"
    --patience "$OC_PATIENCE"
    --lr "$OC_LR"
    --max_depth "$OC_MAX_DEPTH"
    --subsample "$OC_SUBSAMPLE"
    --threads "$OC_THREADS"
    --output_dir "$OUTPUT_DIR"
)

ACTIVE_PID=""
forward_walltime_signal() {
    trap - USR1
    echo "=== FOX WALLTIME WARNING: stopping after the current XGBoost round ===" >&2
    if [[ -n "$ACTIVE_PID" ]]; then
        kill -USR1 "$ACTIVE_PID" 2>/dev/null || true
    fi
}
trap forward_walltime_signal USR1

cd "$REPO_ROOT/repro"
python3 -u -m src.oc_completion.train_xgb_convergence "${TRAIN_ARGS[@]}" &
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
    echo "ERROR: XGBoost convergence trainer exited with status $STATUS" >&2
    exit "$STATUS"
fi

require_file "$OUTPUT_DIR/summary.json"
python3 - "$OUTPUT_DIR/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("status") != "complete":
    raise SystemExit(f"ERROR: incomplete summary status: {summary.get('status')}")
if summary.get("checkpoint_saved") is not False:
    raise SystemExit("ERROR: summary does not confirm checkpoint_saved=false")
if int(summary.get("rounds_trained", 0)) < 1:
    raise SystemExit("ERROR: no completed boosting round was recorded")
print(
    "termination=" + str(summary.get("termination_reason")),
    "rounds=" + str(summary.get("rounds_trained")),
    "best_round=" + str(summary.get("best_round")),
)
PY

if find "$OC_RUN_ROOT" \
        \( -type d -name 'checkpoint-*' -o \
           -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o \
                       -name '*.joblib' -o -name '*.ubj' -o -name '*.json.model' -o \
                       -name '*.safetensors' \) \) \
        -print -quit | grep -q .; then
    echo "ERROR: a forbidden model checkpoint was written under $OC_RUN_ROOT" >&2
    exit 2
fi

echo "=== XGBOOST OC CONVERGENCE JOB FINISHED ==="
