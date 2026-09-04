#!/bin/bash -l
# One-seed, checkpoint-free BERT-LoRA convergence diagnostic on OC-Noisy.
#
# This is intentionally NOT an array and NOT resumable.  It trains on the
# ordinary noisy sequence-classification task, keeps the best trainable state
# in RAM, then evaluates both ordinary test metrics and all matched pairs in
# the same process.  No best.pt/last.pt/final.pt file is created.
#
# Submit from the LITE repository root after exporting a fresh project-storage
# OC_RUN_ROOT.  Example:
#   export OC_RUN_ROOT=/fp/projects01/ec12/ec-michechi/oc_bert_noisy_s9550_nockpt
#   export HF_HOME=/fp/projects01/ec12/ec-michechi/huggingface
#   export HF_HUB_CACHE=$HF_HOME/hub
#   export OC_SEED=9550
#   export OC_MAX_EPOCHS=80
#   sbatch scripts/slurm/FOX/oc_completion_bert_noisy_convergence_h200.slurm

#SBATCH --job-name=oc_bert_noisy_conv
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:USR1@7200
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_bert_noisy_convergence_%j.out

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
if [[ ! -f "$REPO_ROOT/repro/src/oc_completion/train_bert_convergence.py" ]]; then
    echo "ERROR: convergence trainer is missing from REPO_ROOT=$REPO_ROOT" >&2
    exit 2
fi

: "${OC_RUN_ROOT:?ERROR: export OC_RUN_ROOT to a fresh project-storage directory}"
if [[ "$OC_RUN_ROOT" != /* ]]; then
    echo "ERROR: OC_RUN_ROOT must be absolute: $OC_RUN_ROOT" >&2
    exit 2
fi
mkdir -p "$OC_RUN_ROOT"
OC_RUN_ROOT="$(realpath -e -- "$OC_RUN_ROOT")"

HOME_ROOT="$(realpath -e -- "$HOME")"
case "$OC_RUN_ROOT/" in
    "$HOME_ROOT/"*)
        echo "ERROR: OC_RUN_ROOT must be under /fp/projects01, not home." >&2
        exit 2
        ;;
esac
if find "$OC_RUN_ROOT" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: OC_RUN_ROOT must be fresh and empty so result files cannot collide:" >&2
    echo "       $OC_RUN_ROOT" >&2
    exit 2
fi
case "$OC_RUN_ROOT/" in
    /fp/projects01/*) ;;
    *)
        echo "ERROR: OC_RUN_ROOT is not under /fp/projects01: $OC_RUN_ROOT" >&2
        exit 2
        ;;
esac

OC_SEED="${OC_SEED:-9550}"
OC_MAX_EPOCHS="${OC_MAX_EPOCHS:-80}"
OC_PATIENCE="${OC_PATIENCE:-3}"
for value_name in OC_SEED OC_MAX_EPOCHS OC_PATIENCE; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $value_name must be a non-negative integer: $value" >&2
        exit 2
    fi
done
if (( OC_MAX_EPOCHS < 1 )); then
    echo "ERROR: OC_MAX_EPOCHS must be positive" >&2
    exit 2
fi
if (( OC_PATIENCE != 3 )); then
    echo "ERROR: this protocol fixes OC_PATIENCE=3" >&2
    exit 2
fi

OC_RUN_NAME="${OC_RUN_NAME:-bert_lora_ocnoisy_seed${OC_SEED}_e${OC_MAX_EPOCHS}_p3_nockpt}"
if [[ ! "$OC_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: OC_RUN_NAME contains unsupported characters: $OC_RUN_NAME" >&2
    exit 2
fi
OUTPUT_DIR="$OC_RUN_ROOT/results/matched_completion/convergence/$OC_RUN_NAME"
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: convergence output already exists; choose a fresh OC_RUN_ROOT/name:" >&2
    echo "       $OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

export DATA_DIR="$REPO_ROOT/data"
export LLMSEQ_ROOT="$OC_RUN_ROOT"
export HF_HOME="${HF_HOME:-$OC_RUN_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [[ "$HF_HOME" != /* || "$HF_HUB_CACHE" != /* ]]; then
    echo "ERROR: HF_HOME and HF_HUB_CACHE must be absolute" >&2
    exit 2
fi
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
HF_HOME="$(realpath -e -- "$HF_HOME")"
HF_HUB_CACHE="$(realpath -e -- "$HF_HUB_CACHE")"
export HF_HOME HF_HUB_CACHE
for cache_path in "$HF_HOME" "$HF_HUB_CACHE"; do
    case "$cache_path/" in
        "$HOME_ROOT/"*)
            echo "ERROR: Hugging Face cache must not use home: $cache_path" >&2
            exit 2
            ;;
    esac
    case "$cache_path/" in
        /fp/projects01/*) ;;
        *)
            echo "ERROR: Hugging Face cache must be under /fp/projects01: $cache_path" >&2
            exit 2
            ;;
    esac
done

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required input is missing: $1" >&2
        exit 2
    fi
}

for split in train val test; do
    require_file "$DATA_DIR/simulation/oc_completion/X_${split}_ocnoisy.csv"
    require_file "$DATA_DIR/simulation/oc_completion/y_${split}_ocnoisy.csv"
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

echo "=== FOX CHECKPOINT-FREE CONVERGENCE CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repository: $REPO_ROOT"
echo "Data: $DATA_DIR"
echo "Results root: $OC_RUN_ROOT"
echo "Run output: $OUTPUT_DIR"
echo "HF cache: $HF_HUB_CACHE"
echo "Model/task/seed: bert_lora / ocnoisy / $OC_SEED"
echo "Maximum epochs: $OC_MAX_EPOCHS"
echo "Patience: $OC_PATIENCE"
echo "Checkpoint policy: NONE"
echo "Requeue policy: disabled"
module list 2>&1

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
python3 -u -m src.oc_completion.train_bert_convergence \
    --task ocnoisy \
    --seed "$OC_SEED" \
    --max_epochs "$OC_MAX_EPOCHS" \
    --patience "$OC_PATIENCE" \
    --max_length 64 \
    --threads "$SLURM_CPUS_PER_TASK" \
    --workers 4 \
    --diag_pairs 2000 \
    --eval_batch 64 \
    --output_dir "$OUTPUT_DIR" \
    --require_cuda &
ACTIVE_PID=$!

while true; do
    if wait "$ACTIVE_PID"; then
        STATUS=0
        break
    else
        STATUS=$?
    fi
    # A trapped USR1 interrupts bash's wait even though the Python process is
    # still finalizing.  Keep waiting instead of letting Slurm kill the child.
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

summary = json.load(open(sys.argv[1]))
if summary.get("checkpoint_saved") is not False:
    raise SystemExit("ERROR: summary does not confirm checkpoint_saved=false")
print(
    "termination=" + str(summary.get("termination_reason")),
    "converged_by_patience=" + str(summary.get("converged_by_patience")),
    "epochs=" + str(summary.get("epochs_completed")),
    "best_epoch=" + str(summary.get("best_epoch")),
)
PY

if find "$OC_RUN_ROOT" -type f \
        \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) \
        -print -quit | grep -q .; then
    echo "ERROR: a forbidden model checkpoint was written under $OC_RUN_ROOT" >&2
    exit 2
fi

echo "=== FOX CHECKPOINT-FREE CONVERGENCE RUN COMPLETE ==="
echo "Results: $OUTPUT_DIR"
