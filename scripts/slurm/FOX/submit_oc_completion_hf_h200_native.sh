#!/bin/bash
# Submit serialized OC Hugging Face train/eval arrays on FOX.
#
# Usage:
#   bash scripts/slurm/FOX/submit_oc_completion_hf_h200_native.sh primary
#   bash scripts/slurm/FOX/submit_oc_completion_hf_h200_native.sh bert_lora
#   bash scripts/slurm/FOX/submit_oc_completion_hf_h200_native.sh llama_lora
#   bash scripts/slurm/FOX/submit_oc_completion_hf_h200_native.sh llama_full
#   bash scripts/slurm/FOX/submit_oc_completion_hf_h200_native.sh llama_full_extra
#
# Set OC_MAX_PARALLEL to use more than one H200 concurrently (default: 1).
# Set OC_MAX_REQUEUES to change the per-task continuation cap (default: 3).
# By default, a successful run retains best.pt and removes last.pt/final.pt to
# stay within the ec12 quota.  Export OC_PRUNE_COMPLETED=0 to keep all three.

set -o errexit
set -o nounset
set -o pipefail

GROUP="${1:-}"
MAX_PARALLEL="${OC_MAX_PARALLEL:-1}"
if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: OC_MAX_PARALLEL must be a positive integer" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath -e -- "$SCRIPT_DIR/../../..")"
TRAIN_FILE="$SCRIPT_DIR/oc_completion_hf_h200_native.slurm"
EVAL_FILE="$SCRIPT_DIR/oc_completion_hf_eval_h200_native.slurm"

: "${OC_RUN_ROOT:?ERROR: export OC_RUN_ROOT to project storage before submission}"
if [[ "$OC_RUN_ROOT" != /* ]]; then
    echo "ERROR: OC_RUN_ROOT must be an absolute project-storage path" >&2
    exit 2
fi
mkdir -p "$OC_RUN_ROOT" "$REPO_ROOT/logs"
OC_RUN_ROOT="$(realpath -e -- "$OC_RUN_ROOT")"
HOME_ROOT="$(realpath -e -- "$HOME")"
case "$OC_RUN_ROOT/" in
    "$HOME_ROOT/"*)
        echo "ERROR: use /fp/projects01 storage for OC_RUN_ROOT, not home." >&2
        exit 2
        ;;
esac
export OC_RUN_ROOT
export REPO_ROOT
export HF_HOME="${HF_HOME:-$OC_RUN_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [[ "$HF_HOME" != /* || "$HF_HUB_CACHE" != /* ]]; then
    echo "ERROR: HF_HOME and HF_HUB_CACHE must be absolute paths" >&2
    exit 2
fi
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
HF_HOME="$(realpath -e -- "$HF_HOME")"
HF_HUB_CACHE="$(realpath -e -- "$HF_HUB_CACHE")"
for cache_path in "$HF_HOME" "$HF_HUB_CACHE"; do
    case "$cache_path/" in
        "$HOME_ROOT/"*)
            echo "ERROR: Hugging Face cache must use project storage: $cache_path" >&2
            exit 2
            ;;
    esac
done
export HF_HOME HF_HUB_CACHE

LAST_TRAIN_ID=""
LAST_EVAL_ID=""

submit_group() {
    local label="$1"
    local array_range="$2"
    local predecessor="${3:-}"
    local train_dependency=()
    if [[ -n "$predecessor" ]]; then
        train_dependency=(
            --dependency="afterok:$predecessor"
            --kill-on-invalid-dep=yes
        )
    fi

    local train_submission
    train_submission="$(
        sbatch --parsable \
            --array="${array_range}%${MAX_PARALLEL}" \
            --job-name="oc_${label}_train" \
            --output="$REPO_ROOT/logs/oc_${label}_train_%A_%a.out" \
            "${train_dependency[@]}" \
            --export="ALL,REPO_ROOT=$REPO_ROOT,OC_RUN_ROOT=$OC_RUN_ROOT" \
            "$TRAIN_FILE"
    )"
    LAST_TRAIN_ID="${train_submission%%;*}"

    local eval_submission
    eval_submission="$(
        sbatch --parsable \
            --array="${array_range}%1" \
            --job-name="oc_${label}_eval" \
            --output="$REPO_ROOT/logs/oc_${label}_eval_%A_%a.out" \
            --dependency="afterok:$LAST_TRAIN_ID" \
            --kill-on-invalid-dep=yes \
            --export="ALL,REPO_ROOT=$REPO_ROOT,OC_RUN_ROOT=$OC_RUN_ROOT" \
            "$EVAL_FILE"
    )"
    LAST_EVAL_ID="${eval_submission%%;*}"

    echo "$label: train=$LAST_TRAIN_ID eval=$LAST_EVAL_ID"
}

cd "$REPO_ROOT"
case "$GROUP" in
    bert_lora)
        submit_group bert_lora 0-5
        ;;
    llama_lora)
        submit_group llama_lora 6-11
        ;;
    llama_full)
        submit_group llama_full 12-13
        ;;
    llama_full_extra)
        submit_group llama_full_extra 14-17
        ;;
    primary)
        submit_group bert_lora 0-5
        predecessor="$LAST_EVAL_ID"
        submit_group llama_lora 6-11 "$predecessor"
        predecessor="$LAST_EVAL_ID"
        submit_group llama_full 12-13 "$predecessor"
        echo "Primary pipeline final evaluation job: $LAST_EVAL_ID"
        ;;
    *)
        echo "Usage: $0 {primary|bert_lora|llama_lora|llama_full|llama_full_extra}" >&2
        exit 2
        ;;
esac
