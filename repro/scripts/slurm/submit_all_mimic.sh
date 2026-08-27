#!/bin/bash
# ==============================================================
# Submit all MIMIC ordered vs shuffled experiments
# ==============================================================
# Usage:
#   bash scripts/slurm/submit_all_mimic.sh
#   bash scripts/slurm/submit_all_mimic.sh --dry-run   # preview without submitting
#
# Required env:
#   REPRO_ROOT, CONTAINER (used by mimic_experiment.slurm). HF_TOKEN if you
#   want to fine-tune a gated LLM (e.g. Llama-3.x).
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/mimic_experiment.slurm"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN (no jobs will be submitted) ==="
fi

# ---------- Experiment matrix ----------
# Format: MODEL|DATA|TIME|MEM|EXTRA_ARGS
EXPERIMENTS=(
    "xgboost|ordered|00:30:00|16GB|"
    "xgboost|shuffled|00:30:00|16GB|"
    "logreg|ordered|00:30:00|16GB|"
    "logreg|shuffled|00:30:00|16GB|"
    "transformer|ordered|04:00:00|32GB|"
    "transformer|shuffled|04:00:00|32GB|"
    "lstm|ordered|04:00:00|32GB|"
    "lstm|shuffled|04:00:00|32GB|"
    "bilstm|ordered|04:00:00|32GB|"
    "bilstm|shuffled|04:00:00|32GB|"
    "bert|ordered|08:00:00|64GB|--llm_batch_size 16 --max_length 512"
    "bert|shuffled|08:00:00|64GB|--llm_batch_size 16 --max_length 512"
)

mkdir -p logs

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r MODEL DATA TIME MEM EXTRA <<< "$exp"

    JOB_NAME="mimic_${MODEL}_${DATA}"
    LOG_FILE="./logs/${JOB_NAME}_%j.out"

    echo "Submitting ${JOB_NAME} (time=${TIME}, mem=${MEM})"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  sbatch --job-name=${JOB_NAME} --output=${LOG_FILE} --time=${TIME} --mem=${MEM} --export=ALL,MODEL=${MODEL},DATA=${DATA},EXTRA_ARGS=\"${EXTRA}\" ${TEMPLATE}"
    else
        sbatch \
            --job-name="${JOB_NAME}" \
            --output="${LOG_FILE}" \
            --time="${TIME}" \
            --mem="${MEM}" \
            --export=ALL,MODEL="${MODEL}",DATA="${DATA}",EXTRA_ARGS="${EXTRA}" \
            "${TEMPLATE}"
    fi
done

echo
echo "=== All jobs submitted ==="
