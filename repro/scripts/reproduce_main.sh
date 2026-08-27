#!/bin/bash
# Reproduce the main experimental results (Figure 1, Table 2) for one seed.
#
# Modes:
#   --tiny   : 1 epoch, single seed; smoke test (~30 min on 1 CPU + 1 GPU).
#   default  : full protocol (20 epochs for BERT/LLM, 30 for DL; 6 fractions
#              hardcoded in DL_TR_baselines.py: [.01, .10, .30, .50, .75, 1.0]).
#
# This script runs ONE seed (9550). For the paper's 3-seed average (9550, 9551, 9552),
# wrap the calls below in a `for SEED in 9550 9551 9552; do ... --seed $SEED done` loop.
#
# Outputs land in ${RESULTS_DIR:-results}/. Plot Figure 1 with:
#   python -m src.analysis.plot_main_figure --results_dir ${RESULTS_DIR:-results}
#
# LLM fine-tuning (Llama, Qwen) requires HF access — set HF_TOKEN in .env first.

set -euo pipefail
REPRO_ROOT="${REPRO_ROOT:-$PWD}"
cd "$REPRO_ROOT"

MODE="${1:-full}"
if [[ "$MODE" == "--tiny" ]]; then
  EPOCHS=1
  echo "[reproduce_main] TINY mode (1 epoch, 1 seed)"
else
  EPOCHS=20
  echo "[reproduce_main] FULL mode (20 epochs)"
fi

SEED=9550
mkdir -p "${RESULTS_DIR:-results}"

# Note on dataset support per script:
#   --csv_to_use accepts: 6, 9, test_just_pair, alph                  (XGBoost)
#   --number_to_use accepts: 6, 9, test_just_pair, alph (str)         (LLM, BERT, DL_TR_baselines)
#   --number_to_use accepts: 6, 9 only (int)                          (DL_baselines)

for TAG in 6 9 test_just_pair; do
  echo "===================================================="
  echo "Dataset: $TAG"
  echo "===================================================="

  # XGBoost (basic ordinal encoding). Add --run_llm for LLM-embedding variant.
  python -m src.experiments.XGBoost_fraction_experiment \
    --csv_to_use "$TAG" --run_basic --seed "$SEED" || echo "(xgboost failed)"

  # DL_TR_baselines covers LSTM, Transformer, RNNTransformer (paper Table 2 DL row).
  # Models are case-sensitive. Fractions are hardcoded inside the script.
  python -m src.experiments.DL_TR_baselines \
    --number_to_use "$TAG" --models "LSTM,Transformer,RNNTransformer" \
    --final_epochs "$EPOCHS" --seed "$SEED" || echo "(dl_tr failed)"

  # BERT (LoRA fine-tune)
  python -m src.experiments.BERT_fraction_experiment \
    --number_to_use "$TAG" --model_name bert-base-uncased --peft \
    --epochs "$EPOCHS" --seed "$SEED" || echo "(bert failed)"

  # LLM (LoRA fine-tune; requires HF_TOKEN). Skip silently if not set.
  if [[ -n "${HF_TOKEN:-}" ]]; then
    python -m src.experiments.LLM_fraction_experiment \
      --number_to_use "$TAG" --model_name meta-llama/Llama-3.1-8B --peft \
      --epochs "$EPOCHS" --seed "$SEED" || echo "(llm failed)"
  else
    echo "[reproduce_main] HF_TOKEN not set; skipping LLM fine-tuning."
  fi
done

echo "===================================================="
echo "Plotting Figure 1 ..."
python -m src.analysis.plot_main_figure \
  --results_dir "${RESULTS_DIR:-results}" \
  --output figures/auc_combined_main_1row.png || echo "(plot failed; rerun once results JSON exist)"

echo "[reproduce_main] done."
