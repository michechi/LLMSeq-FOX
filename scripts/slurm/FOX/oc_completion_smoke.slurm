#!/bin/bash
# One-job smoke test of the OC matched-completion pipeline on FOX.
# Validates on the cluster, in order:
#   1. unit + invariant tests (oracle, pair algebra, generators)
#   2. smoke dataset + pair generation (seeded, seconds, CPU)
#   3. bert_lora / llama_lora / llama_full smoke training (1 epoch, 256 rows)
#   4. checkpoint -> pair-evaluation path for a Llama checkpoint
# Expected wallclock: 15-30 min. Nothing here touches the full runs.
#
#   export HF_TOKEN=...           # needed for meta-llama/Llama-3.2-1B (gated);
#                                 # bert-base-uncased downloads without it
#   sbatch scripts/slurm/FOX/oc_completion_smoke.slurm
#
#SBATCH --job-name=oc_completion_smoke
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/oc_completion_smoke_%j.out

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# 2024a stack (H200 needs sm_90 kernels; 2022b lacks them).
# VERIFY the exact names with `module avail nlpl-` - these follow the
# mimic_aki_h200_native.slurm precedent.
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3
# peft is required for the LoRA arms; on the 2022b stack it came via
# nlpl-llmtools - load the 2024a equivalent here if the check below fails.
python - <<'EOF'
import peft, transformers, torch, pytest
print(f"peft {peft.__version__}  transformers {transformers.__version__}  "
      f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
EOF

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export DATA_DIR="${REPO_ROOT}/data"
export LLMSEQ_ROOT="${REPO_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
cd "${REPO_ROOT}/repro"

echo "== 1. unit + invariant tests =="
python -m pytest tests/test_oc_completion.py -q

echo "== 2. smoke data + pairs (seeded) =="
if [ ! -f "${DATA_DIR}/simulation/oc_completion/X_train_ocdet_smoke.csv" ]; then
  python -m src.oc_completion.gen_datasets --smoke
fi
if [ ! -f "${DATA_DIR}/simulation/oc_completion/pairs_smoke/pairs_two_hole_heldout_val_smoke.csv" ]; then
  python -m src.oc_completion.gen_pairs --smoke
fi

echo "== 3. smoke training: bert_lora, llama_lora, llama_full =="
for ARM in bert_lora llama_lora llama_full; do
  echo "-- $ARM"
  python -m src.oc_completion.train_hf --arm "$ARM" --task ocdet --seed 9550 \
      --smoke --max_epochs 1 --max_train_rows 256 --diag_pairs 20 \
      --eval_batch 128 --threads 8
done

echo "== 4. checkpoint -> pair evaluation (llama_full) =="
python -m src.oc_completion.eval_pairs --model_kind hf \
    --checkpoint "${REPO_ROOT}/checkpoints/oc_completion/oc_deterministic/llama_full/seed_9550_smoke/best.pt" \
    --task ocdet --smoke --hf_batch_size 256 --threads 8

echo "== OC COMPLETION SMOKE: ALL PASSED =="
