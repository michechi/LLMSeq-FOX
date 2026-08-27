#!/bin/bash
# Matched-completion Stage C on FOX (Educloud): BERT LoRA / Llama-1B LoRA /
# Llama-1B full FT, array over (arm x task x seed).
#
#   sbatch scripts/slurm/FOX/oc_completion_hf.slurm
#
# Fully resumable: rerunning the same array task continues from last.pt and
# skips runs whose done.json exists. Requires the regenerated OC datasets and
# pair CSVs under $REPO_ROOT/data/simulation/oc_completion (rsync them from
# the CPU node, or run gen_datasets + gen_pairs once - both are seeded).
#
#SBATCH --job-name=oc_completion_hf
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/oc_completion_hf_%A_%a.out

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# 2024a stack: the 2022b stack has no H200 sm_90 kernels
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3 || \
  module load nlpl-llmtools/2024a || true
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3 || true
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export DATA_DIR="${REPO_ROOT}/data"
export LLMSEQ_ROOT="${REPO_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
# meta-llama/Llama-3.2-1B is gated: export HF_TOKEN before submitting, or
# pre-populate HF_HUB_CACHE with the snapshot.

cd "${REPO_ROOT}/repro"

# array decode: arm (3) x task (2) x seed (3) minus llama_full extra seeds
# 0-5:   bert_lora   {ocdet,ocnoisy} x {9550,9551,9552}
# 6-11:  llama_lora  {ocdet,ocnoisy} x {9550,9551,9552}
# 12-13: llama_full  {ocdet,ocnoisy} x 9550   (add 9551/9552 when feasible)
IDX=${SLURM_ARRAY_TASK_ID}
SEEDS=(9550 9551 9552)
TASKS=(ocdet ocnoisy)
if [ "$IDX" -lt 6 ]; then
  ARM=bert_lora;  TASK=${TASKS[$((IDX / 3))]};        SEED=${SEEDS[$((IDX % 3))]}
elif [ "$IDX" -lt 12 ]; then
  J=$((IDX - 6)); ARM=llama_lora; TASK=${TASKS[$((J / 3))]}; SEED=${SEEDS[$((J % 3))]}
else
  J=$((IDX - 12)); ARM=llama_full; TASK=${TASKS[$J]}; SEED=9550
fi

echo "[oc_hf] arm=$ARM task=$TASK seed=$SEED host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

python -m src.oc_completion.train_hf \
    --arm "$ARM" --task "$TASK" --seed "$SEED" --resume --max_length 64

TD=$([ "$TASK" = ocdet ] && echo oc_deterministic || echo oc_noisy)
CKPT="${REPO_ROOT}/checkpoints/oc_completion/${TD}/${ARM}/seed_${SEED}/best.pt"
if [ -f "$CKPT" ]; then
  python -m src.oc_completion.eval_pairs \
      --model_kind hf --checkpoint "$CKPT" --task "$TASK" --hf_batch_size 256
fi
