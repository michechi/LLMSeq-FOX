#!/bin/bash
# Matched-completion Stage B on FOX: LSTM / Transformer / RNNTransformer,
# array over (model x task x seed). Normally run locally on a CPU node via
# scripts/oc_completion/stage_b_local.sh; this SLURM variant exists for
# cluster continuation. Resumable (done.json + last.pt).
#
#SBATCH --job-name=oc_completion_dl
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=a100:1
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/oc_completion_dl_%A_%a.out

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export DATA_DIR="${REPO_ROOT}/data"
export LLMSEQ_ROOT="${REPO_ROOT}"
cd "${REPO_ROOT}/repro"

IDX=${SLURM_ARRAY_TASK_ID}
MODELS=(LSTM Transformer RNNTransformer)
TASKS=(ocdet ocnoisy)
SEEDS=(9550 9551 9552)
MODEL=${MODELS[$((IDX / 6))]}
TASK=${TASKS[$(((IDX % 6) / 3))]}
SEED=${SEEDS[$((IDX % 3))]}

echo "[oc_dl] model=$MODEL task=$TASK seed=$SEED host=$(hostname)"

python -m src.oc_completion.train_dl \
    --model "$MODEL" --task "$TASK" --seed "$SEED" --resume

TD=$([ "$TASK" = ocdet ] && echo oc_deterministic || echo oc_noisy)
CKPT="${REPO_ROOT}/checkpoints/oc_completion/${TD}/$(echo $MODEL | tr 'A-Z' 'a-z')/seed_${SEED}/best.pt"
if [ -f "$CKPT" ]; then
  python -m src.oc_completion.eval_pairs \
      --model_kind dl --checkpoint "$CKPT" --task "$TASK"
fi
