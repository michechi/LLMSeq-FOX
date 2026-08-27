#!/bin/bash
# Stage B (96-CPU node): feature baselines + LSTM/Transformer/RNNTransformer,
# 3 seeds x 2 tasks, followed by pair evaluation of every artifact.
# Fully resumable: completed training runs are skipped via done.json markers;
# completed baseline fits are skipped via their .joblib artifacts.
#
# Launch detached:  tmux new-session -d -s oc_stage_b \
#     'bash /root/LLMSeq/scripts/oc_completion/stage_b_local.sh \
#        2>&1 | tee -a /root/LLMSeq/logs/oc_completion/stage_b.log'
set -uo pipefail

REPO=/root/LLMSeq
PY=$REPO/.venv/bin/python
export DATA_DIR=$REPO/data
# A100 node (22 CPUs): DL training runs on the GPU; keep CPU threads modest.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd $REPO/repro

LOGDIR=$REPO/logs/oc_completion
mkdir -p $LOGDIR

SEEDS="9550 9551 9552"
MODELS="LSTM Transformer RNNTransformer"
TASKS="ocdet ocnoisy"

# ---------------------------------------------------------------- baselines
for TASK in $TASKS; do
  TD=$([ "$TASK" = ocdet ] && echo oc_deterministic || echo oc_noisy)
  SUMMARY=$REPO/checkpoints/oc_completion/$TD/baselines/summary.json
  if [ -f "$SUMMARY" ]; then
    echo "[stage_b] baselines $TASK already fitted - skip"
  else
    echo "[stage_b] fitting baselines $TASK ..."
    OMP_NUM_THREADS=45 OPENBLAS_NUM_THREADS=45 MKL_NUM_THREADS=45 \
      $PY -m src.oc_completion.train_baselines --task $TASK \
      > $LOGDIR/baselines_$TASK.log 2>&1 &
  fi
done
wait

# ---------------------------------------------------------------- DL training
JOBLIST=$(mktemp)
for TASK in $TASKS; do
  for MODEL in $MODELS; do
    for SEED in $SEEDS; do
      echo "$TASK $MODEL $SEED" >> $JOBLIST
    done
  done
done

run_one() {
  local TASK=$1 MODEL=$2 SEED=$3
  local LOG=$LOGDIR/dl_${MODEL}_${TASK}_s${SEED}.log
  echo "[stage_b] train $MODEL/$TASK/s$SEED"
  $PY -m src.oc_completion.train_dl --task $TASK --model $MODEL \
      --seed $SEED --resume --threads 4 >> $LOG 2>&1
  local TD=$([ "$TASK" = ocdet ] && echo oc_deterministic || echo oc_noisy)
  local CKPT=$REPO/checkpoints/oc_completion/$TD/$(echo $MODEL | tr 'A-Z' 'a-z')/seed_${SEED}/best.pt
  if [ -f "$CKPT" ]; then
    $PY -m src.oc_completion.eval_pairs --model_kind dl \
        --checkpoint $CKPT --task $TASK --threads 4 >> $LOG 2>&1
  else
    echo "[stage_b] MISSING checkpoint $CKPT" >> $LOG
  fi
}
export -f run_one
export PY REPO LOGDIR DATA_DIR OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS

xargs -a $JOBLIST -L1 -P 4 bash -c 'run_one $0 $1 $2'
rm -f $JOBLIST

# ------------------------------------------------------- baseline pair evals
for TASK in $TASKS; do
  TD=$([ "$TASK" = ocdet ] && echo oc_deterministic || echo oc_noisy)
  for ART in $REPO/checkpoints/oc_completion/$TD/baselines/*.joblib; do
    case "$ART" in *_smoke.joblib) continue;; esac
    echo "[stage_b] eval baseline $(basename $ART) on $TASK pairs"
    $PY -m src.oc_completion.eval_pairs --model_kind baseline \
        --checkpoint $ART --task $TASK --threads 16 \
        >> $LOGDIR/eval_baselines_$TASK.log 2>&1
  done
done

echo "[stage_b] ALL DONE $(date -Is)"
