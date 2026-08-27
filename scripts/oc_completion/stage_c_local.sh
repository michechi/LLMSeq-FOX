#!/bin/bash
# Stage C (96-CPU node): pretrained models, priority order.
#   1. BERT-base LoRA        - seeds 9550/9551/9552 x {ocdet, ocnoisy}
#   2. Llama-3.2-1B LoRA     - full-scale run launched (resumable); CPU
#                              throughput (~2 samples/s) means completion
#                              requires the GPU cluster (see slurm scripts)
#   3. Llama-3.2-1B full FT  - same status as (2)
# Waits for Stage B to finish before claiming the CPUs.
# Fully resumable (done.json markers + last.pt epoch checkpoints).
#
# Launch detached:  tmux new-session -d -s oc_stage_c \
#     'bash /root/LLMSeq/scripts/oc_completion/stage_c_local.sh \
#        2>&1 | tee -a /root/LLMSeq/logs/oc_completion/stage_c.log'
set -uo pipefail

REPO=/root/LLMSeq
PY=/root/kip-venv/bin/python
export DATA_DIR=$REPO/data
export HF_HUB_CACHE=/root/hf_cache
export TOKENIZERS_PARALLELISM=false
cd $REPO/repro
LOGDIR=$REPO/logs/oc_completion
mkdir -p $LOGDIR

# ------------------------------------------------ wait for Stage B to finish
while ! grep -q "ALL DONE" $LOGDIR/stage_b.log 2>/dev/null; do
  echo "[stage_c] waiting for stage B ... $(date -Is)"
  sleep 300
done
echo "[stage_c] stage B finished - starting"

eval_ckpt() {  # arm task
  local TD=$([ "$2" = ocdet ] && echo oc_deterministic || echo oc_noisy)
  local CKPT=$REPO/checkpoints/oc_completion/$TD/$1/seed_$3/best.pt
  if [ -f "$CKPT" ]; then
    $PY -m src.oc_completion.eval_pairs --model_kind hf \
        --checkpoint $CKPT --task $2 --threads 8 --hf_batch_size 256 \
        >> $LOGDIR/eval_hf_$1_$2_s$3.log 2>&1
  fi
}

# ---------------------------------------------------------- 1. BERT LoRA
# Two lanes (one per task), seeds in priority order inside each lane.
bert_lane() {
  local TASK=$1
  for SEED in 9550 9551 9552; do
    echo "[stage_c] bert_lora/$TASK/s$SEED $(date -Is)"
    OMP_NUM_THREADS=8 $PY -m src.oc_completion.train_hf \
        --arm bert_lora --task $TASK --seed $SEED --resume --threads 8 \
        >> $LOGDIR/hf_bert_lora_${TASK}_s${SEED}.log 2>&1
    eval_ckpt bert_lora $TASK $SEED
  done
}
bert_lane ocdet &
bert_lane ocnoisy &
wait
echo "[stage_c] BERT LoRA lanes finished $(date -Is)"

# --------------------------- 2+3. Llama-1B LoRA (3 seeds x 2 tasks) + full FT
# Sequential on the single A100. Priority: seed 9550 on both tasks for every
# arm first, then the extra LoRA seeds.
for SEED in 9550; do
  for ARM in llama_lora llama_full; do
    for TASK in ocdet ocnoisy; do
      echo "[stage_c] $ARM/$TASK/s$SEED $(date -Is)"
      OMP_NUM_THREADS=8 $PY -m src.oc_completion.train_hf \
          --arm $ARM --task $TASK --seed $SEED --resume --threads 8 \
          >> $LOGDIR/hf_${ARM}_${TASK}_s${SEED}.log 2>&1
      eval_ckpt $ARM $TASK $SEED
    done
  done
done
for SEED in 9551 9552; do
  for TASK in ocdet ocnoisy; do
    echo "[stage_c] llama_lora/$TASK/s$SEED $(date -Is)"
    OMP_NUM_THREADS=8 $PY -m src.oc_completion.train_hf \
        --arm llama_lora --task $TASK --seed $SEED --resume --threads 8 \
        >> $LOGDIR/hf_llama_lora_${TASK}_s${SEED}.log 2>&1
    eval_ckpt llama_lora $TASK $SEED
  done
done

# best-effort: extra full-FT seeds ("preferably three" in the protocol)
for SEED in 9551 9552; do
  for TASK in ocdet ocnoisy; do
    echo "[stage_c] llama_full/$TASK/s$SEED (extension) $(date -Is)"
    OMP_NUM_THREADS=8 $PY -m src.oc_completion.train_hf \
        --arm llama_full --task $TASK --seed $SEED --resume --threads 8 \
        >> $LOGDIR/hf_llama_full_${TASK}_s${SEED}.log 2>&1
    eval_ckpt llama_full $TASK $SEED
  done
done

echo "[stage_c] ALL DONE $(date -Is)"
