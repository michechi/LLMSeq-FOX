#!/bin/bash

#SBATCH --job-name=cf_pairs
#SBATCH --account=ec12
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=a100:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=16

# ──────────────────────────────────────────────────────────────────────
# Score counterfactual pairs — trains from scratch, then scores 47K pairs
#
# Usage (submit one per model):
#   sbatch scripts/slurm/FOX/score_counterfactual.slurm DL
#   sbatch scripts/slurm/FOX/score_counterfactual.slurm BERT
#   sbatch scripts/slurm/FOX/score_counterfactual.slurm Llama1B
#   sbatch scripts/slurm/FOX/score_counterfactual.slurm Llama8B
#   sbatch scripts/slurm/FOX/score_counterfactual.slurm Qwen4B
#   sbatch scripts/slurm/FOX/score_counterfactual.slurm Qwen14B
#
# Note: A100 40GB can handle up to ~8B with LoRA+4bit.
#       14B with 4bit needs ~28GB — tight on 40GB, safe on 80GB.
# ──────────────────────────────────────────────────────────────────────

set -o errexit
set -o nounset

MODE=${1:-DL}
echo "=== Counterfactual scoring — mode: $MODE ==="

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8
module load nlpl-bitsandbytes/0.42.0-foss-2022b-Python-3.10.8

# ============== PATHS ==============
DATA_PATH="data/simulation/tested/"
PAIRS_PATH="paper_tables/counterfactual_pairs_tricky_rnd.csv"
OUTPUT_DIR="paper_tables/"
SEED=9950

# ============== BUILD COMMAND ==============
BASE_CMD="python3 -u src/experiments/score_counterfactual_pairs.py \
    --number_to_use 9 \
    --path_csv $DATA_PATH \
    --pairs_path $PAIRS_PATH \
    --output_dir $OUTPUT_DIR \
    --seed $SEED"

case "$MODE" in
    DL)
        echo "LSTM, Transformer, RNNTransformer (~100K params each, fast)"
        $BASE_CMD --model_type DL --models LSTM,Transformer,RNNTransformer
        ;;
    BERT)
        echo "BERT-base (~110M params, ~30min on A100)"
        $BASE_CMD --model_type LLM --model_name google-bert/bert-base-uncased
        ;;
    Llama1B)
        echo "Llama-3.2-1B + LoRA (~8GB VRAM)"
        $BASE_CMD --model_type LLM --model_name meta-llama/Llama-3.2-1B --peft
        ;;
    Llama8B)
        echo "Llama-3.1-8B + LoRA + 4bit (~20GB VRAM)"
        $BASE_CMD --model_type LLM --model_name meta-llama/Llama-3.1-8B --peft --use_quantization
        ;;
    Qwen4B)
        echo "Qwen3-4B + LoRA (~14GB VRAM)"
        $BASE_CMD --model_type LLM --model_name Qwen/Qwen3-4B --peft
        ;;
    Qwen14B)
        echo "Qwen2.5-14B + LoRA + 4bit (~28GB VRAM — needs 40GB+ GPU)"
        $BASE_CMD --model_type LLM --model_name Qwen/Qwen2.5-14B --peft --use_quantization
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Options: DL, BERT, Llama1B, Llama8B, Qwen4B, Qwen14B"
        exit 1
        ;;
esac
