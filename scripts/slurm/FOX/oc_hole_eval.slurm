#!/bin/bash -l
# Ordered Compliance evaluation-only array on FOX.
#
# Required default array 0-115:
#   0-35    selected scratch checkpoints (3 models x 4 pi x 3 seeds)
#   36-39   selected HF checkpoints (BERT/Llama at pi 0.0,0.3; one seed)
#   40-79   baselines (10 families x 4 pi)
#   80-115  random untrained scratch controls (3 models x 4 pi x 3 seeds)
#
# Main test evaluation and post-training epoch trajectories are performed here,
# never by a training job.  Baseline jobs use the allocated GPU node as a CPU
# host so a single dependency-preserving array can cover every scorer.
#
#   mkdir -p logs/oc_hole
#   sbatch --dependency=afterok:<dl_job>:<hf_job>:<baseline_job> \
#       scripts/slurm/FOX/oc_hole_eval.slurm

#SBATCH --job-name=oc_hole_eval
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-115%4
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_hole/eval_%A_%a.out

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
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
DATA_DIR="$(realpath -e -- "$DATA_DIR")"
: "${HF_HOME:?Set HF_HOME to persistent project storage before submission}"
: "${HF_HUB_CACHE:?Set HF_HUB_CACHE to persistent project storage before submission}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
HF_HOME="$(realpath -e -- "$HF_HOME")"
HF_HUB_CACHE="$(realpath -e -- "$HF_HUB_CACHE")"
OC_RESULT_ROOT="${OC_RESULT_ROOT:-$REPO_ROOT/results/oc_hole_audit}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cluster/work/projects/ec12/michechi/checkpoints}"
OC_CHECKPOINT_ROOT="${OC_CHECKPOINT_ROOT:-$CHECKPOINT_DIR/oc_hole_audit}"
OC_HF_MODEL_SEED="${OC_HF_MODEL_SEED:-9950}"
OC_DATA_ROOT="$DATA_DIR/simulation/oc_hole_audit"
HOLE_MANIFEST="$OC_RESULT_ROOT/hole_manifest.parquet"
HOLE_ORACLE="$OC_RESULT_ROOT/hole_oracle_labels.npz"
STRICT_MANIFEST="$OC_RESULT_ROOT/strict_pair_manifest.parquet"

for required in \
    "$REPO_ROOT/repro/src/oc_completion/ordered_eval.py" \
    "$HOLE_MANIFEST" \
    "$HOLE_ORACLE" \
    "$STRICT_MANIFEST"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: missing evaluation input: $required" >&2
        exit 2
    fi
done

mkdir -p "$OC_RESULT_ROOT/job_markers/evaluation"
export REPO_ROOT DATA_DIR HF_HOME HF_HUB_CACHE CHECKPOINT_DIR OC_CHECKPOINT_ROOT
export OC_HF_MODEL_SEED
export LLMSEQ_ROOT="$REPO_ROOT"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

MODELS=(LSTM Transformer RNNTransformer)
PIS=(0.0 0.1 0.2 0.3)
SEEDS=(9550 9551 9552)
BASELINES=(
    letter_count_logreg
    lag_pair_logreg
    chain_occupancy_logreg
    chain_occupancy_xgb
    position_lag_pair_logreg
    lag_trigram_logreg
    occupancy_max_count
    occupancy_n_chains_ge2
    occupancy_max_run
    oracle
)
HF_REQUIRED=(
    "bert_lora 0.0" "bert_lora 0.3"
    "llama_lora 0.0" "llama_lora 0.3"
)

INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( INDEX < 0 || INDEX > 115 )); then
    echo "ERROR: evaluation array index must be in 0-115: $INDEX" >&2
    exit 2
fi

MODEL_KIND=""
MODEL=""
PI=""
SEED="9550"
CHECKPOINT=""
RUN_DIR=""
NEEDS_TRAJECTORY=0

if (( INDEX < 36 )); then
    LOCAL=$INDEX
    MODEL="${MODELS[$((LOCAL / 12))]}"
    PI="${PIS[$(((LOCAL % 12) / 3))]}"
    SEED="${SEEDS[$((LOCAL % 3))]}"
    MODEL_KIND="scratch"
    PI_TAG="${PI/./p}"
    RUN_DIR="$OC_CHECKPOINT_ROOT/${MODEL,,}/pi_$PI_TAG/seed_$SEED"
    CHECKPOINT="$RUN_DIR/best_standard.pt"
    NEEDS_TRAJECTORY=1
elif (( INDEX < 40 )); then
    read -r MODEL PI <<< "${HF_REQUIRED[$((INDEX - 36))]}"
    SEED="$OC_HF_MODEL_SEED"
    MODEL_KIND="hf"
    PI_TAG="${PI/./p}"
    RUN_DIR="$OC_CHECKPOINT_ROOT/$MODEL/pi_$PI_TAG/seed_$SEED"
    CHECKPOINT="$RUN_DIR/best_standard.pt"
    NEEDS_TRAJECTORY=1
elif (( INDEX < 80 )); then
    LOCAL=$((INDEX - 40))
    PI="${PIS[$((LOCAL / 10))]}"
    MODEL="${BASELINES[$((LOCAL % 10))]}"
    MODEL_KIND="baseline"
    PI_TAG="${PI/./p}"
    RUN_DIR="$OC_CHECKPOINT_ROOT/baselines/pi_$PI_TAG/$MODEL"
    CHECKPOINT="$RUN_DIR/artifact.joblib"
else
    LOCAL=$((INDEX - 80))
    MODEL="${MODELS[$((LOCAL / 12))]}"
    PI="${PIS[$(((LOCAL % 12) / 3))]}"
    SEED="${SEEDS[$((LOCAL % 3))]}"
    MODEL_KIND="random_scratch"
fi

PI_TAG="${PI/./p}"
RUN_KEY="${MODEL_KIND}_${MODEL}_pi_${PI_TAG}_seed_${SEED}"
MARKER="$OC_RESULT_ROOT/job_markers/evaluation/${RUN_KEY}.done.json"

valid_marker() {
    python3 - "$MARKER" "$MODEL_KIND" "$MODEL" "$PI" "$SEED" <<'PY'
import json
import pathlib
import sys

try:
    marker = json.loads(pathlib.Path(sys.argv[1]).read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
expected = {
    "status": "complete",
    "model_kind": sys.argv[2],
    "model": sys.argv[3],
    "noise_pi": float(sys.argv[4]),
    "model_seed": int(sys.argv[5]),
}
raise SystemExit(0 if all(marker.get(k) == v for k, v in expected.items()) else 1)
PY
}

echo "[oc_hole_eval] index=$INDEX kind=$MODEL_KIND model=$MODEL pi=$PI seed=$SEED"
if valid_marker; then
    echo "[oc_hole_eval] valid completion marker exists; skipping"
    exit 0
fi
if [[ -n "$CHECKPOINT" && ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: required selected checkpoint is missing: $CHECKPOINT" >&2
    exit 2
fi
if [[ "$MODEL_KIND" == hf ]]; then
    EVAL_BATCH="${OC_HF_EVAL_BATCH_SIZE:-256}"
elif [[ "$MODEL_KIND" == baseline ]]; then
    EVAL_BATCH="${OC_BASELINE_EVAL_BATCH_SIZE:-520}"
else
    EVAL_BATCH="${OC_SCRATCH_EVAL_BATCH_SIZE:-520}"
fi

cd "$REPO_ROOT/repro"
EVAL_ARGS=(
    --model-kind "$MODEL_KIND"
    --model "$MODEL"
    --pi "$PI"
    --seed "$SEED"
    --device auto
    --batch-size "$EVAL_BATCH"
    --hole-split test
    --pair-split test
    --backgrounds clean,heldout
    --hole-manifest "$HOLE_MANIFEST"
    --hole-oracle "$HOLE_ORACLE"
    --strict-manifest "$STRICT_MANIFEST"
    --result-root "$OC_RESULT_ROOT"
    --chunk-bases "${OC_EVAL_CHUNK_BASES:-100}"
    --bootstrap-iters "${OC_BOOTSTRAP_ITERS:-1000}"
)
if [[ -n "$CHECKPOINT" ]]; then
    EVAL_ARGS+=(--checkpoint "$CHECKPOINT")
fi
python3 -u -m src.oc_completion.ordered_eval "${EVAL_ARGS[@]}"

if (( NEEDS_TRAJECTORY == 1 )); then
    python3 -u -m src.oc_completion.ordered_epoch_diagnostics \
        --model-kind "$MODEL_KIND" \
        --run-dir "$RUN_DIR" \
        --hole-manifest "$HOLE_MANIFEST" \
        --hole-oracle "$HOLE_ORACLE" \
        --strict-manifest "$STRICT_MANIFEST" \
        --device auto \
        --batch-size "$EVAL_BATCH" \
        --diagnostic-bases 500 \
        --strict-pairs 2000 \
        --bootstrap-iters "${OC_DIAGNOSTIC_BOOTSTRAP_ITERS:-200}"
fi

python3 - "$MARKER" "$MODEL_KIND" "$MODEL" "$PI" "$SEED" "$INDEX" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "status": "complete",
    "model_kind": sys.argv[2],
    "model": sys.argv[3],
    "noise_pi": float(sys.argv[4]),
    "model_seed": int(sys.argv[5]),
    "array_index": int(sys.argv[6]),
}
temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
echo "[oc_hole_eval] complete: $RUN_KEY"
