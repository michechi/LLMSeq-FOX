#!/bin/bash -l
# Ordered Compliance baseline fitting on FOX CPUs.
#
# Array 0-3 maps directly to pi = 0.0, 0.1, 0.2, 0.3.  Each task fits all
# required learned baselines and writes deterministic/oracle artifacts.  Hole
# and strict-pair evaluation remains in oc_hole_eval.slurm.
#
#   mkdir -p logs/oc_hole
#   sbatch scripts/slurm/FOX/oc_hole_baselines.slurm

#SBATCH --job-name=oc_hole_base
#SBATCH --account=ec12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_hole/baselines_%A_%a.out

set -o errexit
set -o nounset
set -o pipefail

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
DATA_DIR="$(realpath -e -- "$DATA_DIR")"
HF_HOME="${HF_HOME:-$DATA_DIR/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
OC_RESULT_ROOT="${OC_RESULT_ROOT:-$REPO_ROOT/results/oc_hole_audit}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cluster/work/projects/ec12/michechi/checkpoints}"
OC_CHECKPOINT_ROOT="${OC_CHECKPOINT_ROOT:-$CHECKPOINT_DIR/oc_hole_audit}"
OC_DATA_ROOT="$DATA_DIR/simulation/oc_hole_audit"

if [[ ! -f "$REPO_ROOT/repro/src/oc_completion/ordered_baselines.py" ]]; then
    echo "ERROR: invalid REPO_ROOT or missing baseline runner: $REPO_ROOT" >&2
    exit 2
fi
for required in \
    "$OC_DATA_ROOT/splits/train.parquet" \
    "$OC_DATA_ROOT/splits/val.parquet" \
    "$OC_DATA_ROOT/splits/test.parquet" \
    "$OC_DATA_ROOT/noise/train.parquet" \
    "$OC_DATA_ROOT/noise/val.parquet" \
    "$OC_DATA_ROOT/noise/test.parquet"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: missing ordered-data input: $required" >&2
        exit 2
    fi
done

mkdir -p "$OC_RESULT_ROOT" "$OC_CHECKPOINT_ROOT"
export REPO_ROOT DATA_DIR HF_HOME HF_HUB_CACHE CHECKPOINT_DIR OC_CHECKPOINT_ROOT
export LLMSEQ_ROOT="$REPO_ROOT"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

PIS=(0.0 0.1 0.2 0.3)
FAMILIES=(
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
INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( INDEX < 0 || INDEX >= 4 )); then
    echo "ERROR: baseline array index must be in 0-3: $INDEX" >&2
    exit 2
fi
PI="${PIS[$INDEX]}"
PI_TAG="${PI/./p}"

all_valid() {
    python3 - "$OC_CHECKPOINT_ROOT" "$PI_TAG" "$PI" "${FAMILIES[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
tag, pi = sys.argv[2], float(sys.argv[3])
families = sys.argv[4:]
for family in families:
    run = root / "baselines" / f"pi_{tag}" / family
    try:
        done = json.loads((run / "done.json").read_text())
    except (OSError, json.JSONDecodeError):
        raise SystemExit(1)
    expected = {"status": "complete", "family": family, "noise_pi": pi, "seed": 9550}
    if not all(done.get(key) == value for key, value in expected.items()):
        raise SystemExit(1)
    if not (run / "artifact.joblib").is_file() or not (run / "result.json").is_file():
        raise SystemExit(1)
raise SystemExit(0)
PY
}

echo "[oc_hole_baselines] index=$INDEX pi=$PI node=$(hostname)"
if all_valid; then
    echo "[oc_hole_baselines] every baseline has a valid done.json; skipping"
    exit 0
fi

cd "$REPO_ROOT/repro"
python3 -u -m src.oc_completion.ordered_baselines \
    --pi "$PI" \
    --families all \
    --seed 9550 \
    --threads "$SLURM_CPUS_PER_TASK" \
    --data-root "$OC_DATA_ROOT" \
    --checkpoint-root "$OC_CHECKPOINT_ROOT" \
    --result-root "$OC_RESULT_ROOT"

if ! all_valid; then
    echo "ERROR: one or more baseline artifacts are incomplete for pi=$PI" >&2
    exit 1
fi
echo "[oc_hole_baselines] complete: pi=$PI"
