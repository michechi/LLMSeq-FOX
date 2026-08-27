#!/bin/bash
#SBATCH --job-name=repro_smoke
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=_smoke_test.out

# Smoke test for the anonymized repro/ tree.
# Verifies that imports work and CLI scripts parse inside the apptainer container.

set -o errexit
set -o nounset
set -o pipefail

REPO=${REPRO_ROOT:-$PWD}
CONTAINER=${CONTAINER:?ERROR: export CONTAINER=/path/to/your/pytorch_arm64.sif (any image with Python 3.10+, torch 2.x, transformers, peft, scikit-learn, xgboost works)}

echo "=== smoke_test  job=$SLURM_JOB_ID  node=$(hostname)  $(date) ==="
echo "container exists? $(ls -la $CONTAINER | awk '{print $5}') bytes"

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster || true

cd "$REPO"

echo "=== check Python version ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 --version

echo "=== check src.common imports ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -c "import sys; sys.path.insert(0, '$REPO'); import src.common as c; print('REPRO_ROOT:', c.REPRO_ROOT); print('DATA_DIR:', c.DATA_DIR); print('TESTED_DIR:', c.TESTED_DIR); print('exists:', c.TESTED_DIR.exists())"

echo "=== check XGBoost script CLI parses ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -m src.experiments.XGBoost_fraction_experiment --help 2>&1 | head -25

echo "=== check LLM script CLI parses ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -m src.experiments.LLM_fraction_experiment --help 2>&1 | head -25

echo "=== check BERT script CLI parses ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -m src.experiments.BERT_fraction_experiment --help 2>&1 | head -10

echo "=== check DL_TR_baselines script CLI parses ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -m src.experiments.DL_TR_baselines --help 2>&1 | head -10

echo "=== check mechanism_id common imports ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -c "import sys; sys.path.insert(0, '$REPO'); from src.analysis.mechanism_id.scripts import common; print('mechid REPO_ROOT:', common.REPO_ROOT); print('mechid DATA_DIR:', common.DATA_DIR); print('exists:', common.DATA_DIR.exists())"

echo "=== check plot_main_figure script CLI parses ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -m src.analysis.plot_main_figure --help 2>&1 | head -10

echo "=== check parity_decomposition_dl CLI parses ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -m src.experiments.parity_decomposition_dl --help 2>&1 | head -10

echo "=== check generators import ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -c "import sys; sys.path.insert(0, '$REPO'); from src.generators._internals.do_check_lag import check_lag; print('check_lag OK:', check_lag.__doc__[:60] if check_lag.__doc__ else 'no docstring')"

echo "=== check src.mimic.train_mimic CLI parses ==="
srun apptainer exec --nv \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -m src.mimic.train_mimic --help 2>&1 | head -10

echo "=== check src.mimic.kgram_analysis CLI parses ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -m src.mimic.kgram_analysis --help 2>&1 | head -10

echo "=== check src.mimic.shuffle_test CLI parses ==="
srun apptainer exec \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPRO_ROOT="$REPO" \
    "$CONTAINER" \
    python3 -m src.mimic.shuffle_test --help 2>&1 | head -10

echo "=== smoke_test DONE  $(date) ==="
