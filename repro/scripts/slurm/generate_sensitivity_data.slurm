#!/bin/bash
#SBATCH --job-name=gen_sensitivity
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --mem=90G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/gen_sensitivity_%j.out
#SBATCH --error=logs/gen_sensitivity_%j.err

set -o errexit
set -o nounset

echo "=== Generating sensitivity datasets ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster

if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}
OUTPUT_DIR="${REPRO_ROOT:-$PWD}/data/simulation/sensitivity/"

srun apptainer exec \
    --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp} \
    --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD} \
    "$CONTAINER" \
    python3 -u -m simulation.generate_sensitivity_datasets \
        --only 120,140    \
        --output_dir "$OUTPUT_DIR"

echo "=== DONE ==="
