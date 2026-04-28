#!/bin/bash
#SBATCH --job-name=log2vec-smoke
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%j_output.txt
#SBATCH --error=logs/%j_error.txt

set -euo pipefail

module load apptainer || true
mkdir -p logs

RUN_ID="${RUN_ID:-EXP-$(date +%Y%m%d-%H%M%S)}"

apptainer exec --nv log-to-vec.sif \
  python examples/train_toy_example.py --config configs/toy_example.yaml

echo "Completed RUN_ID=${RUN_ID}"
