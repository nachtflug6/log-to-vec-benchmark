#!/usr/bin/env bash
#SBATCH --job-name=rq4_moment
#SBATCH --account=naiss2026-4-91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/rq4/%x_%j.out
#SBATCH --error=logs/rq4/%x_%j.err

set -euo pipefail

ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ4_DIR="${ALVIS_REPO}/experiments/rq4_cnc_benchmark"

echo "=== RQ4 MOMENT Embedding ==="
echo "  ALVIS_REPO=${ALVIS_REPO}"

echo "[0] Installing dependencies..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet momentfm 2>&1 | tail -3

echo "[1] Generating datasets (if not already done)..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ4_DIR}/scripts/01_generate_scenarios.py" \
    --output_dir "${RQ4_DIR}/results/data"

echo "[2] Extracting MOMENT embeddings..."
apptainer exec --nv --pwd "${ALVIS_REPO}" \
  --env TRANSFORMERS_OFFLINE=1 \
  "${ALVIS_IMAGE}" \
  python -u "${RQ4_DIR}/scripts/03_extract_moment.py" \
    --data_dir   "${RQ4_DIR}/results/data" \
    --output_dir "${RQ4_DIR}/results/embeddings" \
    --device     cuda \
    --batch_size 64

echo "=== MOMENT done ==="
