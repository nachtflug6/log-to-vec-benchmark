#!/usr/bin/env bash
#SBATCH --job-name=rq5_moment
#SBATCH --account=NAISS2024-5-XXX
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/rq5_moment_%j.out
#SBATCH --error=logs/rq5_moment_%j.err

set -euo pipefail

ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ5_DIR="${ALVIS_REPO}/experiments/rq5_window_sweep"

echo "=== RQ5 MOMENT Embedding (window sweep) ==="
echo "  ALVIS_REPO=${ALVIS_REPO}"

echo "[0] Installing dependencies..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet momentfm 2>&1 | tail -3

echo "[1] Generating datasets (all window sizes)..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ5_DIR}/scripts/01_generate_sweep.py" \
    --output_dir "${RQ5_DIR}/results/data"

echo "[2] Extracting MOMENT embeddings..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ5_DIR}/scripts/03_extract_moment.py" \
    --data_dir    "${RQ5_DIR}/results/data" \
    --output_dir  "${RQ5_DIR}/results/embeddings" \
    --batch_size  64

echo "=== RQ5 MOMENT done ==="
