#!/usr/bin/env bash
#SBATCH --job-name=log2vec-rq3-moment
#SBATCH --account=naiss2026-4-91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:50:00
#SBATCH --output=logs/rq3/%x_%j.out
#SBATCH --error=logs/rq3/%x_%j.err

set -euo pipefail

ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ3_DIR="${ALVIS_REPO}/experiments/rq3_landscape_sweep"

mkdir -p "${ALVIS_REPO}/logs/rq3" "${RQ3_DIR}/results/embeddings"

echo "=== RQ3 MOMENT extraction ==="
echo "  ALVIS_REPO=${ALVIS_REPO}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "[0] Installing dependencies..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet momentfm scikit-learn matplotlib 2>&1 | tail -3

echo "[1] Extracting MOMENT embeddings for all scenarios..."
apptainer exec --nv --pwd "${ALVIS_REPO}" \
  --env TRANSFORMERS_OFFLINE=1 \
  "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/03_extract_moment.py" \
    --data_dir    "${RQ3_DIR}/results/data" \
    --output_dir  "${RQ3_DIR}/results/embeddings" \
    --device cuda

echo "=== RQ3 MOMENT done ==="
