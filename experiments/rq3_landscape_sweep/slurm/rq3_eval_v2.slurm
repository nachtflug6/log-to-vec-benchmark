#!/usr/bin/env bash
# Run directly on the Alvis login node — no GPU needed, do NOT sbatch this.
# Adds MiniRocket and Distribution-per-Bucket (DBF) baselines to existing RQ3
# embeddings, computes metrics, and regenerates the report.
#
# Requires: RQ3 data + existing fft/moment embeddings already in results/
#
# Usage:
#   cd /cephyr/users/silvan/Alvis/log-to-vec-benchmark
#   bash experiments/rq3_landscape_sweep/slurm/rq3_eval_v2.slurm

set -euo pipefail

ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ3_DIR="${ALVIS_REPO}/experiments/rq3_landscape_sweep"

echo "=== RQ3 Eval v2 (Rocket + DBF baselines) ==="
echo "  ALVIS_REPO=${ALVIS_REPO}"

echo "[0] Installing CPU dependencies (scikit-learn, matplotlib)..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet scikit-learn matplotlib 2>&1 | tail -3

echo "[1/4] Extracting MiniRocket embeddings..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/02b_extract_rocket.py" \
    --data_dir   "${RQ3_DIR}/results/data" \
    --output_dir "${RQ3_DIR}/results/embeddings" \
    --pca_dim 128

echo "[2/4] Extracting Distribution-per-Bucket (DBF) embeddings..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/02c_extract_dbf.py" \
    --data_dir   "${RQ3_DIR}/results/data" \
    --output_dir "${RQ3_DIR}/results/embeddings" \
    --num_buckets 8

echo "[3/4] Computing trace metrics for new models..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/05_compute_metrics.py" \
    --data_dir       "${RQ3_DIR}/results/data" \
    --embeddings_dir "${RQ3_DIR}/results/embeddings" \
    --output_dir     "${RQ3_DIR}/results/metrics" \
    --models rocket dbf

echo "[4/4] Regenerating report (all models)..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/07_build_report.py" \
    --metrics_dir "${RQ3_DIR}/results/metrics" \
    --plots_dir   "${RQ3_DIR}/results/plots" \
    --output_dir  "${RQ3_DIR}/results/reports"

echo "=== RQ3 eval v2 done ==="
echo "  Report: ${RQ3_DIR}/results/reports/report.md"
