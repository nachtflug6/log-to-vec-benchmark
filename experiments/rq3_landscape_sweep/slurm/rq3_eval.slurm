#!/usr/bin/env bash
# Run directly on the Alvis login node — no GPU needed, do NOT sbatch this.
# CPU-only work: sklearn metrics, matplotlib plots, JSON report.
# A GPU allocation with 0% utilisation will be auto-killed on Alvis.
#
# Usage:
#   cd /cephyr/users/silvan/Alvis/log-to-vec-benchmark
#   bash experiments/rq3_landscape_sweep/slurm/rq3_eval.slurm

set -euo pipefail

ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ3_DIR="${ALVIS_REPO}/experiments/rq3_landscape_sweep"

echo "=== RQ3 Eval (metrics + viz + report) ==="
echo "  ALVIS_REPO=${ALVIS_REPO}"

echo "[0] Installing CPU dependencies..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet scikit-learn matplotlib 2>&1 | tail -3

echo "[1/3] Computing trace metrics..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/05_compute_metrics.py" \
    --data_dir       "${RQ3_DIR}/results/data" \
    --embeddings_dir "${RQ3_DIR}/results/embeddings" \
    --output_dir     "${RQ3_DIR}/results/metrics"

echo "[2/3] Generating worm plots..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/06_visualize_traces.py" \
    --data_dir       "${RQ3_DIR}/results/data" \
    --embeddings_dir "${RQ3_DIR}/results/embeddings" \
    --output_dir     "${RQ3_DIR}/results/plots"

echo "[3/3] Building report..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/07_build_report.py" \
    --metrics_dir "${RQ3_DIR}/results/metrics" \
    --plots_dir   "${RQ3_DIR}/results/plots" \
    --output_dir  "${RQ3_DIR}/results/reports"

echo "=== RQ3 eval done ==="
echo "  Report: ${RQ3_DIR}/results/reports/report.md"
