#!/usr/bin/env bash
# Run on Alvis LOGIN NODE — CPU only (metrics + viz).
# Do NOT sbatch this.
#
# Usage:
#   cd /cephyr/users/silvan/Alvis/log-to-vec-benchmark
#   bash experiments/rq4_cnc_benchmark/slurm/rq4_eval.slurm

set -euo pipefail

ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ4_DIR="${ALVIS_REPO}/experiments/rq4_cnc_benchmark"

echo "=== RQ4 Eval ==="

echo "[0] Installing CPU dependencies..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet scikit-learn matplotlib 2>&1 | tail -3

echo "[1] FFT embeddings (CPU, ~2 min)..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ4_DIR}/scripts/02_extract_fft.py" \
    --data_dir   "${RQ4_DIR}/results/data" \
    --output_dir "${RQ4_DIR}/results/embeddings"

echo "[2] Computing metrics..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ4_DIR}/scripts/04_compute_metrics.py" \
    --data_dir       "${RQ4_DIR}/results/data" \
    --embeddings_dir "${RQ4_DIR}/results/embeddings" \
    --output_dir     "${RQ4_DIR}/results/metrics"

echo "[3] Generating plots..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ4_DIR}/scripts/05_visualize_traces.py" \
    --embeddings_dir "${RQ4_DIR}/results/embeddings" \
    --output_dir     "${RQ4_DIR}/results/plots"

echo "[4] Building report..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ4_DIR}/scripts/06_build_report.py" \
    --metrics_dir "${RQ4_DIR}/results/metrics" \
    --output_dir  "${RQ4_DIR}/results/reports"

echo "=== RQ4 eval done ==="
echo "  Report: ${RQ4_DIR}/results/reports/report.md"
