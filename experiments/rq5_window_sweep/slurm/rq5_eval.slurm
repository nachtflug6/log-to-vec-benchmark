#!/usr/bin/env bash
# Run on Alvis LOGIN NODE — CPU only (FFT + metrics + plots + report).
# Do NOT sbatch this.
#
# Usage:
#   cd /cephyr/users/silvan/Alvis/log-to-vec-benchmark
#   bash experiments/rq5_window_sweep/slurm/rq5_eval.slurm

set -euo pipefail

ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ5_DIR="${ALVIS_REPO}/experiments/rq5_window_sweep"

echo "=== RQ5 Window-Sweep Eval ==="

echo "[0] Installing CPU dependencies..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet scikit-learn matplotlib 2>&1 | tail -3

echo "[1] Generating datasets (if not already done)..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ5_DIR}/scripts/01_generate_sweep.py" \
    --output_dir "${RQ5_DIR}/results/data"

echo "[2] FFT embeddings (CPU)..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ5_DIR}/scripts/02_extract_fft.py" \
    --data_dir   "${RQ5_DIR}/results/data" \
    --output_dir "${RQ5_DIR}/results/embeddings"

echo "[3] Computing metrics..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ5_DIR}/scripts/04_compute_metrics.py" \
    --data_dir       "${RQ5_DIR}/results/data" \
    --embeddings_dir "${RQ5_DIR}/results/embeddings" \
    --output_dir     "${RQ5_DIR}/results/metrics"

echo "[4] Plotting sweep figures..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ5_DIR}/scripts/05_plot_sweep.py" \
    --metrics_dir "${RQ5_DIR}/results/metrics" \
    --output_dir  "${RQ5_DIR}/results/plots"

echo "[5] Building report..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ5_DIR}/scripts/06_build_report.py" \
    --metrics_dir "${RQ5_DIR}/results/metrics" \
    --output_dir  "${RQ5_DIR}/results/reports"

echo "=== RQ5 eval done ==="
echo "  Report: ${RQ5_DIR}/results/reports/report.md"
echo "  Plots:  ${RQ5_DIR}/results/plots/"
