#!/usr/bin/env bash
# Run directly on the login node — no GPU needed, no SLURM submission.
# Usage: bash experiments/rq2_trace_comparison/slurm/rq2_eval.slurm
#
# Do NOT sbatch this script: steps 4-6 are pure CPU (sklearn, matplotlib,
# JSON report) and a GPU allocation with 0% utilisation will be auto-killed.

set -euo pipefail

RUN_ID="${RUN_ID:-RQ2-EVAL-${SLURM_JOB_ID:-manual}}"
ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"

RQ2_DIR="${ALVIS_REPO}/experiments/rq2_trace_comparison"
DATA_DIR="${RQ2_DIR}/data"
EMB_DIR="${RQ2_DIR}/embeddings"
METRICS_DIR="${RQ2_DIR}/metrics"
PLOTS_DIR="${RQ2_DIR}/plots"
REPORTS_DIR="${RQ2_DIR}/reports"

mkdir -p "${ALVIS_REPO}/logs/rq2" "${METRICS_DIR}" "${PLOTS_DIR}" "${REPORTS_DIR}"

echo "=== RQ2 Evaluation (metrics + viz + report) ==="
echo "  RUN_ID=${RUN_ID}"
echo "  ALVIS_REPO=${ALVIS_REPO}"

echo "[0/3] Installing dependencies..."
apptainer exec --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet scikit-learn matplotlib 2>&1 | tail -3

# ---------------------------------------------------------------------------
# Step 4: Compute trace metrics
# ---------------------------------------------------------------------------
echo "[1/3] Computing trace metrics..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ2_DIR}/scripts/04_compute_trace_metrics.py" \
    --data_dir "${DATA_DIR}" \
    --embeddings_dir "${EMB_DIR}" \
    --output_dir "${METRICS_DIR}"

# ---------------------------------------------------------------------------
# Step 5: Visualize traces
# ---------------------------------------------------------------------------
echo "[2/3] Generating visualizations..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ2_DIR}/scripts/05_visualize_traces.py" \
    --data_dir "${DATA_DIR}" \
    --embeddings_dir "${EMB_DIR}" \
    --output_dir "${PLOTS_DIR}"

# ---------------------------------------------------------------------------
# Step 6: Build report
# ---------------------------------------------------------------------------
echo "[3/3] Building report..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ2_DIR}/scripts/06_build_report.py" \
    --metrics_dir "${METRICS_DIR}" \
    --plots_dir "${PLOTS_DIR}" \
    --output_dir "${REPORTS_DIR}"

echo "=== RQ2 eval done. RUN_ID=${RUN_ID} ==="
echo "  Report:  ${REPORTS_DIR}/report.md"
echo "  Metrics: ${METRICS_DIR}/"
echo "  Plots:   ${PLOTS_DIR}/"
