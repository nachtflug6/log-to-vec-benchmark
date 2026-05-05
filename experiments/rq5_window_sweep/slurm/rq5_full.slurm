#!/usr/bin/env bash
#SBATCH --job-name=log2vec-rq5-full
#SBATCH --account=naiss2026-4-91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/rq5/%x_%j.out
#SBATCH --error=logs/rq5/%x_%j.err

# Full RQ5 pipeline: generate → FFT → MOMENT → metrics → sweep plots → report.
#
# Usage:
#   cd /cephyr/users/silvan/Alvis/log-to-vec-benchmark
#   sbatch experiments/rq5_window_sweep/slurm/rq5_full.slurm

set -euo pipefail

RUN_ID="${RUN_ID:-RQ5-${SLURM_JOB_ID:-manual}}"
ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ5_DIR="${ALVIS_REPO}/experiments/rq5_window_sweep"

mkdir -p "${ALVIS_REPO}/logs/rq5" \
         "${RQ5_DIR}/results/data" \
         "${RQ5_DIR}/results/embeddings" \
         "${RQ5_DIR}/results/metrics" \
         "${RQ5_DIR}/results/plots" \
         "${RQ5_DIR}/results/reports"

echo "=== RQ5 Window Sweep — full pipeline ==="
echo "  RUN_ID=${RUN_ID}"
echo "  ALVIS_REPO=${ALVIS_REPO}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "[0] Installing dependencies..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet momentfm scikit-learn matplotlib 2>&1 | tail -3

# ---------------------------------------------------------------------------
# Step 1: Generate datasets (skip if already done)
# ---------------------------------------------------------------------------
if [[ -f "${RQ5_DIR}/results/data/cnc_ws_512/windows.npz" ]]; then
  echo "[1/6] Datasets already exist, skipping generation."
else
  echo "[1/6] Generating window-size variants..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python -u "${RQ5_DIR}/scripts/01_generate_sweep.py" \
      --output_dir "${RQ5_DIR}/results/data"
fi

# ---------------------------------------------------------------------------
# Step 2a: FFT embeddings (skip if already done)
# ---------------------------------------------------------------------------
if [[ -f "${RQ5_DIR}/results/embeddings/cnc_ws_512_fft.npz" ]]; then
  echo "[2a/6] FFT embeddings already exist, skipping."
else
  echo "[2a/6] Extracting FFT embeddings..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python -u "${RQ5_DIR}/scripts/02_extract_fft.py" \
      --data_dir   "${RQ5_DIR}/results/data" \
      --output_dir "${RQ5_DIR}/results/embeddings"
fi

# ---------------------------------------------------------------------------
# Step 2b: MOMENT embeddings (GPU)
# ---------------------------------------------------------------------------
if [[ -f "${RQ5_DIR}/results/embeddings/cnc_ws_512_moment.npz" ]]; then
  echo "[2b/6] MOMENT embeddings already exist, skipping."
else
  echo "[2b/6] Extracting MOMENT embeddings (GPU)..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" \
    --env TRANSFORMERS_OFFLINE=1 \
    "${ALVIS_IMAGE}" \
    python -u "${RQ5_DIR}/scripts/03_extract_moment.py" \
      --data_dir   "${RQ5_DIR}/results/data" \
      --output_dir "${RQ5_DIR}/results/embeddings" \
      --device     cuda \
      --batch_size 64
fi

# ---------------------------------------------------------------------------
# Steps 3-5: Metrics, sweep plots, report
# ---------------------------------------------------------------------------
echo "[4/6] Computing trace metrics..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ5_DIR}/scripts/04_compute_metrics.py" \
    --data_dir       "${RQ5_DIR}/results/data" \
    --embeddings_dir "${RQ5_DIR}/results/embeddings" \
    --output_dir     "${RQ5_DIR}/results/metrics"

echo "[5/6] Generating sweep plots..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ5_DIR}/scripts/05_plot_sweep.py" \
    --metrics_dir "${RQ5_DIR}/results/metrics" \
    --output_dir  "${RQ5_DIR}/results/plots"

echo "[6/6] Building report..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ5_DIR}/scripts/06_build_report.py" \
    --metrics_dir "${RQ5_DIR}/results/metrics" \
    --output_dir  "${RQ5_DIR}/results/reports"

echo "=== RQ5 complete. RUN_ID=${RUN_ID} ==="
echo "  Report: ${RQ5_DIR}/results/reports/report.md"
echo "  Plots:  ${RQ5_DIR}/results/plots/"
