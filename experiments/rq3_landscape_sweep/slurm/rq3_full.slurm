#!/usr/bin/env bash
#SBATCH --job-name=log2vec-rq3-full
#SBATCH --account=naiss2026-4-91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --output=logs/rq3/%x_%j.out
#SBATCH --error=logs/rq3/%x_%j.err

# Full RQ3 pipeline: generate → FFT → MOMENT → metrics → plots → report.
# DCC training is excluded (separate array job: rq3_dcc.slurm).
#
# Usage:
#   cd /cephyr/users/silvan/Alvis/log-to-vec-benchmark
#   sbatch experiments/rq3_landscape_sweep/slurm/rq3_full.slurm

set -euo pipefail

RUN_ID="${RUN_ID:-RQ3-${SLURM_JOB_ID:-manual}}"
ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ3_DIR="${ALVIS_REPO}/experiments/rq3_landscape_sweep"

mkdir -p "${ALVIS_REPO}/logs/rq3" \
         "${RQ3_DIR}/results/data" \
         "${RQ3_DIR}/results/embeddings" \
         "${RQ3_DIR}/results/metrics" \
         "${RQ3_DIR}/results/plots" \
         "${RQ3_DIR}/results/reports"

echo "=== RQ3 Landscape Sweep — full pipeline ==="
echo "  RUN_ID=${RUN_ID}"
echo "  ALVIS_REPO=${ALVIS_REPO}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "[0] Installing dependencies..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet momentfm scikit-learn matplotlib 2>&1 | tail -3

# ---------------------------------------------------------------------------
# Step 1: Generate datasets (skip if already done)
# ---------------------------------------------------------------------------
if [[ -f "${RQ3_DIR}/results/data/baseline/windows.npz" ]]; then
  echo "[1/6] Datasets already exist, skipping generation."
else
  echo "[1/6] Generating landscape sweep datasets..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python -u "${RQ3_DIR}/scripts/01_generate_scenarios.py" \
      --output_dir "${RQ3_DIR}/results/data"
fi

# ---------------------------------------------------------------------------
# Step 2a: FFT embeddings (skip if already done)
# ---------------------------------------------------------------------------
if [[ -f "${RQ3_DIR}/results/embeddings/baseline_fft8.npz" ]]; then
  echo "[2a/6] FFT embeddings already exist, skipping."
else
  echo "[2a/6] Extracting FFT embeddings..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python -u "${RQ3_DIR}/scripts/02_extract_fft.py" \
      --data_dir   "${RQ3_DIR}/results/data" \
      --output_dir "${RQ3_DIR}/results/embeddings"
fi

# ---------------------------------------------------------------------------
# Step 2b: MOMENT embeddings (GPU)
# ---------------------------------------------------------------------------
if [[ -f "${RQ3_DIR}/results/embeddings/baseline_moment.npz" ]]; then
  echo "[2b/6] MOMENT embeddings already exist, skipping."
else
  echo "[2b/6] Extracting MOMENT embeddings (GPU)..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" \
    --env TRANSFORMERS_OFFLINE=1 \
    "${ALVIS_IMAGE}" \
    python -u "${RQ3_DIR}/scripts/03_extract_moment.py" \
      --data_dir   "${RQ3_DIR}/results/data" \
      --output_dir "${RQ3_DIR}/results/embeddings" \
      --device     cuda
fi

# ---------------------------------------------------------------------------
# Steps 3-5: Metrics, plots, report (CPU — runs on the GPU node after MOMENT)
# ---------------------------------------------------------------------------
echo "[4/6] Computing trace metrics..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/05_compute_metrics.py" \
    --data_dir       "${RQ3_DIR}/results/data" \
    --embeddings_dir "${RQ3_DIR}/results/embeddings" \
    --output_dir     "${RQ3_DIR}/results/metrics"

echo "[5/6] Generating visualisations..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/06_visualize_traces.py" \
    --data_dir       "${RQ3_DIR}/results/data" \
    --embeddings_dir "${RQ3_DIR}/results/embeddings" \
    --output_dir     "${RQ3_DIR}/results/plots"

echo "[6/6] Building report..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/07_build_report.py" \
    --metrics_dir "${RQ3_DIR}/results/metrics" \
    --plots_dir   "${RQ3_DIR}/results/plots" \
    --output_dir  "${RQ3_DIR}/results/reports"

echo "=== RQ3 complete. RUN_ID=${RUN_ID} ==="
echo "  Report: ${RQ3_DIR}/results/reports/report.md"
