#!/usr/bin/env bash
#SBATCH --job-name=log2vec-rq6-full
#SBATCH --account=naiss2026-4-91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/rq6/%x_%j.out
#SBATCH --error=logs/rq6/%x_%j.err

# Full RQ6 pipeline: generate → FFT → MOMENT → metrics → report.
# 8 scenarios × 3 models = 24 metric cells.
#
# Usage:
#   cd /cephyr/users/silvan/Alvis/log-to-vec-benchmark
#   sbatch experiments/rq6_signal_ablation/slurm/rq6_full.slurm

set -euo pipefail

RUN_ID="${RUN_ID:-RQ6-${SLURM_JOB_ID:-manual}}"
ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ6_DIR="${ALVIS_REPO}/experiments/rq6_signal_ablation"

mkdir -p "${ALVIS_REPO}/logs/rq6" \
         "${RQ6_DIR}/results/data" \
         "${RQ6_DIR}/results/embeddings" \
         "${RQ6_DIR}/results/metrics" \
         "${RQ6_DIR}/results/reports"

echo "=== RQ6 Signal Ablation — full pipeline ==="
echo "  RUN_ID=${RUN_ID}"
echo "  ALVIS_REPO=${ALVIS_REPO}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "[0] Installing dependencies..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet momentfm scikit-learn matplotlib 2>&1 | tail -3

# ---------------------------------------------------------------------------
# Step 1: Generate datasets (skip if already done)
# ---------------------------------------------------------------------------
if [[ -f "${RQ6_DIR}/results/data/ratio_0step/windows.npz" ]]; then
  echo "[1/5] Datasets already exist, skipping generation."
else
  echo "[1/5] Generating RQ6 datasets..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python -u "${RQ6_DIR}/scripts/01_generate_scenarios.py" \
      --output_dir "${RQ6_DIR}/results/data"
fi

# ---------------------------------------------------------------------------
# Step 2: FFT embeddings
# ---------------------------------------------------------------------------
if [[ -f "${RQ6_DIR}/results/embeddings/ratio_0step_fft8.npz" ]]; then
  echo "[2/5] FFT embeddings already exist, skipping."
else
  echo "[2/5] Extracting FFT embeddings..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python -u "${RQ6_DIR}/scripts/02_extract_fft.py" \
      --data_dir   "${RQ6_DIR}/results/data" \
      --output_dir "${RQ6_DIR}/results/embeddings"
fi

# ---------------------------------------------------------------------------
# Step 3: MOMENT embeddings (GPU)
# ---------------------------------------------------------------------------
if [[ -f "${RQ6_DIR}/results/embeddings/ratio_0step_moment.npz" ]]; then
  echo "[3/5] MOMENT embeddings already exist, skipping."
else
  echo "[3/5] Extracting MOMENT embeddings (GPU)..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" \
    --env TRANSFORMERS_OFFLINE=1 \
    "${ALVIS_IMAGE}" \
    python -u "${RQ6_DIR}/scripts/03_extract_moment.py" \
      --data_dir   "${RQ6_DIR}/results/data" \
      --output_dir "${RQ6_DIR}/results/embeddings" \
      --device     cuda
fi

# ---------------------------------------------------------------------------
# Steps 4-5: Metrics + report (CPU — runs on the GPU node after MOMENT)
# ---------------------------------------------------------------------------
echo "[4/5] Computing trace metrics..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ6_DIR}/scripts/04_compute_metrics.py" \
    --data_dir       "${RQ6_DIR}/results/data" \
    --embeddings_dir "${RQ6_DIR}/results/embeddings" \
    --output_dir     "${RQ6_DIR}/results/metrics"

echo "[5/5] Building report..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ6_DIR}/scripts/05_build_report.py" \
    --metrics_dir "${RQ6_DIR}/results/metrics" \
    --output_dir  "${RQ6_DIR}/results/reports"

echo "=== RQ6 complete. RUN_ID=${RUN_ID} ==="
echo "  Report: ${RQ6_DIR}/results/reports/report.md"
