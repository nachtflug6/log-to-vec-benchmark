#!/usr/bin/env bash
#SBATCH --job-name=rq7_real_data
#SBATCH --account=naiss2026-4-91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq7/%x_%j.out
#SBATCH --error=logs/rq7/%x_%j.err

# Full RQ7 pipeline: prepare → FFT → MOMENT → metrics → report
# Runs swat_binary and msl_binary unconditionally; skips tep_* if raw data absent.
#
# Submit from repo root on Alvis:
#   mkdir -p logs/rq7
#   ALVIS_REPO=$PWD sbatch experiments/rq7_real_data/slurm/rq7_full.slurm

set -euo pipefail

ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ7_DIR="${ALVIS_REPO}/experiments/rq7_real_data"

echo "=== RQ7 Real-World Dataset Evaluation ==="
echo "  ALVIS_REPO=${ALVIS_REPO}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

echo ""
echo "[0] Installing dependencies..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet momentfm 2>&1 | tail -3

echo ""
echo "[1] Preparing data (swat_binary + msl_binary)..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ7_DIR}/scripts/01_prepare_data.py" \
    --scenario swat_binary msl_binary \
    --output_dir "${RQ7_DIR}/results/data"

# TEP: only attempt if raw data has been downloaded
if [ -d "${RQ7_DIR}/raw/tep" ] && ls "${RQ7_DIR}/raw/tep/"*.csv &>/dev/null; then
  echo ""
  echo "[1b] Preparing TEP data..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python -u "${RQ7_DIR}/scripts/01_prepare_data.py" \
      --scenario tep_modes tep_faults \
      --output_dir "${RQ7_DIR}/results/data"
else
  echo "[1b] TEP raw data not found at ${RQ7_DIR}/raw/tep — skipping TEP scenarios."
fi

echo ""
echo "[2] Extracting FFT embeddings..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ7_DIR}/scripts/02_extract_fft.py" \
    --data_dir   "${RQ7_DIR}/results/data" \
    --output_dir "${RQ7_DIR}/results/embeddings"

echo ""
echo "[3] Extracting MOMENT embeddings..."
apptainer exec --nv --pwd "${ALVIS_REPO}" \
  --env TRANSFORMERS_OFFLINE=1 \
  "${ALVIS_IMAGE}" \
  python -u "${RQ7_DIR}/scripts/03_extract_moment.py" \
    --data_dir   "${RQ7_DIR}/results/data" \
    --output_dir "${RQ7_DIR}/results/embeddings" \
    --device     cuda \
    --batch_size 64

echo ""
echo "[4] Computing metrics..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ7_DIR}/scripts/04_compute_metrics.py" \
    --embeddings_dir "${RQ7_DIR}/results/embeddings" \
    --output_dir     "${RQ7_DIR}/results/metrics"

echo ""
echo "[5] Building report..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ7_DIR}/scripts/05_build_report.py" \
    --metrics_dir "${RQ7_DIR}/results/metrics" \
    --output_dir  "${RQ7_DIR}/results/reports"

echo ""
echo "=== RQ7 complete ==="
echo "Report: ${RQ7_DIR}/results/reports/report.md"
