#!/usr/bin/env bash
# Run directly on the login node — no GPU needed, no SLURM submission.
# Usage: bash experiments/rq2_trace_comparison/slurm/rq2_fft.slurm
#
# Do NOT sbatch this script: FFT is pure CPU work and a GPU allocation
# with 0% utilisation will be auto-killed.

set -euo pipefail

RUN_ID="${RUN_ID:-RQ2-FFT-${SLURM_JOB_ID:-manual}}"
ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"

RQ2_DIR="${ALVIS_REPO}/experiments/rq2_trace_comparison"
DATA_DIR="${RQ2_DIR}/data"
EMB_DIR="${RQ2_DIR}/embeddings"

SEED="${SEED:-42}"

mkdir -p "${ALVIS_REPO}/logs/rq2" "${DATA_DIR}" "${EMB_DIR}"

echo "=== RQ2 FFT embeddings ==="
echo "  RUN_ID=${RUN_ID}"
echo "  ALVIS_REPO=${ALVIS_REPO}"

# ---------------------------------------------------------------------------
# Step 1: Generate datasets (skip if already done)
# ---------------------------------------------------------------------------
if [[ -f "${DATA_DIR}/p3_hard_noisy/windows.npz" ]]; then
  echo "[1/2] Datasets already exist, skipping generation."
else
  echo "[1/2] Generating periodic-mode datasets..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python "${RQ2_DIR}/scripts/01_generate_periodic_modes.py" \
      --output_dir "${DATA_DIR}" \
      --num_trajectories 20 \
      --periods_per_segment 5 \
      --seed "${SEED}"
fi

# ---------------------------------------------------------------------------
# Step 2: FFT embeddings (skip if already done)
# ---------------------------------------------------------------------------
if [[ -f "${EMB_DIR}/p3_hard_noisy_fft.npz" ]]; then
  echo "[2/2] FFT embeddings already exist, skipping."
else
  echo "[2/2] Extracting FFT embeddings..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python "${RQ2_DIR}/scripts/02_extract_embeddings.py" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${EMB_DIR}" \
      --model fft
fi

echo "=== RQ2 FFT done. RUN_ID=${RUN_ID} ==="
