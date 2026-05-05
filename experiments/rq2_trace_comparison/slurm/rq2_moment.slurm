#!/usr/bin/env bash
#SBATCH --job-name=log2vec-rq2-moment
#SBATCH --account=naiss2026-4-91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/rq2/%x_%j.out
#SBATCH --error=logs/rq2/%x_%j.err

set -euo pipefail

RUN_ID="${RUN_ID:-RQ2-MOMENT-${SLURM_JOB_ID:-manual}}"
ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"

RQ2_DIR="${ALVIS_REPO}/experiments/rq2_trace_comparison"
DATA_DIR="${RQ2_DIR}/data"
EMB_DIR="${RQ2_DIR}/embeddings"

mkdir -p "${ALVIS_REPO}/logs/rq2" "${DATA_DIR}" "${EMB_DIR}"

echo "=== RQ2 MOMENT embeddings ==="
echo "  RUN_ID=${RUN_ID}"
echo "  ALVIS_REPO=${ALVIS_REPO}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "[0/1] Installing momentfm..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet momentfm 2>&1 | tail -3

# ---------------------------------------------------------------------------
# MOMENT embeddings (skip if already done)
# ---------------------------------------------------------------------------
if [[ -f "${EMB_DIR}/p3_hard_noisy_moment.npz" ]]; then
  echo "[1/1] MOMENT embeddings already exist, skipping."
else
  echo "[1/1] Extracting MOMENT embeddings..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" \
    --env TRANSFORMERS_OFFLINE=1 \
    "${ALVIS_IMAGE}" \
    python -u "${RQ2_DIR}/scripts/02_extract_embeddings.py" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${EMB_DIR}" \
      --model moment \
      --device cuda
fi

echo "=== RQ2 MOMENT done. RUN_ID=${RUN_ID} ==="
