#!/usr/bin/env bash
#SBATCH --job-name=log2vec-rq2-ts2vec
#SBATCH --account=naiss2026-4-91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/rq2/%x_%j.out
#SBATCH --error=logs/rq2/%x_%j.err

set -euo pipefail

RUN_ID="${RUN_ID:-RQ2-TS2VEC-${SLURM_JOB_ID:-manual}}"
ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"

RQ2_DIR="${ALVIS_REPO}/experiments/rq2_trace_comparison"
DATA_DIR="${RQ2_DIR}/data"
EMB_DIR="${RQ2_DIR}/embeddings"

TS2VEC_EPOCHS="${TS2VEC_EPOCHS:-60}"
SEED="${SEED:-42}"

mkdir -p "${ALVIS_REPO}/logs/rq2" "${DATA_DIR}" "${EMB_DIR}"

echo "=== RQ2 TS2Vec training ==="
echo "  RUN_ID=${RUN_ID}"
echo "  ALVIS_REPO=${ALVIS_REPO}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  TS2VEC_EPOCHS=${TS2VEC_EPOCHS}"

for PROBLEM in p1_simple_1d p2_multichannel p3_hard_noisy; do
  if [[ -f "${EMB_DIR}/${PROBLEM}_ts2vec.npz" ]]; then
    echo "[${PROBLEM}] Embeddings already exist, skipping."
    continue
  fi
  echo "[${PROBLEM}] Training TS2Vec..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python -u "${RQ2_DIR}/scripts/03_train_ts2vec.py" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${EMB_DIR}" \
      --problem "${PROBLEM}" \
      --epochs "${TS2VEC_EPOCHS}" \
      --seed "${SEED}" \
      --device cuda
done

echo "=== RQ2 TS2Vec done. RUN_ID=${RUN_ID} ==="
