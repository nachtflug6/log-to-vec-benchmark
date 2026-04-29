#!/usr/bin/env bash
#SBATCH --job-name=log2vec-smoke-deploy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/alvis_smoke/%x_%j.out
#SBATCH --error=logs/alvis_smoke/%x_%j.err

set -euo pipefail

RUN_ID="${RUN_ID:-ALVIS-SMOKE-${SLURM_JOB_ID:-manual}}"
ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${ALVIS_REPO}/outputs/alvis_smoke/${RUN_ID}}"
SMOKE_NUM_EPOCHS="${SMOKE_NUM_EPOCHS:-2}"
SMOKE_NUM_SEQUENCES="${SMOKE_NUM_SEQUENCES:-64}"
SMOKE_SEQUENCE_LENGTH="${SMOKE_SEQUENCE_LENGTH:-12}"
SMOKE_NUM_FEATURES="${SMOKE_NUM_FEATURES:-4}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-7}"
SMOKE_SEED="${SMOKE_SEED:-42}"

mkdir -p "${ALVIS_REPO}/logs/alvis_smoke" "${SMOKE_OUTPUT_DIR}"

echo "Starting Alvis smoke deploy"
echo "  RUN_ID=${RUN_ID}"
echo "  ALVIS_REPO=${ALVIS_REPO}"
echo "  ALVIS_IMAGE=${ALVIS_IMAGE}"
echo "  SMOKE_OUTPUT_DIR=${SMOKE_OUTPUT_DIR}"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python scripts/smoke_deploy_model.py \
    --run-id "${RUN_ID}" \
    --repo-root "${ALVIS_REPO}" \
    --output-dir "${SMOKE_OUTPUT_DIR}" \
    --num-sequences "${SMOKE_NUM_SEQUENCES}" \
    --sequence-length "${SMOKE_SEQUENCE_LENGTH}" \
    --num-features "${SMOKE_NUM_FEATURES}" \
    --num-epochs "${SMOKE_NUM_EPOCHS}" \
    --batch-size "${SMOKE_BATCH_SIZE}" \
    --seed "${SMOKE_SEED}" \
    --device cuda

echo "Completed Alvis smoke deploy for RUN_ID=${RUN_ID}"
