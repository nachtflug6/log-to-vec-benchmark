#!/usr/bin/env bash
#SBATCH --job-name=log2vec-rq3-dcc
#SBATCH --account=naiss2026-4-91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --array=0-13%6
#SBATCH --output=logs/rq3/%x_%A_%a.out
#SBATCH --error=logs/rq3/%x_%A_%a.err

set -euo pipefail

ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
RQ3_DIR="${ALVIS_REPO}/experiments/rq3_landscape_sweep"

# Map array task ID to scenario ID
SCENARIO_IDS=(
  baseline
  sig_step
  sig_event
  sig_mixed
  modes_2
  modes_5
  modes_8
  noise_000
  noise_020
  noise_040
  miss_rand10
  miss_rand30
  miss_block30
  miss_channel
)
SCENARIO_ID="${SCENARIO_IDS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${ALVIS_REPO}/logs/rq3" "${RQ3_DIR}/results/embeddings"

echo "=== RQ3 DCC [task ${SLURM_ARRAY_TASK_ID}] scenario=${SCENARIO_ID} ==="
echo "  ALVIS_REPO=${ALVIS_REPO}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ3_DIR}/scripts/04_train_dcc.py" \
    --scenario_id "${SCENARIO_ID}" \
    --data_dir    "${RQ3_DIR}/results/data" \
    --output_dir  "${RQ3_DIR}/results/embeddings" \
    --device cuda \
    --epochs 60

echo "=== RQ3 DCC done: ${SCENARIO_ID} ==="
