#!/usr/bin/env bash
#SBATCH --job-name=log2vec-rq2-trace
#SBATCH --account=naiss2026-4-91
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/rq2/%x_%j.out
#SBATCH --error=logs/rq2/%x_%j.err

set -euo pipefail

RUN_ID="${RUN_ID:-RQ2-${SLURM_JOB_ID:-manual}}"
ALVIS_REPO="${ALVIS_REPO:-$PWD}"
ALVIS_IMAGE="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"

RQ2_DIR="${ALVIS_REPO}/experiments/rq2_trace_comparison"
DATA_DIR="${RQ2_DIR}/data"
EMB_DIR="${RQ2_DIR}/embeddings"
METRICS_DIR="${RQ2_DIR}/metrics"
PLOTS_DIR="${RQ2_DIR}/plots"
REPORTS_DIR="${RQ2_DIR}/reports"

TS2VEC_EPOCHS="${TS2VEC_EPOCHS:-60}"
SEED="${SEED:-42}"

mkdir -p "${ALVIS_REPO}/logs/rq2" "${DATA_DIR}" "${EMB_DIR}" "${METRICS_DIR}" "${PLOTS_DIR}" "${REPORTS_DIR}"

echo "=== RQ2 Trace Comparison ==="
echo "  RUN_ID=${RUN_ID}"
echo "  ALVIS_REPO=${ALVIS_REPO}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Install momentfm into user site-packages (home dir is writable inside Apptainer)
echo "[0/6] Installing momentfm..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  pip install --user --quiet momentfm 2>&1 | tail -3

cd "${RQ2_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Generate datasets (skip if already done)
# ---------------------------------------------------------------------------
if [[ -f "${DATA_DIR}/p3_hard_noisy/windows.npz" ]]; then
  echo "[1/6] Datasets already exist, skipping generation."
else
  echo "[1/6] Generating periodic-mode datasets..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python "${RQ2_DIR}/scripts/01_generate_periodic_modes.py" \
      --output_dir "${DATA_DIR}" \
      --num_trajectories 20 \
      --periods_per_segment 5 \
      --seed "${SEED}"
fi

# ---------------------------------------------------------------------------
# Step 2a: FFT embeddings (skip if already done)
# ---------------------------------------------------------------------------
if [[ -f "${EMB_DIR}/p3_hard_noisy_fft.npz" ]]; then
  echo "[2a/6] FFT embeddings already exist, skipping."
else
  echo "[2a/6] Extracting FFT embeddings..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python "${RQ2_DIR}/scripts/02_extract_embeddings.py" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${EMB_DIR}" \
      --model fft
fi

# ---------------------------------------------------------------------------
# Step 2b: MOMENT embeddings (pretrained, GPU)
# ---------------------------------------------------------------------------
if [[ -f "${EMB_DIR}/p3_hard_noisy_moment.npz" ]]; then
  echo "[2b/6] MOMENT embeddings already exist, skipping."
else
  echo "[2b/6] Extracting MOMENT embeddings..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" \
    --env TRANSFORMERS_OFFLINE=1 \
    "${ALVIS_IMAGE}" \
    python -u "${RQ2_DIR}/scripts/02_extract_embeddings.py" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${EMB_DIR}" \
      --model moment \
      --device cuda
fi

# ---------------------------------------------------------------------------
# Step 3: Train TS2Vec on each problem
# ---------------------------------------------------------------------------
echo "[3/6] Training TS2Vec encoders..."
for PROBLEM in p1_simple_1d p2_multichannel p3_hard_noisy; do
  echo "  Training on ${PROBLEM}..."
  apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
    python -u "${RQ2_DIR}/scripts/03_train_ts2vec.py" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${EMB_DIR}" \
      --problem "${PROBLEM}" \
      --epochs "${TS2VEC_EPOCHS}" \
      --seed "${SEED}" \
      --device cuda
done

# ---------------------------------------------------------------------------
# Step 4: Compute trace metrics
# ---------------------------------------------------------------------------
echo "[4/6] Computing trace metrics..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ2_DIR}/scripts/04_compute_trace_metrics.py" \
    --data_dir "${DATA_DIR}" \
    --embeddings_dir "${EMB_DIR}" \
    --output_dir "${METRICS_DIR}"

# ---------------------------------------------------------------------------
# Step 5: Visualize traces
# ---------------------------------------------------------------------------
echo "[5/6] Generating visualizations..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ2_DIR}/scripts/05_visualize_traces.py" \
    --data_dir "${DATA_DIR}" \
    --embeddings_dir "${EMB_DIR}" \
    --output_dir "${PLOTS_DIR}"

# ---------------------------------------------------------------------------
# Step 6: Build report
# ---------------------------------------------------------------------------
echo "[6/6] Building report..."
apptainer exec --nv --pwd "${ALVIS_REPO}" "${ALVIS_IMAGE}" \
  python -u "${RQ2_DIR}/scripts/06_build_report.py" \
    --metrics_dir "${METRICS_DIR}" \
    --plots_dir "${PLOTS_DIR}" \
    --output_dir "${REPORTS_DIR}"

echo "=== RQ2 complete. RUN_ID=${RUN_ID} ==="
echo "  Report: ${REPORTS_DIR}/report.md"
echo "  Metrics: ${METRICS_DIR}/"
echo "  Plots:   ${PLOTS_DIR}/"
