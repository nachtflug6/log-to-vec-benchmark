#!/usr/bin/env bash
# Submit all log-to-vec-benchmark experiments to Alvis.
#
# Usage (from repo root on Alvis):
#   cd /cephyr/users/silvan/Alvis/log-to-vec-benchmark
#   bash submit_all.sh
#
# Each RQ runs as a single GPU job (generate → embed → metrics → plots → report).
# RQ3 DCC training is submitted separately as an array job after RQ3 finishes.
#
# Expected total GPU hours: ~7 h  (4 jobs × T4:1)
#   RQ2: 1.5 h  |  RQ3: 2.5 h  |  RQ4: 2.0 h  |  RQ5: 1.5 h

set -euo pipefail

REPO_ROOT="${ALVIS_REPO:-$PWD}"

# Verify we are in the right place
if [[ ! -f "${REPO_ROOT}/experiments/rq2_trace_comparison/slurm/rq2_full.slurm" ]]; then
  echo "ERROR: run this script from the repo root (or set ALVIS_REPO)."
  exit 1
fi

# Create log directories
mkdir -p logs/rq2 logs/rq3 logs/rq4 logs/rq5

echo "=== Submitting log-to-vec-benchmark ==="

# ── RQ2: Trace comparison (FFT vs MOMENT vs TS2Vec, 3 problems) ──────────────
RQ2_JID=$(sbatch --parsable \
  experiments/rq2_trace_comparison/slurm/rq2_full.slurm)
echo "  RQ2 full pipeline  → job ${RQ2_JID}  (1:30 h)"

# ── RQ3: Landscape sweep (14 scenarios × FFT + MOMENT) ───────────────────────
RQ3_JID=$(sbatch --parsable \
  experiments/rq3_landscape_sweep/slurm/rq3_full.slurm)
echo "  RQ3 full pipeline  → job ${RQ3_JID}  (2:30 h)"

# ── RQ3 DCC: optional contrastive training, starts after RQ3 data exists ─────
RQ3_DCC_JID=$(sbatch --parsable \
  --dependency=afterok:${RQ3_JID} \
  experiments/rq3_landscape_sweep/slurm/rq3_dcc.slurm)
echo "  RQ3 DCC training   → job ${RQ3_DCC_JID}  (depends on ${RQ3_JID})"

# ── RQ4: CNC benchmark (7 scenarios × FFT + MOMENT, dual labels) ─────────────
RQ4_JID=$(sbatch --parsable \
  experiments/rq4_cnc_benchmark/slurm/rq4_full.slurm)
echo "  RQ4 full pipeline  → job ${RQ4_JID}  (2:00 h)"

# ── RQ5: Window sweep (5 sizes × FFT + MOMENT) ───────────────────────────────
RQ5_JID=$(sbatch --parsable \
  experiments/rq5_window_sweep/slurm/rq5_full.slurm)
echo "  RQ5 full pipeline  → job ${RQ5_JID}  (1:30 h)"

echo ""
echo "=== All jobs submitted ==="
echo "  Monitor with:  squeue -u \$USER"
echo "  Cancel all:    scancel ${RQ2_JID} ${RQ3_JID} ${RQ3_DCC_JID} ${RQ4_JID} ${RQ5_JID}"
echo ""
echo "  Logs:"
echo "    logs/rq2/   logs/rq3/   logs/rq4/   logs/rq5/"
