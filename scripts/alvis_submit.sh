#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <job_script> [run_id]"
  exit 1
fi

job_script="$1"
run_id="${2:-EXP-$(date +%Y%m%d-%H%M%S)}"

export RUN_ID="$run_id"

echo "Submitting ${job_script} with RUN_ID=${RUN_ID}"
sbatch "$job_script"
