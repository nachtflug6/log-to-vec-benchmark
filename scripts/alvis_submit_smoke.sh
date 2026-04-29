#!/usr/bin/env bash
set -euo pipefail

host="${ALVIS_HOST:-alvis1}"
if [[ -n "${ALVIS_REMOTE_ROOT:-}" ]]; then
  remote_root="${ALVIS_REMOTE_ROOT}"
else
  remote_user="$(ssh "${host}" 'printf "%s" "$USER"')"
  remote_root="/cephyr/users/${remote_user}/Alvis/log-to-vec-benchmark"
fi
account="${ALVIS_ACCOUNT:-naiss2026-4-91}"
partition="${ALVIS_PARTITION:-alvis}"
gpu="${ALVIS_GPU:-T4:1}"
time_limit="${ALVIS_TIME:-00:20:00}"
image="${ALVIS_IMAGE:-/apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif}"
run_id="${RUN_ID:-ALVIS-SMOKE-$(date +%Y%m%d-%H%M%S)}"
num_epochs="${SMOKE_NUM_EPOCHS:-2}"
num_sequences="${SMOKE_NUM_SEQUENCES:-64}"
sequence_length="${SMOKE_SEQUENCE_LENGTH:-12}"
num_features="${SMOKE_NUM_FEATURES:-4}"
batch_size="${SMOKE_BATCH_SIZE:-7}"
seed="${SMOKE_SEED:-42}"
local_collect_dir="${ALVIS_LOCAL_RESULTS_DIR:-outputs/alvis_smoke/${run_id}}"

wait_for_job=0
collect=0
sync_only=0

usage() {
  cat <<USAGE
Usage: $0 [--wait] [--collect] [--sync-only]

Sync the repo to Alvis and submit a tiny Apptainer/Slurm model-deploy smoke test.

Environment:
  ALVIS_HOST          SSH host alias. Default: alvis1
  ALVIS_REMOTE_ROOT   Remote repo directory. Default: /cephyr/users/\$USER/Alvis/log-to-vec-benchmark
  ALVIS_ACCOUNT       Slurm account. Default: naiss2026-4-91
  ALVIS_PARTITION     Slurm partition. Default: alvis
  ALVIS_GPU           Slurm GPU gres suffix. Default: T4:1
  ALVIS_TIME          Slurm time limit. Default: 00:20:00
  ALVIS_IMAGE         Apptainer SIF. Default: /apps/containers/codeserver/codeserver-PyTorch-2.9.1.sif
  RUN_ID              Smoke run identifier. Default: ALVIS-SMOKE-<timestamp>
  ALVIS_LOCAL_RESULTS_DIR  Local collection dir. Default: outputs/alvis_smoke/\$RUN_ID

Smoke sizing:
  SMOKE_NUM_EPOCHS, SMOKE_NUM_SEQUENCES, SMOKE_SEQUENCE_LENGTH, SMOKE_NUM_FEATURES,
  SMOKE_BATCH_SIZE, SMOKE_SEED
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait)
      wait_for_job=1
      shift
      ;;
    --collect)
      collect=1
      shift
      ;;
    --sync-only)
      sync_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALVIS_HOST="${host}" ALVIS_REMOTE_ROOT="${remote_root}" "${script_dir}/alvis_sync.sh"

if [[ "${sync_only}" -eq 1 ]]; then
  echo "Sync complete; skipping submission because --sync-only was provided."
  exit 0
fi

remote_log_dir="${remote_root}/logs/alvis_smoke"
remote_run_dir="${remote_root}/outputs/alvis_smoke/${run_id}"
remote_job_script="${remote_root}/scripts/slurm/alvis_smoke_deploy.slurm"

ssh "${host}" "mkdir -p ${remote_log_dir} ${remote_run_dir}"

export_vars="ALL,RUN_ID=${run_id},ALVIS_IMAGE=${image},ALVIS_REPO=${remote_root},SMOKE_OUTPUT_DIR=${remote_run_dir},SMOKE_NUM_EPOCHS=${num_epochs},SMOKE_NUM_SEQUENCES=${num_sequences},SMOKE_SEQUENCE_LENGTH=${sequence_length},SMOKE_NUM_FEATURES=${num_features},SMOKE_BATCH_SIZE=${batch_size},SMOKE_SEED=${seed}"

job_id="$(
  ssh "${host}" "cd ${remote_root} && sbatch --parsable --account=${account} --partition=${partition} --gres=gpu:${gpu} --time=${time_limit} --export=${export_vars} ${remote_job_script}"
)"
job_id="${job_id%%;*}"

echo "Submitted Alvis smoke job ${job_id}"
echo "  RUN_ID=${run_id}"
echo "  Remote artifacts: ${host}:${remote_run_dir}"
echo "  Remote logs:      ${host}:${remote_log_dir}"

if [[ "${wait_for_job}" -eq 1 ]]; then
  echo "Waiting for job ${job_id} to leave the queue..."
  while ssh "${host}" "test -n \"\$(squeue -h -j ${job_id})\""; do
    sleep 20
    printf "."
  done
  printf "\n"
  ssh "${host}" "sacct -j ${job_id} --format=JobID,JobName%24,State,ExitCode,Elapsed -P 2>/dev/null || true"
fi

if [[ "${collect}" -eq 1 ]]; then
  mkdir -p "${local_collect_dir}/slurm_logs"
  rsync -az "${host}:${remote_run_dir}/" "${local_collect_dir}/"
  rsync -az "${host}:${remote_log_dir}/"*${job_id}* "${local_collect_dir}/slurm_logs/" 2>/dev/null || true
  echo "Collected Alvis smoke artifacts to ${local_collect_dir}"
fi
