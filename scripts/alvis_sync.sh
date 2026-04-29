#!/usr/bin/env bash
set -euo pipefail

host="${ALVIS_HOST:-alvis1}"
if [[ -n "${ALVIS_REMOTE_ROOT:-}" ]]; then
  remote_root="${ALVIS_REMOTE_ROOT}"
else
  remote_user="$(ssh "${host}" 'printf "%s" "$USER"')"
  remote_root="/cephyr/users/${remote_user}/Alvis/log-to-vec-benchmark"
fi

usage() {
  cat <<USAGE
Usage: ALVIS_REMOTE_ROOT=<remote_dir> $0 [--dry-run]

Sync this repository to Alvis for Apptainer/Slurm smoke runs.

Environment:
  ALVIS_HOST         SSH host alias. Default: alvis1
  ALVIS_REMOTE_ROOT Remote repo directory. Default: /cephyr/users/\$USER/Alvis/log-to-vec-benchmark
USAGE
}

dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
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
repo_root="$(cd "${script_dir}/.." && pwd)"

rsync_args=(
  -az
  --delete
  --exclude .git/
  --exclude .codex/
  --exclude .pytest_cache/
  --exclude __pycache__/
  --exclude '*.pyc'
  --exclude .venv/
  --exclude venv/
  --exclude env/
  --exclude ENV/
  --exclude build/
  --exclude dist/
  --exclude '*.egg-info/'
  --exclude /data/
  --exclude /outputs/
  --exclude /checkpoints/
  --exclude /results/
  --exclude /logs/
  --exclude /runs/
  --exclude /wandb/
  --exclude /context/
)

if [[ "${dry_run}" -eq 1 ]]; then
  rsync_args+=(--dry-run)
fi

ssh "${host}" "mkdir -p ${remote_root}"
rsync "${rsync_args[@]}" "${repo_root}/" "${host}:${remote_root}/"

echo "Synced ${repo_root} to ${host}:${remote_root}"
