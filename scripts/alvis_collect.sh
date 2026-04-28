#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <remote_dir_on_alvis> <local_results_dir>"
  exit 1
fi

remote_dir="$1"
local_dir="$2"

mkdir -p "$local_dir"
rsync -avz alvis1:"${remote_dir}"/ "$local_dir"/

echo "Collected artifacts from alvis1:${remote_dir} to ${local_dir}"
