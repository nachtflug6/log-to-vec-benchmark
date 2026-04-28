#!/usr/bin/env bash
set -euo pipefail

user_name="${1:-$USER}"

echo "Active jobs for ${user_name}:"
squeue -u "${user_name}"
