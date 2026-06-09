#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${LIBERO_ROOT:-}" ]]; then
  BDDL_ROOT="${LIBERO_ROOT}/libero/libero/bddl_files"
  INIT_ROOT="${LIBERO_ROOT}/libero/libero/init_files"
else
  BDDL_ROOT="${REPO_ROOT}/benchmark_assets/bddl_files"
  INIT_ROOT="${REPO_ROOT}/benchmark_assets/init_files"
fi

for suite in libero_spatial_occluded libero_goal_occluded libero_object_occluded libero_10_occluded; do
  bddl_dir="${BDDL_ROOT}/${suite}"
  init_dir="${INIT_ROOT}/${suite}"

  if [[ ! -d "${bddl_dir}" ]]; then
    echo "Missing BDDL directory: ${bddl_dir}" >&2
    exit 1
  fi
  if [[ ! -d "${init_dir}" ]]; then
    echo "Missing init directory: ${init_dir}" >&2
    exit 1
  fi

  bddl_count="$(find "${bddl_dir}" -type f \( -name '*.bddl' -o -name 'tasks_info.txt' \) | wc -l)"
  init_count="$(find "${init_dir}" -type f -name '*.pruned_init' | wc -l)"
  echo "${suite}: ${bddl_count} BDDL/task-info files, ${init_count} init files"
done

echo "LIBERO-Occ asset check passed."

