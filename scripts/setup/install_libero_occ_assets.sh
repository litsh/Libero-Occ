#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIBERO_ROOT="${LIBERO_ROOT:-${1:-}}"

if [[ -z "${LIBERO_ROOT}" ]]; then
  echo "Usage: LIBERO_ROOT=/path/to/LIBERO bash scripts/setup/install_libero_occ_assets.sh" >&2
  exit 1
fi

BDDL_DST="${LIBERO_ROOT}/libero/libero/bddl_files"
INIT_DST="${LIBERO_ROOT}/libero/libero/init_files"

if [[ ! -d "${BDDL_DST}" || ! -d "${INIT_DST}" ]]; then
  echo "Could not find LIBERO asset directories under: ${LIBERO_ROOT}" >&2
  echo "Expected: ${BDDL_DST} and ${INIT_DST}" >&2
  exit 1
fi

for suite in libero_spatial_occluded libero_goal_occluded libero_object_occluded libero_10_occluded; do
  cp -R "${REPO_ROOT}/benchmark_assets/bddl_files/${suite}" "${BDDL_DST}/"
  cp -R "${REPO_ROOT}/benchmark_assets/init_files/${suite}" "${INIT_DST}/"
done

echo "Installed LIBERO-Occ assets into ${LIBERO_ROOT}"

