#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"

INPUT_DIR="${INPUT_DIR:-out/1_rollout}"
OUTPUT_DIR="${OUTPUT_DIR:-out/2_filter}"

python construction/2_filter/filter_runs.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}"

echo "filter done."
echo "  input:  ${INPUT_DIR}"
echo "  output: ${OUTPUT_DIR} (shortest.json / longest.json / all_valid.json / discarded.json / summary.json)"
