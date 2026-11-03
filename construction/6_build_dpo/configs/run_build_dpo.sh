#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"

PARTIALS_DIR="${PARTIALS_DIR:-out/5_regenerate/regenerate_partials}"
SOURCE_RECORDS="${SOURCE_RECORDS:-out/2_filter/shortest.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-out/6_build_dpo/dpo.jsonl}"
TOKENIZER="${TOKENIZER:-model/DeepSeek-R1-Distill-Qwen-7B}"
EXCLUDE_FROM="${EXCLUDE_FROM:-}"
MIN_TOKEN_SAVED="${MIN_TOKEN_SAVED:-300}"
MIN_PAIR_TOKEN_GAP="${MIN_PAIR_TOKEN_GAP:-300}"
MAX_PAIRS_PER_QUESTION="${MAX_PAIRS_PER_QUESTION:-3}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-12000}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-6000}"
INCLUDE_META="${INCLUDE_META:-1}"

ARGS=(
  construction/6_build_dpo/build_dpo.py
  --partials-dir "${PARTIALS_DIR}"
  --source-records "${SOURCE_RECORDS}"
  --output-jsonl "${OUTPUT_JSONL}"
  --tokenizer "${TOKENIZER}"
  --min-token-saved "${MIN_TOKEN_SAVED}"
  --min-pair-token-gap "${MIN_PAIR_TOKEN_GAP}"
  --max-pairs-per-question "${MAX_PAIRS_PER_QUESTION}"
  --max-response-tokens "${MAX_RESPONSE_TOKENS}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS}"
)

if [[ -n "${EXCLUDE_FROM}" ]]; then
  ARGS+=(--exclude-question-ids-from "${EXCLUDE_FROM}")
fi
if [[ "${INCLUDE_META}" == "1" ]]; then
  ARGS+=(--include-meta)
fi

python "${ARGS[@]}"

echo
echo "build_dpo done."
echo "  partials: ${PARTIALS_DIR}"
echo "  source:   ${SOURCE_RECORDS}"
echo "  output:   ${OUTPUT_JSONL}"
echo "  rows:     $(wc -l < "${OUTPUT_JSONL}")"
