#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"

BASE_MODEL="${BASE_MODEL:-model/DeepSeek-R1-Distill-Qwen-7B}"
GPUS="${GPUS:-0,1,2,3}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-3}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-out/1_rollout}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DATASET_NAME="${DATASET_NAME:-data/math14k}"

ARGS=(
  construction/1_rollout/multi_run.py
  --num-rollouts "${NUM_ROLLOUTS}"
  --base-seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --response-mode full
  --checkpoint-every-batches 10
  --dataset-name "${DATASET_NAME}"
  --dataset-split train
  --dataset-question-field question
  --dataset-answer-field answer
  --output-root "${OUTPUT_ROOT}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

CUDA_VISIBLE_DEVICES="${GPUS}" BASE_MODEL="${BASE_MODEL}" python "${ARGS[@]}"

echo "rollout done: ${OUTPUT_ROOT}/run_*/answers.json"
