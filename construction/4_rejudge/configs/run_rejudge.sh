#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"

JUDGE_MODEL="${JUDGE_MODEL:-model/Qwen3-30B-A3B-Instruct-2507}"
[[ -d "${JUDGE_MODEL}" ]] || { echo "missing judge model: ${JUDGE_MODEL}"; exit 1; }
INPUT_JSON="${INPUT_JSON:-out/3_detection/overthink.json}"
OUTPUT_JSON="${OUTPUT_JSON:-out/4_rejudge/judged.json}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_STEPS_PER_SAMPLE="${MAX_STEPS_PER_SAMPLE:-}"
BATCH_SIZE="${BATCH_SIZE:-500}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_TOKENS="${MAX_TOKENS:-256}"
DTYPE="${DTYPE:-bfloat16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
CHECKPOINT_EVERY_BATCHES="${CHECKPOINT_EVERY_BATCHES:-100}"

ARGS=(
  construction/4_rejudge/judge_overthink_vllm.py
  --model_name_or_path "${JUDGE_MODEL}"
  --input_json "${INPUT_JSON}"
  --output_json "${OUTPUT_JSON}"
  --batch_size "${BATCH_SIZE}"
  --max_model_len "${MAX_MODEL_LEN}"
  --max_tokens "${MAX_TOKENS}"
  --dtype "${DTYPE}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --checkpoint_every_batches "${CHECKPOINT_EVERY_BATCHES}"
)

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--enforce_eager)
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_STEPS_PER_SAMPLE}" ]]; then
  ARGS+=(--max_steps_per_sample "${MAX_STEPS_PER_SAMPLE}")
fi

python "${ARGS[@]}"

echo "input json: ${INPUT_JSON}"
echo "judge json: ${OUTPUT_JSON}"
