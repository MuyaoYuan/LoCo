#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-model/DeepSeek-R1-Distill-Qwen-7B}"
INPUT_JSON="${INPUT_JSON:-out/2_filter/shortest.json}"
OUTPUT_JSON="${OUTPUT_JSON:-out/3_detection/overthink.json}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_LOGPROBS_TOPK="${PROMPT_LOGPROBS_TOPK:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

ARGS=(
  construction/3_detection/label_filtered_overthink_vllm.py
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --input_json "${INPUT_JSON}"
  --output_json "${OUTPUT_JSON}"
  --dtype "${DTYPE}"
  --max_seq_len "${MAX_SEQ_LEN}"
  --batch_size "${BATCH_SIZE}"
  --prompt_logprobs_topk "${PROMPT_LOGPROBS_TOPK}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  ARGS+=(--max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--enforce_eager)
fi

python "${ARGS[@]}"

echo "input json: ${INPUT_JSON}"
echo "label json: ${OUTPUT_JSON}"
