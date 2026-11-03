#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-model/DeepSeek-R1-Distill-Qwen-7B}"
GPUS="${GPUS:-0,1,2,3}"
JUDGE_JSON="${JUDGE_JSON:-out/4_rejudge/judged.json}"
SAMPLED_JSON="${SAMPLED_JSON:-out/5_regenerate/judged_sampled.json}"
OUTPUT_JSON="${OUTPUT_JSON:-out/5_regenerate/regenerate.json}"
MAX_STEPS_PER_QUESTION="${MAX_STEPS_PER_QUESTION:-10}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-32000}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_TASKS="${MAX_TASKS:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-34000}"
DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
STEP_LABEL_SOURCE="${STEP_LABEL_SOURCE:-judge}"
CHECKPOINT_EVERY_BATCHES="${CHECKPOINT_EVERY_BATCHES:-10}"

python construction/5_regenerate/sample.py \
  --input "${JUDGE_JSON}" \
  --output "${SAMPLED_JSON}" \
  --max-steps-per-question "${MAX_STEPS_PER_QUESTION}" \
  --seed "${SAMPLE_SEED}"

ARGS=(
  construction/5_regenerate/regenerate_from_overthink_vllm.py
  --input_json "${SAMPLED_JSON}"
  --output_json "${OUTPUT_JSON}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --num_rollouts "${NUM_ROLLOUTS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_tokens "${MAX_TOKENS}"
  --seed "${SEED}"
  --batch_size "${BATCH_SIZE}"
  --max_seq_len "${MAX_SEQ_LEN}"
  --dtype "${DTYPE}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --step_label_source "${STEP_LABEL_SOURCE}"
  --checkpoint_every_batches "${CHECKPOINT_EVERY_BATCHES}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_TASKS}" ]]; then
  ARGS+=(--max_tasks "${MAX_TASKS}")
fi

CUDA_VISIBLE_DEVICES="${GPUS}" python "${ARGS[@]}"

echo "input json: ${SAMPLED_JSON}"
echo "output json: ${OUTPUT_JSON}"
echo "partials:   ${OUTPUT_JSON%.json}_partials"
