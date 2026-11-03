#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BASE_MODEL="${BASE_MODEL:-model/DeepSeek-R1-Distill-Qwen-7B}"
DATA="${DATA:-out/6_build_dpo/dpo.jsonl}"
OUT="${OUT:-out/train}"
CONFIG="${CONFIG:-train/configs/dpo_loco.yaml}"
EPOCHS="${EPOCHS:-2}"
EFF_BS="${EFF_BS:-64}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
ZERO_STAGE="${ZERO_STAGE:-2}"
GPUS="${GPUS:-0,1,2,3}"

IFS=',' read -ra GPULIST <<< "${GPUS}"
NGPU="${#GPULIST[@]}"
GA="$(python3 -c "print(max(1, round(${EFF_BS}/${NGPU})))")"

[[ -f "${DATA}" ]] || { echo "missing dataset: ${DATA}"; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "missing config: ${CONFIG}"; exit 1; }

export ZERO_STAGE

TMP_YAML="$(mktemp --suffix=.yaml)"
trap 'rm -f "${TMP_YAML}"' EXIT
sed \
  -e "s|^  name: .*|  name: \"${BASE_MODEL}\"|" \
  -e "s|^  path: .*|  path: \"${DATA}\"|" \
  -e "s|^  output_dir: .*|  output_dir: \"${OUT}\"|" \
  -e "s|^  num_train_epochs: .*|  num_train_epochs: ${EPOCHS}|" \
  -e "s|^  gradient_accumulation_steps: .*|  gradient_accumulation_steps: ${GA}|" \
  -e "s|^  per_device_train_batch_size: .*|  per_device_train_batch_size: ${PER_DEVICE_BS}|" \
  "${CONFIG}" > "${TMP_YAML}"

RESUME_ARG=""
NCK="$(ls -d "${OUT}"/checkpoint-* 2>/dev/null | wc -l)"
[[ "${NCK}" -gt 0 ]] && RESUME_ARG="--resume" && echo "resuming from ${OUT} (${NCK} checkpoints)"

LOG_DIR="${OUT}/logs"; mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/train_$(date '+%Y%m%d_%H%M%S').log"

echo "gpus=${GPUS} (${NGPU})  eff_bs=$((PER_DEVICE_BS * GA * NGPU))  grad_accum=${GA}  epochs=${EPOCHS}  zero=${ZERO_STAGE}"
echo "data=${DATA}  base=${BASE_MODEL}"
echo "out=${OUT}  log=${LOG}"

deepspeed --master_port="${MASTER_PORT:-29500}" --include "localhost:${GPUS}" \
  train/train_dpo.py --config "${TMP_YAML}" ${RESUME_ARG} 2>&1 | tee "${LOG}"
RC="${PIPESTATUS[0]}"

if [[ "${RC}" -eq 0 ]]; then
  echo "done. adapters:"
  ls -d "${OUT}"/checkpoint-* 2>/dev/null || echo "  (no checkpoint)"
else
  echo "FAILED rc=${RC} -- see ${LOG}"
fi
exit "${RC}"
