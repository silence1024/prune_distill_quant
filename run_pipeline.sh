#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash run_pipeline.sh
#   MODEL_ID=Qwen/Qwen3-0.6B SPARSITY=0.7 EPOCHS=1 bash run_pipeline.sh

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
WORK_DIR="${WORK_DIR:-outputs/qwen3_06b_sparse70}"
SPARSITY="${SPARSITY:-0.7}"
SEQ_LEN="${SEQ_LEN:-512}"
CALIB_SAMPLES="${CALIB_SAMPLES:-64}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-5}"
DTYPE="${DTYPE:-float16}"

PRUNE_DIR="${WORK_DIR}/pruned"
DISTILL_DIR="${WORK_DIR}/distilled"
INT8_DIR="${WORK_DIR}/quant_int8"
FP8_DIR="${WORK_DIR}/quant_fp8"

mkdir -p "${WORK_DIR}"

python scripts/prune_sparsegpt.py \
  --model_id "${MODEL_ID}" \
  --output_dir "${PRUNE_DIR}" \
  --sparsity "${SPARSITY}" \
  --seq_len "${SEQ_LEN}" \
  --nsamples "${CALIB_SAMPLES}" \
  --dtype "${DTYPE}"

python scripts/distill_sparse_finetune.py \
  --teacher_model "${MODEL_ID}" \
  --student_model "${PRUNE_DIR}" \
  --mask_file "${PRUNE_DIR}/sparsity_mask.pt" \
  --output_dir "${DISTILL_DIR}" \
  --seq_len "${SEQ_LEN}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --dtype "${DTYPE}"

python scripts/quantize_model.py \
  --model_dir "${DISTILL_DIR}" \
  --output_dir "${INT8_DIR}" \
  --format int8 \
  --dtype "${DTYPE}"

set +e
python scripts/quantize_model.py \
  --model_dir "${DISTILL_DIR}" \
  --output_dir "${FP8_DIR}" \
  --format fp8 \
  --dtype "${DTYPE}"
fp8_status=$?
set -e

if [[ ${fp8_status} -ne 0 ]]; then
  echo "[WARN] FP8 quantization failed (likely missing optimum-quanto). INT8 artifact is still available."
fi

echo "[INFO] Pipeline completed. Artifacts under: ${WORK_DIR}"
