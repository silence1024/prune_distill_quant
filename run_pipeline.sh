#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash run_pipeline.sh
#   MODEL_ID=Qwen/Qwen3-0.6B SPARSITY=0.7 EPOCHS=1 bash run_pipeline.sh

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
WORK_DIR="${WORK_DIR:-outputs/qwen3_06b_sparse70}"
SPARSITY="${SPARSITY:-0.7}"
SEQ_LEN="${SEQ_LEN:-2048}"
CALIB_SAMPLES="${CALIB_SAMPLES:-256}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-2e-5}"
DTYPE="${DTYPE:-bfloat16}"
# Distill dataset options
DISTILL_DATASET="${DISTILL_DATASET:-wikitext}"
DISTILL_DATASET_SUBSET="${DISTILL_DATASET_SUBSET:-wikitext-2-raw-v1}"
DISTILL_TRAIN_SPLIT="${DISTILL_TRAIN_SPLIT:-train[:99%]}"
DISTILL_EVAL_SPLIT="${DISTILL_EVAL_SPLIT:-train[99%:]}"
DISTILL_MAX_TRAIN_SAMPLES="${DISTILL_MAX_TRAIN_SAMPLES:-4000}"
DISTILL_MAX_EVAL_SAMPLES="${DISTILL_MAX_EVAL_SAMPLES:-400}"
DISTILL_BACKEND="${DISTILL_BACKEND:-none}"  # none|deepspeed|fsdp
DISTILL_LAUNCHER="${DISTILL_LAUNCHER:-python}"  # python|torchrun|accelerate
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
SPARSE_CHECK_EPS="${SPARSE_CHECK_EPS:-0.0}"
FAIL_ON_SPARSE_VIOLATION="${FAIL_ON_SPARSE_VIOLATION:-1}"  # 1|0
USE_WANDB="${USE_WANDB:-1}"  # 1|0
WANDB_PROJECT="${WANDB_PROJECT:-prune_dist_quant}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-prune_dist_quant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"  # online|offline|disabled

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

DISTILL_CMD=(
  scripts/distill_sparse_finetune.py
  --teacher_model "${MODEL_ID}"
  --student_model "${PRUNE_DIR}"
  --mask_file "${PRUNE_DIR}/sparsity_mask.pt"
  --output_dir "${DISTILL_DIR}"
  --dataset "${DISTILL_DATASET}"
  --dataset_subset "${DISTILL_DATASET_SUBSET}"
  --train_split "${DISTILL_TRAIN_SPLIT}"
  --eval_split "${DISTILL_EVAL_SPLIT}"
  --max_train_samples "${DISTILL_MAX_TRAIN_SAMPLES}"
  --max_eval_samples "${DISTILL_MAX_EVAL_SAMPLES}"
  --seq_len "${SEQ_LEN}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM}"
  --lr "${LR}"
  --dtype "${DTYPE}"
  --sparse_check_eps "${SPARSE_CHECK_EPS}"
)

if [[ "${FAIL_ON_SPARSE_VIOLATION}" == "1" ]]; then
  DISTILL_CMD+=(--fail_on_sparse_violation)
fi

if [[ "${USE_WANDB}" == "1" ]]; then
  DISTILL_CMD+=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_run_name "${WANDB_RUN_NAME}" --wandb_mode "${WANDB_MODE}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    DISTILL_CMD+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

if [[ "${DISTILL_BACKEND}" == "deepspeed" ]]; then
  DISTILL_CMD+=(--use_deepspeed --deepspeed_config "${DEEPSPEED_CONFIG}")
elif [[ "${DISTILL_BACKEND}" == "fsdp" ]]; then
  DISTILL_CMD+=(--use_fsdp)
elif [[ "${DISTILL_BACKEND}" != "none" ]]; then
  echo "[ERROR] DISTILL_BACKEND must be one of: none|deepspeed|fsdp" >&2
  exit 1
fi

if [[ "${DISTILL_BACKEND}" == "fsdp" && "${DISTILL_LAUNCHER}" == "python" ]]; then
  echo "[ERROR] FSDP needs multi-process launch. Set DISTILL_LAUNCHER=torchrun or accelerate." >&2
  exit 1
fi

if [[ "${DISTILL_LAUNCHER}" == "python" ]]; then
  python "${DISTILL_CMD[@]}"
elif [[ "${DISTILL_LAUNCHER}" == "torchrun" ]]; then
  torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    "${DISTILL_CMD[@]}"
elif [[ "${DISTILL_LAUNCHER}" == "accelerate" ]]; then
  ACCELERATE_CMD=(accelerate launch)
  if [[ -n "${ACCELERATE_CONFIG}" ]]; then
    ACCELERATE_CMD+=(--config_file "${ACCELERATE_CONFIG}")
  else
    ACCELERATE_CMD+=(--num_processes "${NPROC_PER_NODE}")
  fi
  "${ACCELERATE_CMD[@]}" "${DISTILL_CMD[@]}"
else
  echo "[ERROR] DISTILL_LAUNCHER must be one of: python|torchrun|accelerate" >&2
  exit 1
fi

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
