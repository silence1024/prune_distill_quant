# Sparse Prune -> Distill -> Quant Pipeline

This repository now contains a runnable reference pipeline for:

1. Loading a pretrained LLM (default: `Qwen/Qwen3-0.6B`)
2. One-shot sparse pruning to 70% sparsity (SparseGPT-style, unstructured)
3. Distillation finetuning on WikiText-2 while training only non-zero sparse weights
4. Post-training quantization to INT8 and optional FP8

## Files

- `scripts/prune_sparsegpt.py`
  - Performs Hessian-aware sparse pruning per linear layer.
  - Saves pruned model and `sparsity_mask.pt`.
  - Prints `[PPL] before_pruning` and `[PPL] after_pruning`.
- `scripts/distill_sparse_finetune.py`
  - Teacher-student distillation on WikiText-2.
  - Only parameters covered by mask are trainable.
  - Gradient and weight updates are masked so zero weights stay zero.
  - Final sparse integrity check: verifies pruned region (mask=0) stays zero after finetune.
  - Optional DeepSpeed backend.
  - Optional FSDP backend (for torchrun / accelerate multi-process launch).
  - Prints `[PPL] finetune_start` and `[PPL] epoch=... eval_ppl=...`.
- `scripts/quantize_model.py`
  - Quantizes the distilled model to INT8 or FP8.
  - Prefers `optimum-quanto`.
  - INT8 fallback: PyTorch dynamic quantization if `optimum-quanto` is unavailable.
  - Prints `[PPL] after_quantization`.
- `run_pipeline.sh`
  - End-to-end script to run prune -> distill -> quantize.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
# Optional for FP8 and better INT8 path:
# python -m pip install optimum-quanto
# Optional for DeepSpeed distillation:
# python -m pip install deepspeed
# accelerate already included in requirements and can be used as launcher.
```

## Quick Start

```bash
bash run_pipeline.sh
```

Override defaults:

```bash
MODEL_ID=Qwen/Qwen3-0.6B \
SPARSITY=0.7 \
SEQ_LEN=512 \
CALIB_SAMPLES=64 \
EPOCHS=1 \
BATCH_SIZE=1 \
GRAD_ACCUM=8 \
LR=2e-5 \
DTYPE=float16 \
SPARSE_CHECK_EPS=0.0 \
FAIL_ON_SPARSE_VIOLATION=1 \
bash run_pipeline.sh
```

Enable DeepSpeed during distillation:

```bash
DISTILL_BACKEND=deepspeed \
DISTILL_LAUNCHER=python \
DEEPSPEED_CONFIG=configs/deepspeed_zero2.json \
bash run_pipeline.sh
```

Enable FSDP (torchrun):

```bash
DISTILL_BACKEND=fsdp \
DISTILL_LAUNCHER=torchrun \
NPROC_PER_NODE=4 \
MASTER_PORT=29501 \
bash run_pipeline.sh
```

Enable FSDP (accelerate launcher):

```bash
DISTILL_BACKEND=fsdp \
DISTILL_LAUNCHER=accelerate \
NPROC_PER_NODE=4 \
bash run_pipeline.sh
```

## Direct Commands

Pruning:

```bash
python scripts/prune_sparsegpt.py \
  --model_id Qwen/Qwen3-0.6B \
  --output_dir outputs/pruned \
  --sparsity 0.7 \
  --nsamples 64 \
  --seq_len 1024
```

Distillation with sparse update constraint:

```bash
python scripts/distill_sparse_finetune.py \
  --teacher_model Qwen/Qwen3-0.6B \
  --student_model outputs/pruned \
  --mask_file outputs/pruned/sparsity_mask.pt \
  --output_dir outputs/distilled \
  --seq_len 512 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --fail_on_sparse_violation \
  --use_deepspeed \
  --deepspeed_config configs/deepspeed_zero2.json
```

Distillation with FSDP via torchrun:

```bash
torchrun --nproc_per_node 4 --master_port 29501 scripts/distill_sparse_finetune.py \
  --teacher_model Qwen/Qwen3-0.6B \
  --student_model outputs/pruned \
  --mask_file outputs/pruned/sparsity_mask.pt \
  --output_dir outputs/distilled_fsdp \
  --seq_len 512 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --use_fsdp
```

Distillation with FSDP via accelerate launch:

```bash
accelerate launch --num_processes 4 scripts/distill_sparse_finetune.py \
  --teacher_model Qwen/Qwen3-0.6B \
  --student_model outputs/pruned \
  --mask_file outputs/pruned/sparsity_mask.pt \
  --output_dir outputs/distilled_fsdp \
  --seq_len 512 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --use_fsdp
```

INT8 quantization:

```bash
python scripts/quantize_model.py \
  --model_dir outputs/distilled \
  --output_dir outputs/quant_int8 \
  --format int8
```

FP8 quantization:

```bash
python scripts/quantize_model.py \
  --model_dir outputs/distilled \
  --output_dir outputs/quant_fp8 \
  --format fp8
```

## Notes

- The pruning implementation is SparseGPT-style Hessian-aware unstructured pruning.
- Distillation loss uses `alpha_kd * KL(student, teacher) + (1 - alpha_kd) * CE`.
- Zeroed sparse weights are re-masked after each optimizer step to enforce exact sparsity.
- FP8 requires a backend that supports FP8 weight quantization (`optimum-quanto` in this implementation).
- PPL evaluation defaults to WikiText-2 `test` split and can be disabled with `--skip_ppl_eval`.
- DeepSpeed path currently supports ZeRO stage `<=2` for sparse mask compatibility.
- FSDP path requires multi-process launch (`torchrun` or `accelerate launch`) and CUDA.
- Finetune ends with a sparse integrity check, saved as `sparse_integrity_check` in `distill_stats.json`.
