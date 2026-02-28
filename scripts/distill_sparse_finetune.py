#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from ppl_utils import (
    build_lm_tensor_dataset,
    evaluate_causal_lm_loss_and_ppl_from_dataloader,
    loss_to_ppl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distillation finetuning for a sparse model with masked weight updates."
    )
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--mask_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=4000)
    parser.add_argument("--max_eval_samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha_kd", type=float, default=0.7)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--sparse_check_eps", type=float, default=0.0)
    parser.add_argument("--fail_on_sparse_violation", action="store_true")
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default="")
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "hybrid_shard", "no_shard"],
    )
    parser.add_argument("--fsdp_min_num_params", type=int, default=1_000_000)
    parser.add_argument("--fsdp_backward_prefetch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def adapt_dtype_for_device(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    if device.type == "cpu" and dtype == torch.float16:
        print("[WARN] float16 on CPU is not supported reliably; switching to float32.")
        return torch.float32
    return dtype


def load_deepspeed_config(path: str) -> Dict:
    if not path:
        raise ValueError("--use_deepspeed was set but --deepspeed_config is empty.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    zero_stage = int(cfg.get("zero_optimization", {}).get("stage", 0))
    if zero_stage > 2:
        raise ValueError(
            "Current sparse-mask training only supports DeepSpeed ZeRO stage <= 2."
        )
    if int(cfg.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError(
            "Set DeepSpeed config gradient_accumulation_steps=1; this script controls "
            "accumulation via --grad_accum_steps."
        )
    return cfg


def get_dist_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def normalize_param_name(name: str) -> str:
    token = "_fsdp_wrapped_module."
    while token in name:
        name = name.replace(token, "")
    return name


def build_param_lookup(model: nn.Module) -> Dict[str, torch.nn.Parameter]:
    lookup: Dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        lookup[name] = param
        normalized = normalize_param_name(name)
        if normalized not in lookup:
            lookup[normalized] = param
    return lookup


def apply_masks_and_freeze(
    model: nn.Module,
    mask_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    trainable_masks: Dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        normalized = normalize_param_name(name)
        if normalized in mask_dict:
            mask = mask_dict[normalized].to(device=param.device, dtype=param.dtype)
            param.requires_grad = True
            with torch.no_grad():
                param.mul_(mask)
            param.register_hook(lambda grad, m=mask: grad * m)
            trainable_masks[normalized] = mask
        else:
            param.requires_grad = False

    if not trainable_masks:
        raise RuntimeError("No trainable parameters matched mask entries.")
    return trainable_masks


@torch.no_grad()
def enforce_sparse_masks(model: nn.Module, trainable_masks: Dict[str, torch.Tensor]) -> None:
    param_lookup = build_param_lookup(model)
    for mask_name, mask in trainable_masks.items():
        if mask_name not in param_lookup:
            raise KeyError(f"Parameter matching mask key `{mask_name}` not found in model.")
        param = param_lookup[mask_name]
        param.mul_(mask)


def masked_weight_count(trainable_masks: Dict[str, torch.Tensor]) -> int:
    total = 0
    for mask in trainable_masks.values():
        total += int(mask.sum().item())
    return total


def run_sparse_integrity_check(
    state_tensors: Dict[str, torch.Tensor],
    trainable_masks: Dict[str, torch.Tensor],
    eps: float,
) -> Dict:
    per_param = {}
    total_elements = 0
    total_nonzero = 0
    total_pruned_elements = 0
    total_pruned_violations = 0
    missing_params = []

    for name, mask in trainable_masks.items():
        if name not in state_tensors:
            missing_params.append(name)
            continue

        param = state_tensors[name].detach()
        mask_bool = mask.to(device=param.device, dtype=torch.bool)
        abs_param = param.abs()
        nonzero = abs_param > eps
        pruned_region = ~mask_bool
        pruned_violations = (nonzero & pruned_region).sum().item()
        param_nonzero = nonzero.sum().item()
        param_elems = param.numel()
        pruned_elems = pruned_region.sum().item()

        total_elements += int(param_elems)
        total_nonzero += int(param_nonzero)
        total_pruned_elements += int(pruned_elems)
        total_pruned_violations += int(pruned_violations)

        per_param[name] = {
            "elements": int(param_elems),
            "nonzero_eps": int(param_nonzero),
            "actual_sparsity_eps": 1.0 - (float(param_nonzero) / max(1, float(param_elems))),
            "target_sparsity_from_mask": 1.0 - (float(mask_bool.sum().item()) / max(1, float(param_elems))),
            "pruned_elements": int(pruned_elems),
            "violations_outside_mask": int(pruned_violations),
        }

    return {
        "eps": eps,
        "checked_param_count": len(per_param),
        "missing_param_count": len(missing_params),
        "missing_params": missing_params,
        "total_elements": total_elements,
        "total_nonzero_eps": total_nonzero,
        "global_sparsity_eps": 1.0 - (float(total_nonzero) / max(1, float(total_elements))),
        "total_pruned_elements": total_pruned_elements,
        "total_violations_outside_mask": total_pruned_violations,
        "passed": (total_pruned_violations == 0 and len(missing_params) == 0),
        "per_param": per_param,
    }


def fsdp_sharding_strategy_from_str(name: str) -> ShardingStrategy:
    mapping = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    return mapping[name]


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_deepspeed and args.use_fsdp:
        raise ValueError("Choose only one backend: --use_deepspeed or --use_fsdp.")

    rank, local_rank, world_size = get_dist_info()
    if args.use_fsdp and world_size <= 1:
        raise ValueError(
            "FSDP requires multi-process launch. Use torchrun/accelerate with world_size > 1."
        )
    if world_size > 1 and not (args.use_fsdp or args.use_deepspeed):
        raise ValueError(
            "Distributed launch detected but no distributed backend enabled. "
            "Set --use_fsdp or --use_deepspeed."
        )

    # Device selection: for distributed CUDA runs, each process pins to LOCAL_RANK.
    if args.device.startswith("cuda") and torch.cuda.is_available():
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(args.device)
    else:
        device = torch.device(args.device)

    if args.use_fsdp and device.type != "cuda":
        raise ValueError("FSDP mode requires CUDA device.")
    if args.use_deepspeed and device.type != "cuda":
        raise ValueError("DeepSpeed mode requires CUDA device.")

    dtype = adapt_dtype_for_device(resolve_dtype(args.dtype), device)

    fsdp_dist_initialized_here = False
    if args.use_fsdp and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        fsdp_dist_initialized_here = True
        rank, local_rank, world_size = get_dist_info()

    is_main_process = rank == 0

    def log(msg: str) -> None:
        if is_main_process:
            print(msg, flush=True)

    set_seed(args.seed + rank)

    log(f"[INFO] rank={rank} local_rank={local_rank} world_size={world_size}")
    log(f"[INFO] Loading tokenizer from: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"[INFO] Loading teacher model: {args.teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    log(f"[INFO] Loading student model: {args.student_model}")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    student.train()

    student_train_model: nn.Module = student
    if args.use_fsdp:
        auto_wrap = partial(
            size_based_auto_wrap_policy,
            min_num_params=args.fsdp_min_num_params,
        )
        mp_policy = None
        if dtype in (torch.float16, torch.bfloat16):
            mp_policy = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
            )
        student_train_model = FSDP(
            student,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=fsdp_sharding_strategy_from_str(args.fsdp_sharding_strategy),
            mixed_precision=mp_policy,
            device_id=device,
            use_orig_params=True,
            backward_prefetch=(
                BackwardPrefetch.BACKWARD_PRE if args.fsdp_backward_prefetch else None
            ),
        )
        log(
            "[INFO] FSDP enabled. "
            f"sharding={args.fsdp_sharding_strategy}, "
            f"min_num_params={args.fsdp_min_num_params}"
        )

    log(f"[INFO] Loading mask file: {args.mask_file}")
    mask_dict: Dict[str, torch.Tensor] = torch.load(args.mask_file, map_location="cpu")
    trainable_masks = apply_masks_and_freeze(student_train_model, mask_dict)
    log(f"[INFO] Trainable sparse weights: {masked_weight_count(trainable_masks):,}")
    log(f"[INFO] Masked parameter tensors: {len(trainable_masks)}")

    log("[INFO] Building train/eval datasets...")
    train_ds = build_lm_tensor_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        subset=args.dataset_subset,
        split=args.train_split,
        seq_len=args.seq_len,
        max_samples=args.max_train_samples,
    )
    eval_ds = build_lm_tensor_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        subset=args.dataset_subset,
        split=args.eval_split,
        seq_len=args.seq_len,
        max_samples=args.max_eval_samples,
    )

    train_sampler = None
    eval_sampler = None
    if args.use_fsdp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
        eval_sampler = DistributedSampler(
            eval_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=eval_sampler,
        drop_last=False,
    )

    trainable_params = [p for p in student_train_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_update_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    use_deepspeed = args.use_deepspeed
    student_engine = None
    if use_deepspeed:
        try:
            import deepspeed
        except Exception as exc:
            raise RuntimeError(
                "DeepSpeed is not installed. Install it first: pip install deepspeed"
            ) from exc

        ds_cfg = load_deepspeed_config(args.deepspeed_config)
        ds_cfg["train_micro_batch_size_per_gpu"] = args.batch_size
        ds_cfg["gradient_accumulation_steps"] = 1
        ds_cfg.setdefault("fp16", {"enabled": False})
        ds_cfg.setdefault("bf16", {"enabled": False})
        ds_cfg["fp16"]["enabled"] = dtype == torch.float16
        ds_cfg["bf16"]["enabled"] = dtype == torch.bfloat16

        student_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=student_train_model,
            model_parameters=trainable_params,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_cfg,
        )
        log(
            "[INFO] DeepSpeed enabled. "
            f"ZeRO stage={ds_cfg.get('zero_optimization', {}).get('stage', 0)}"
        )

    train_model: nn.Module
    mask_target_model: nn.Module
    if use_deepspeed:
        train_model = student_engine
        mask_target_model = student_engine.module
    else:
        train_model = student_train_model
        mask_target_model = student_train_model

    log(f"[INFO] Total update steps: {total_update_steps}")
    log(f"[INFO] Warmup steps: {warmup_steps}")

    global_step = 0
    if use_deepspeed:
        student_engine.zero_grad()
    else:
        optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    running_forward_steps = 0
    epoch_eval_history = []

    eval_model = student_engine.module if use_deepspeed else train_model
    init_eval_loss, init_eval_ppl = evaluate_causal_lm_loss_and_ppl_from_dataloader(
        model=eval_model,
        dataloader=eval_loader,
        device=device,
        desc="Eval PPL before finetune",
        show_progress=is_main_process,
        distributed=args.use_fsdp,
    )
    log(f"[PPL] finetune_start eval_loss={init_eval_loss:.4f} eval_ppl={init_eval_ppl:.4f}")

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not is_main_process,
        )

        for step, (input_ids,) in enumerate(pbar, start=1):
            input_ids = input_ids.to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids).logits

            student_logits = train_model(input_ids=input_ids).logits
            s = student_logits[:, :-1, :].contiguous()
            t = teacher_logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()

            ce_loss = F.cross_entropy(
                s.view(-1, s.size(-1)),
                labels.view(-1),
                reduction="mean",
            )
            kd_loss = F.kl_div(
                F.log_softmax(s / args.temperature, dim=-1),
                F.softmax(t / args.temperature, dim=-1),
                reduction="batchmean",
            ) * (args.temperature**2)

            should_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))
            loss = args.alpha_kd * kd_loss + (1.0 - args.alpha_kd) * ce_loss
            loss = loss / args.grad_accum_steps

            if use_deepspeed:
                student_engine.backward(loss)
            else:
                backward_ctx = (
                    train_model.no_sync()
                    if args.use_fsdp and not should_step
                    else nullcontext()
                )
                with backward_ctx:
                    loss.backward()

            running_loss += loss.item() * args.grad_accum_steps
            running_ce += ce_loss.item()
            running_kd += kd_loss.item()
            running_forward_steps += 1

            if should_step:
                if use_deepspeed:
                    student_engine.step()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    enforce_sparse_masks(mask_target_model, trainable_masks)

                global_step += 1

                if global_step % args.log_every == 0 and is_main_process:
                    denom = max(1, running_forward_steps)
                    avg_loss = running_loss / denom
                    avg_ce = running_ce / denom
                    avg_kd = running_kd / denom
                    avg_train_ppl = loss_to_ppl(avg_ce)
                    if use_deepspeed:
                        current_lr = (
                            float(student_engine.get_lr()[0])
                            if hasattr(student_engine, "get_lr")
                            else float(optimizer.param_groups[0]["lr"])
                        )
                    else:
                        current_lr = float(scheduler.get_last_lr()[0])
                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "ce": f"{avg_ce:.4f}",
                            "kd": f"{avg_kd:.4f}",
                            "train_ppl": f"{avg_train_ppl:.2f}",
                            "lr": f"{current_lr:.2e}",
                        }
                    )
                    running_loss = 0.0
                    running_ce = 0.0
                    running_kd = 0.0
                    running_forward_steps = 0

        eval_model = student_engine.module if use_deepspeed else train_model
        eval_loss, eval_ppl = evaluate_causal_lm_loss_and_ppl_from_dataloader(
            model=eval_model,
            dataloader=eval_loader,
            device=device,
            desc=f"Eval PPL epoch {epoch + 1}",
            show_progress=is_main_process,
            distributed=args.use_fsdp,
        )
        epoch_eval_history.append(
            {
                "epoch": epoch + 1,
                "eval_loss": eval_loss,
                "eval_ppl": eval_ppl,
            }
        )
        log(f"[PPL] epoch={epoch + 1} eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.4f}")

    with torch.no_grad():
        enforce_sparse_masks(mask_target_model, trainable_masks)

    # Save final model and keep one full state_dict on main rank for stats in FSDP mode.
    full_state_dict = None
    if args.use_fsdp:
        fsdp_model = train_model
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            full_cfg,
        ):
            full_state_dict = fsdp_model.state_dict()
        if is_main_process:
            fsdp_model.module.save_pretrained(args.output_dir, state_dict=full_state_dict)
            tokenizer.save_pretrained(args.output_dir)
    else:
        model_to_save = student_engine.module if use_deepspeed else train_model
        if is_main_process:
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    if args.use_fsdp and dist.is_initialized():
        dist.barrier()

    if is_main_process:
        out_mask = os.path.join(args.output_dir, "sparsity_mask.pt")
        torch.save(mask_dict, out_mask)

        final_stats = {}
        total = 0
        total_nonzero = 0
        sparse_check = {}

        if args.use_fsdp:
            if full_state_dict is None:
                raise RuntimeError("Expected FSDP full state_dict on main process.")
            for name, mask in trainable_masks.items():
                if name not in full_state_dict:
                    continue
                param = full_state_dict[name]
                nonzero = int((param != 0).sum().item())
                elems = param.numel()
                total += elems
                total_nonzero += nonzero
                final_stats[name] = {
                    "elements": elems,
                    "nonzero": nonzero,
                    "sparsity": 1.0 - (nonzero / elems),
                }
            sparse_check = run_sparse_integrity_check(
                state_tensors=full_state_dict,
                trainable_masks=trainable_masks,
                eps=args.sparse_check_eps,
            )
        else:
            stats_model = student_engine.module if use_deepspeed else train_model
            param_lookup = build_param_lookup(stats_model)
            with torch.no_grad():
                for name, mask in trainable_masks.items():
                    if name not in param_lookup:
                        continue
                    param = param_lookup[name]
                    nonzero = int((param != 0).sum().item())
                    elems = param.numel()
                    total += elems
                    total_nonzero += nonzero
                    final_stats[name] = {
                        "elements": elems,
                        "nonzero": nonzero,
                        "sparsity": 1.0 - (nonzero / elems),
                    }
            sparse_check = run_sparse_integrity_check(
                state_tensors=param_lookup,
                trainable_masks=trainable_masks,
                eps=args.sparse_check_eps,
            )

        if sparse_check.get("passed", False):
            log(
                "[CHECK] Sparse integrity PASS: "
                f"violations_outside_mask={sparse_check['total_violations_outside_mask']}, "
                f"checked_params={sparse_check['checked_param_count']}, "
                f"eps={sparse_check['eps']}"
            )
        else:
            log(
                "[CHECK] Sparse integrity FAIL: "
                f"violations_outside_mask={sparse_check['total_violations_outside_mask']}, "
                f"missing_params={sparse_check['missing_param_count']}, "
                f"eps={sparse_check['eps']}"
            )
            if args.fail_on_sparse_violation:
                raise RuntimeError(
                    "Sparse integrity check failed. "
                    "Set --sparse_check_eps to a larger value or investigate mask enforcement."
                )

        with open(
            os.path.join(args.output_dir, "distill_stats.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "teacher_model": args.teacher_model,
                    "student_model": args.student_model,
                    "backend": (
                        "deepspeed"
                        if use_deepspeed
                        else ("fsdp" if args.use_fsdp else "none")
                    ),
                    "world_size": world_size,
                    "global_step": global_step,
                    "finetune_start_eval_loss": init_eval_loss,
                    "finetune_start_eval_ppl": init_eval_ppl,
                    "epoch_eval_history": epoch_eval_history,
                    "final_eval_loss": (
                        epoch_eval_history[-1]["eval_loss"]
                        if epoch_eval_history
                        else init_eval_loss
                    ),
                    "final_eval_ppl": (
                        epoch_eval_history[-1]["eval_ppl"]
                        if epoch_eval_history
                        else init_eval_ppl
                    ),
                    "global_sparsity_masked_params": 1.0
                    - (total_nonzero / max(1, total)),
                    "trainable_sparse_weight_count": masked_weight_count(trainable_masks),
                    "sparse_integrity_check": sparse_check,
                    "param_stats": final_stats,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        log(f"[INFO] Distilled sparse model saved to: {args.output_dir}")
        log(f"[INFO] Mask file saved to: {out_mask}")

    if args.use_fsdp and dist.is_initialized() and fsdp_dist_initialized_here:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
