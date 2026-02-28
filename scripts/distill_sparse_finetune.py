#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from ppl_utils import build_lm_tensor_dataset, evaluate_causal_lm_loss_and_ppl, loss_to_ppl


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


def apply_masks_and_freeze(
    student: AutoModelForCausalLM,
    mask_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    trainable_masks: Dict[str, torch.Tensor] = {}
    model_state = dict(student.named_parameters())

    for name, param in model_state.items():
        if name in mask_dict:
            mask = mask_dict[name].to(device=param.device, dtype=param.dtype)
            param.requires_grad = True
            with torch.no_grad():
                param.mul_(mask)
            param.register_hook(lambda grad, m=mask: grad * m)
            trainable_masks[name] = mask
        else:
            param.requires_grad = False

    if not trainable_masks:
        raise RuntimeError("No trainable parameters matched mask entries.")
    return trainable_masks


def masked_weight_count(trainable_masks: Dict[str, torch.Tensor]) -> int:
    total = 0
    for mask in trainable_masks.values():
        total += int(mask.sum().item())
    return total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = adapt_dtype_for_device(resolve_dtype(args.dtype), device)

    print(f"[INFO] Loading tokenizer from: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading teacher model: {args.teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[INFO] Loading student model: {args.student_model}")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    student.train()

    print(f"[INFO] Loading mask file: {args.mask_file}")
    mask_dict: Dict[str, torch.Tensor] = torch.load(args.mask_file, map_location="cpu")
    trainable_masks = apply_masks_and_freeze(student, mask_dict)
    print(f"[INFO] Trainable sparse weights: {masked_weight_count(trainable_masks):,}")
    print(f"[INFO] Masked parameter tensors: {len(trainable_masks)}")

    print("[INFO] Building train/eval datasets...")
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
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

    print(f"[INFO] Total update steps: {total_update_steps}")
    print(f"[INFO] Warmup steps: {warmup_steps}")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    running_forward_steps = 0
    epoch_eval_history = []

    init_eval_loss, init_eval_ppl = evaluate_causal_lm_loss_and_ppl(
        model=student,
        dataset=eval_ds,
        device=device,
        batch_size=args.eval_batch_size,
        desc="Eval PPL before finetune",
        show_progress=True,
    )
    print(f"[PPL] finetune_start eval_loss={init_eval_loss:.4f} eval_ppl={init_eval_ppl:.4f}")

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, (input_ids,) in enumerate(pbar, start=1):
            input_ids = input_ids.to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids).logits

            student_logits = student(input_ids=input_ids).logits
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

            loss = args.alpha_kd * kd_loss + (1.0 - args.alpha_kd) * ce_loss
            loss = loss / args.grad_accum_steps
            loss.backward()

            running_loss += loss.item() * args.grad_accum_steps
            running_ce += ce_loss.item()
            running_kd += kd_loss.item()
            running_forward_steps += 1

            should_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    named_params = dict(student.named_parameters())
                    for name, mask in trainable_masks.items():
                        named_params[name].mul_(mask)

                global_step += 1

                if global_step % args.log_every == 0:
                    denom = max(1, running_forward_steps)
                    avg_loss = running_loss / denom
                    avg_ce = running_ce / denom
                    avg_kd = running_kd / denom
                    avg_train_ppl = loss_to_ppl(avg_ce)
                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "ce": f"{avg_ce:.4f}",
                            "kd": f"{avg_kd:.4f}",
                            "train_ppl": f"{avg_train_ppl:.2f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        }
                    )
                    running_loss = 0.0
                    running_ce = 0.0
                    running_kd = 0.0
                    running_forward_steps = 0

        eval_loss, eval_ppl = evaluate_causal_lm_loss_and_ppl(
            model=student,
            dataset=eval_ds,
            device=device,
            batch_size=args.eval_batch_size,
            desc=f"Eval PPL epoch {epoch + 1}",
            show_progress=True,
        )
        epoch_eval_history.append(
            {
                "epoch": epoch + 1,
                "eval_loss": eval_loss,
                "eval_ppl": eval_ppl,
            }
        )
        print(f"[PPL] epoch={epoch + 1} eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.4f}")

    with torch.no_grad():
        named_params = dict(student.named_parameters())
        for name, mask in trainable_masks.items():
            named_params[name].mul_(mask)

    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Re-save mask to keep a consistent artifact set.
    out_mask = os.path.join(args.output_dir, "sparsity_mask.pt")
    torch.save(mask_dict, out_mask)

    final_stats = {}
    total = 0
    total_nonzero = 0
    with torch.no_grad():
        named_params = dict(student.named_parameters())
        for name, mask in trainable_masks.items():
            param = named_params[name]
            nonzero = int((param != 0).sum().item())
            elems = param.numel()
            total += elems
            total_nonzero += nonzero
            final_stats[name] = {
                "elements": elems,
                "nonzero": nonzero,
                "sparsity": 1.0 - (nonzero / elems),
            }

    with open(os.path.join(args.output_dir, "distill_stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "teacher_model": args.teacher_model,
                "student_model": args.student_model,
                "global_step": global_step,
                "finetune_start_eval_loss": init_eval_loss,
                "finetune_start_eval_ppl": init_eval_ppl,
                "epoch_eval_history": epoch_eval_history,
                "final_eval_loss": epoch_eval_history[-1]["eval_loss"] if epoch_eval_history else init_eval_loss,
                "final_eval_ppl": epoch_eval_history[-1]["eval_ppl"] if epoch_eval_history else init_eval_ppl,
                "global_sparsity_masked_params": 1.0 - (total_nonzero / max(1, total)),
                "trainable_sparse_weight_count": masked_weight_count(trainable_masks),
                "param_stats": final_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[INFO] Distilled sparse model saved to: {args.output_dir}")
    print(f"[INFO] Mask file saved to: {out_mask}")


if __name__ == "__main__":
    main()
