#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ppl_utils import build_lm_tensor_dataset, evaluate_causal_lm_loss_and_ppl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SparseGPT-style one-shot pruning for causal LMs."
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sparsity", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib_dataset", type=str, default="openwebtext")
    parser.add_argument("--calib_subset", type=str, default="none")
    parser.add_argument("--calib_split", type=str, default="train[:1%]")
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_hessian_rows", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
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
    parser.add_argument("--include_lm_head", action="store_true")
    parser.add_argument("--skip_ppl_eval", action="store_true")
    parser.add_argument("--ppl_dataset", type=str, default="openwebtext")
    parser.add_argument("--ppl_subset", type=str, default="none")
    parser.add_argument("--ppl_split", type=str, default="train[99%:]")
    parser.add_argument("--ppl_seq_len", type=int, default=512)
    parser.add_argument("--ppl_max_samples", type=int, default=256)
    parser.add_argument("--ppl_batch_size", type=int, default=1)
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


def build_calibration_samples(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    subset: str | None,
    split: str,
    nsamples: int,
    seq_len: int,
    seed: int,
) -> List[torch.Tensor]:
    subset_norm = subset
    if subset_norm is not None and str(subset_norm).strip().lower() in {"", "none", "null"}:
        subset_norm = None
    if subset_norm is None:
        ds = load_dataset(dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name, subset_norm, split=split)
    if "text" not in ds.column_names:
        raise ValueError(
            f"Calibration dataset `{dataset_name}` split `{split}` has no `text` column. "
            f"Available columns: {list(ds.column_names)}"
        )

    num_rows = len(ds)
    if num_rows == 0:
        raise ValueError(f"Calibration dataset `{dataset_name}` split `{split}` is empty.")

    eos_id = tokenizer.eos_token_id
    rng = random.Random(seed)
    samples: List[torch.Tensor] = []
    max_attempts = max(1000, nsamples * 64)
    attempts = 0

    while len(samples) < nsamples and attempts < max_attempts:
        attempts += 1
        idx = rng.randrange(num_rows)
        text = ds[int(idx)]["text"]
        if not isinstance(text, str) or not text.strip():
            continue

        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )["input_ids"]
        if eos_id is not None:
            token_ids = token_ids + [eos_id]
        if len(token_ids) < seq_len + 1:
            continue

        start = rng.randint(0, len(token_ids) - seq_len - 1)
        chunk = torch.tensor(token_ids[start : start + seq_len], dtype=torch.long)
        samples.append(chunk.unsqueeze(0))

    if len(samples) < nsamples:
        raise ValueError(
            "Unable to build enough calibration samples from the current dataset settings. "
            f"Requested={nsamples}, collected={len(samples)}. "
            "Try smaller --seq_len, larger/more diverse --calib_split, or different dataset."
        )

    return samples


def find_linear_layers(model: nn.Module, include_lm_head: bool) -> Iterable[Tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not include_lm_head and name.endswith("lm_head"):
                continue
            yield name, module


@torch.no_grad()
def collect_inputs_for_layer(
    model: nn.Module,
    layer: nn.Linear,
    calib_samples: List[torch.Tensor],
    device: torch.device,
    max_rows: int,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    row_count = 0

    def hook_fn(_module: nn.Module, inputs: Tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        nonlocal row_count
        x = inputs[0].detach()
        x = x.reshape(-1, x.shape[-1]).float().cpu()
        rows.append(x)
        row_count += x.shape[0]

    handle = layer.register_forward_hook(hook_fn)
    model.eval()
    for sample in calib_samples:
        _ = model(input_ids=sample.to(device))
        if row_count >= max_rows:
            break
    handle.remove()

    if not rows:
        raise RuntimeError("No activations were collected for a linear layer.")
    x_all = torch.cat(rows, dim=0)
    if x_all.shape[0] > max_rows:
        x_all = x_all[:max_rows]
    return x_all


def invert_hessian(hessian: torch.Tensor, percdamp: float) -> torch.Tensor:
    n = hessian.shape[0]
    damp = percdamp * torch.mean(torch.diag(hessian))
    hessian = hessian + torch.eye(n, device=hessian.device, dtype=hessian.dtype) * damp

    jitter = 1e-6
    for _ in range(6):
        try:
            chol = torch.linalg.cholesky(hessian)
            return torch.cholesky_inverse(chol)
        except RuntimeError:
            hessian = hessian + torch.eye(n, device=hessian.device, dtype=hessian.dtype) * jitter
            jitter *= 10
    raise RuntimeError("Failed to invert Hessian; try increasing --percdamp.")


def sparsegpt_prune_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    sparsity: float,
    block_size: int,
    percdamp: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = weight.float().clone()
    h = hessian.float().clone()

    dead = torch.diag(h) == 0
    if dead.any():
        h[dead, dead] = 1
        w[:, dead] = 0

    h_inv = invert_hessian(h, percdamp=percdamp)
    n_cols = w.shape[1]
    out = w.clone()
    mask = torch.zeros_like(w, dtype=torch.bool)

    for start in range(0, n_cols, block_size):
        end = min(start + block_size, n_cols)
        w_block = out[:, start:end]
        h_inv_block = h_inv[start:end, start:end]
        d = torch.diag(h_inv_block).clamp_min(1e-8).unsqueeze(0)

        keep_per_row = max(1, int(round((1.0 - sparsity) * w_block.shape[1])))
        if keep_per_row >= w_block.shape[1]:
            m_block = torch.ones_like(w_block, dtype=torch.bool)
            q_block = w_block
        else:
            scores = (w_block**2) / (d**2)
            topk_idx = torch.topk(scores, k=keep_per_row, dim=1, largest=True).indices
            m_block = torch.zeros_like(w_block, dtype=torch.bool)
            m_block.scatter_(1, topk_idx, True)
            q_block = torch.where(m_block, w_block, torch.zeros_like(w_block))

        err = (w_block - q_block) / d
        if end < n_cols:
            out[:, end:] -= err @ h_inv[start:end, end:]

        out[:, start:end] = q_block
        mask[:, start:end] = m_block

    return out, mask


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = adapt_dtype_for_device(resolve_dtype(args.dtype), device)
    print(f"[INFO] Loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    ppl_before_pruning = None
    ppl_after_pruning = None
    ppl_loss_before = None
    ppl_loss_after = None

    ppl_eval_dataset = None
    if not args.skip_ppl_eval:
        print("[INFO] Building PPL eval dataset...")
        ppl_eval_dataset = build_lm_tensor_dataset(
            tokenizer=tokenizer,
            dataset_name=args.ppl_dataset,
            subset=args.ppl_subset,
            split=args.ppl_split,
            seq_len=args.ppl_seq_len,
            max_samples=args.ppl_max_samples,
        )
        print("[INFO] Evaluating PPL before pruning...")
        ppl_loss_before, ppl_before_pruning = evaluate_causal_lm_loss_and_ppl(
            model=model,
            dataset=ppl_eval_dataset,
            device=device,
            batch_size=args.ppl_batch_size,
            desc="PPL before pruning",
            show_progress=True,
        )
        print(f"[PPL] before_pruning loss={ppl_loss_before:.4f} ppl={ppl_before_pruning:.4f}")

    print("[INFO] Building calibration samples...")
    calib_samples = build_calibration_samples(
        tokenizer=tokenizer,
        dataset_name=args.calib_dataset,
        subset=args.calib_subset,
        split=args.calib_split,
        nsamples=args.nsamples,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    print(f"[INFO] Calibration samples: {len(calib_samples)}")

    layer_masks: Dict[str, torch.Tensor] = {}
    total_elems = 0
    total_nonzero = 0
    layer_stats = []

    linear_layers = list(find_linear_layers(model, include_lm_head=args.include_lm_head))
    print(f"[INFO] Linear layers to prune: {len(linear_layers)}")

    for i, (name, layer) in enumerate(linear_layers, start=1):
        print(f"[INFO] ({i}/{len(linear_layers)}) Collecting activations for: {name}")
        x = collect_inputs_for_layer(
            model=model,
            layer=layer,
            calib_samples=calib_samples,
            device=device,
            max_rows=args.max_hessian_rows,
        )
        x = x.to(device=device, dtype=torch.float32)
        h = (x.t() @ x) / x.shape[0]
        del x
        torch.cuda.empty_cache()

        print(f"[INFO] ({i}/{len(linear_layers)}) Pruning layer: {name}")
        pruned_w, mask = sparsegpt_prune_weight(
            weight=layer.weight.data.to(torch.float32),
            hessian=h,
            sparsity=args.sparsity,
            block_size=args.block_size,
            percdamp=args.percdamp,
        )
        layer.weight.data.copy_(pruned_w.to(dtype=layer.weight.dtype))
        layer_masks[f"{name}.weight"] = mask.cpu()

        elems = mask.numel()
        nonzero = int(mask.sum().item())
        total_elems += elems
        total_nonzero += nonzero
        current_sparsity = 1.0 - (nonzero / elems)
        layer_stats.append(
            {
                "layer": name,
                "elements": elems,
                "nonzero": nonzero,
                "sparsity": current_sparsity,
            }
        )
        print(f"[INFO] Layer sparsity={current_sparsity:.4f}")

    global_sparsity = 1.0 - (total_nonzero / max(1, total_elems))
    print(f"[INFO] Global sparsity over pruned layers: {global_sparsity:.4f}")

    if not args.skip_ppl_eval and ppl_eval_dataset is not None:
        print("[INFO] Evaluating PPL after pruning...")
        ppl_loss_after, ppl_after_pruning = evaluate_causal_lm_loss_and_ppl(
            model=model,
            dataset=ppl_eval_dataset,
            device=device,
            batch_size=args.ppl_batch_size,
            desc="PPL after pruning",
            show_progress=True,
        )
        print(f"[PPL] after_pruning loss={ppl_loss_after:.4f} ppl={ppl_after_pruning:.4f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    mask_path = os.path.join(args.output_dir, "sparsity_mask.pt")
    torch.save(layer_masks, mask_path)

    with open(os.path.join(args.output_dir, "prune_stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": args.model_id,
                "target_sparsity": args.sparsity,
                "global_sparsity": global_sparsity,
                "num_linear_layers": len(linear_layers),
                "layer_stats": layer_stats,
                "ppl_before_pruning": ppl_before_pruning,
                "ppl_after_pruning": ppl_after_pruning,
                "ppl_loss_before_pruning": ppl_loss_before,
                "ppl_loss_after_pruning": ppl_loss_after,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Saved pruned model + mask to: {args.output_dir}")
    print(f"[INFO] Mask file: {mask_path}")


if __name__ == "__main__":
    main()
