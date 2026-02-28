#!/usr/bin/env python3
import math
from typing import Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer


def loss_to_ppl(loss: float) -> float:
    return math.exp(min(loss, 50.0))


def build_lm_tensor_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    subset: str | None,
    split: str,
    seq_len: int,
    max_samples: int,
    show_progress: bool = True,
    tokenize_batch_size: int = 512,
) -> TensorDataset:
    subset_norm = subset
    if subset_norm is not None and str(subset_norm).strip().lower() in {"", "none", "null"}:
        subset_norm = None
    if subset_norm is None:
        ds = load_dataset(dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name, subset_norm, split=split)

    if "text" not in ds.column_names:
        raise ValueError(
            f"Dataset `{dataset_name}` split `{split}` has no `text` column. "
            f"Available columns: {list(ds.column_names)}"
        )

    target_tokens = max_samples * seq_len if max_samples > 0 else None
    eos_id = tokenizer.eos_token_id
    token_buffer: list[int] = []

    if tokenize_batch_size <= 0:
        raise ValueError("--tokenize_batch_size must be > 0.")

    total_rows = len(ds)
    row_iter = range(0, total_rows, tokenize_batch_size)
    pbar = tqdm(
        row_iter,
        desc=f"Tokenizing {dataset_name}:{split}",
        disable=not show_progress,
    )
    for start in pbar:
        end = min(start + tokenize_batch_size, total_rows)
        batch_texts = ds[start:end]["text"]
        batch_texts = [t for t in batch_texts if isinstance(t, str) and t.strip()]
        if not batch_texts:
            continue

        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )
        for ids in encoded["input_ids"]:
            if not ids:
                continue
            token_buffer.extend(ids)
            if eos_id is not None:
                token_buffer.append(eos_id)

            if target_tokens is not None and len(token_buffer) >= target_tokens:
                break
        if target_tokens is not None and len(token_buffer) >= target_tokens:
            break

    token_count = len(token_buffer)
    num_chunks = token_count // seq_len
    if num_chunks == 0:
        raise ValueError(f"Not enough tokens to create sequence length {seq_len}.")
    if max_samples > 0:
        num_chunks = min(num_chunks, max_samples)

    ids = torch.tensor(token_buffer[: num_chunks * seq_len], dtype=torch.long).reshape(
        num_chunks, seq_len
    )
    return TensorDataset(ids)


@torch.no_grad()
def evaluate_causal_lm_loss_and_ppl_from_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    desc: str = "PPL Eval",
    show_progress: bool = True,
    distributed: bool = False,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    iterator = tqdm(dataloader, desc=desc) if show_progress else dataloader
    for (input_ids,) in iterator:
        input_ids = input_ids.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = float(outputs.loss.item())
        num_target_tokens = input_ids.numel() - input_ids.shape[0]
        total_nll += loss * num_target_tokens
        total_tokens += num_target_tokens

    if distributed and dist.is_available() and dist.is_initialized():
        total_nll_tensor = torch.tensor(total_nll, device=device, dtype=torch.float64)
        total_tokens_tensor = torch.tensor(total_tokens, device=device, dtype=torch.float64)
        dist.all_reduce(total_nll_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
        total_nll = float(total_nll_tensor.item())
        total_tokens = int(total_tokens_tensor.item())

    avg_loss = total_nll / max(1, total_tokens)
    ppl = loss_to_ppl(avg_loss)

    if was_training:
        model.train()
    return avg_loss, ppl


@torch.no_grad()
def evaluate_causal_lm_loss_and_ppl(
    model: torch.nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int = 1,
    desc: str = "PPL Eval",
    show_progress: bool = True,
) -> Tuple[float, float]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return evaluate_causal_lm_loss_and_ppl_from_dataloader(
        model=model,
        dataloader=dataloader,
        device=device,
        desc=desc,
        show_progress=show_progress,
        distributed=False,
    )
