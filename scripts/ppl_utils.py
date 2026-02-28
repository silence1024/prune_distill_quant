#!/usr/bin/env python3
import math
import re
from typing import Any, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset, load_dataset_builder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer


def loss_to_ppl(loss: float) -> float:
    return math.exp(min(loss, 50.0))


def _normalize_subset(subset: str | None) -> str | None:
    subset_norm = subset
    if subset_norm is not None and str(subset_norm).strip().lower() in {"", "none", "null"}:
        subset_norm = None
    return subset_norm


def _load_dataset_split(
    dataset_name: str,
    subset_norm: str | None,
    split: str,
    streaming: bool,
):
    if not streaming:
        if subset_norm is None:
            return load_dataset(dataset_name, split=split, streaming=False)
        return load_dataset(dataset_name, subset_norm, split=split, streaming=False)

    # Streaming backend may reject split slicing like train[:99%], so resolve via skip/take.
    split_norm = split.strip()
    m = re.match(r"^([^\[\]]+)\[([^\[\]]+)\]$", split_norm)
    if m is None:
        if subset_norm is None:
            return load_dataset(dataset_name, split=split_norm, streaming=True)
        return load_dataset(dataset_name, subset_norm, split=split_norm, streaming=True)

    base_split = m.group(1).strip()
    slice_expr = m.group(2).strip()
    if ":" not in slice_expr:
        raise ValueError(
            f"Unsupported split expression for streaming: `{split}`. Expected forms like "
            "`train[:99%]`, `train[99%:]`, or `train[1000:2000]`."
        )

    if subset_norm is None:
        builder = load_dataset_builder(dataset_name)
    else:
        builder = load_dataset_builder(dataset_name, subset_norm)
    split_info = builder.info.splits.get(base_split)
    if split_info is None:
        raise ValueError(
            f"Bad split: {split}. Available splits: {list(builder.info.splits.keys())}"
        )
    total_rows = split_info.num_examples
    if total_rows is None:
        raise ValueError(
            "Unable to resolve streaming split slice because total row count is unknown "
            f"for split `{base_split}`."
        )

    def parse_bound(spec: str, default: int) -> int:
        token = spec.strip()
        if token == "":
            return default
        if token.endswith("%"):
            ratio = float(token[:-1]) / 100.0
            return int(total_rows * ratio)
        return int(token)

    left_spec, right_spec = slice_expr.split(":", 1)
    start = parse_bound(left_spec, 0)
    end = parse_bound(right_spec, int(total_rows))
    start = max(0, min(int(total_rows), start))
    end = max(0, min(int(total_rows), end))
    if end < start:
        raise ValueError(f"Invalid split slice `{split}`: end < start.")

    if subset_norm is None:
        ds = load_dataset(dataset_name, split=base_split, streaming=True)
    else:
        ds = load_dataset(dataset_name, subset_norm, split=base_split, streaming=True)
    if start > 0:
        ds = ds.skip(start)
    if end - start >= 0:
        ds = ds.take(end - start)
    return ds


def _build_tensor_dataset_from_loaded_dataset(
    ds: Any,
    tokenizer: AutoTokenizer,
    seq_len: int,
    max_samples: int,
    show_progress: bool = True,
    tokenize_batch_size: int = 512,
    streaming: bool = False,
    progress_desc: str = "Tokenizing",
) -> TensorDataset:
    column_names = getattr(ds, "column_names", None) or []
    if "text" not in column_names:
        raise ValueError(
            "Dataset has no `text` column. "
            f"Available columns: {list(column_names)}"
        )

    target_tokens = max_samples * seq_len if max_samples > 0 else None
    eos_id = tokenizer.eos_token_id
    token_buffer: list[int] = []

    if tokenize_batch_size <= 0:
        raise ValueError("--tokenize_batch_size must be > 0.")

    if streaming:
        processed_chunks = 0
        pbar = tqdm(
            total=max_samples if max_samples > 0 else None,
            desc=progress_desc,
            disable=not show_progress,
        )
        text_batch: list[str] = []
        stop = False

        def consume_text_batch(batch: list[str]) -> None:
            nonlocal processed_chunks, stop
            if not batch:
                return
            encoded = tokenizer(
                batch,
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
                    stop = True
                    break
            if max_samples > 0:
                ready_chunks = min(max_samples, len(token_buffer) // seq_len)
            else:
                ready_chunks = len(token_buffer) // seq_len
            if ready_chunks > processed_chunks:
                pbar.update(ready_chunks - processed_chunks)
                processed_chunks = ready_chunks

        for row in ds:
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            text_batch.append(text)
            if len(text_batch) >= tokenize_batch_size:
                consume_text_batch(text_batch)
                text_batch = []
            if stop:
                break
        if not stop and text_batch:
            consume_text_batch(text_batch)
        pbar.close()
    else:
        total_rows = len(ds)
        row_iter = range(0, total_rows, tokenize_batch_size)
        pbar = tqdm(
            row_iter,
            desc=progress_desc,
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


def build_lm_tensor_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    subset: str | None,
    split: str,
    seq_len: int,
    max_samples: int,
    show_progress: bool = True,
    tokenize_batch_size: int = 512,
    streaming: bool = False,
) -> TensorDataset:
    subset_norm = _normalize_subset(subset)
    ds = _load_dataset_split(
        dataset_name=dataset_name,
        subset_norm=subset_norm,
        split=split,
        streaming=streaming,
    )
    return _build_tensor_dataset_from_loaded_dataset(
        ds=ds,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_samples=max_samples,
        show_progress=show_progress,
        tokenize_batch_size=tokenize_batch_size,
        streaming=streaming,
        progress_desc=(
            f"Tokenizing(stream) {dataset_name}:{split}"
            if streaming
            else f"Tokenizing {dataset_name}:{split}"
        ),
    )


def build_lm_tensor_dataset_from_hf_dataset(
    tokenizer: AutoTokenizer,
    ds: Any,
    seq_len: int,
    max_samples: int,
    show_progress: bool = True,
    tokenize_batch_size: int = 512,
    streaming: bool = False,
    progress_desc: str = "Tokenizing",
) -> TensorDataset:
    return _build_tensor_dataset_from_loaded_dataset(
        ds=ds,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_samples=max_samples,
        show_progress=show_progress,
        tokenize_batch_size=tokenize_batch_size,
        streaming=streaming,
        progress_desc=progress_desc,
    )


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
