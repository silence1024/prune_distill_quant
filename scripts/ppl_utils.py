#!/usr/bin/env python3
import math
from typing import Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer


def loss_to_ppl(loss: float) -> float:
    return math.exp(min(loss, 50.0))


def build_lm_tensor_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    subset: str,
    split: str,
    seq_len: int,
    max_samples: int,
) -> TensorDataset:
    ds = load_dataset(dataset_name, subset, split=split)
    text = "\n\n".join([t for t in ds["text"] if t and not t.isspace()])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]

    num_chunks = ids.numel() // seq_len
    if num_chunks == 0:
        raise ValueError(f"Not enough tokens to create sequence length {seq_len}.")
    if max_samples > 0:
        num_chunks = min(num_chunks, max_samples)

    ids = ids[: num_chunks * seq_len].reshape(num_chunks, seq_len)
    return TensorDataset(ids)


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

    avg_loss = total_nll / max(1, total_tokens)
    ppl = loss_to_ppl(avg_loss)

    if was_training:
        model.train()
    return avg_loss, ppl
