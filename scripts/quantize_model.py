#!/usr/bin/env python3
import argparse
import json
import os
from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ppl_utils import build_lm_tensor_dataset, evaluate_causal_lm_loss_and_ppl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a distilled sparse model to INT8 or FP8.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--format", type=str, choices=["int8", "fp8"], default="int8")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--skip_ppl_eval", action="store_true")
    parser.add_argument("--ppl_dataset", type=str, default="wikitext")
    parser.add_argument("--ppl_subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--ppl_split", type=str, default="test")
    parser.add_argument("--ppl_seq_len", type=int, default=512)
    parser.add_argument("--ppl_max_samples", type=int, default=256)
    parser.add_argument("--ppl_batch_size", type=int, default=1)
    return parser.parse_args()


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


def try_quanto_quantize(
    model: torch.nn.Module,
    quant_format: str,
) -> Tuple[bool, str]:
    try:
        from optimum.quanto import freeze, qfloat8, qint8, quantize
    except Exception as exc:  # pragma: no cover
        return False, f"optimum-quanto import failed: {exc}"

    try:
        qtype = qint8 if quant_format == "int8" else qfloat8
        quantize(model, weights=qtype)
        freeze(model)
        return True, "quantized by optimum-quanto"
    except Exception as exc:  # pragma: no cover
        return False, f"optimum-quanto quantization failed: {exc}"


def int8_dynamic_fallback(model: torch.nn.Module) -> torch.nn.Module:
    model_cpu = model.cpu().eval()
    qmodel = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return qmodel


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = adapt_dtype_for_device(resolve_dtype(args.dtype), device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()

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

    success, msg = try_quanto_quantize(model, quant_format=args.format)
    quant_backend = "optimum-quanto" if success else "none"

    if success:
        quant_ppl = None
        quant_loss = None
        if ppl_eval_dataset is not None:
            eval_device = torch.device("cpu")
            try:
                model = model.to(eval_device)
                quant_loss, quant_ppl = evaluate_causal_lm_loss_and_ppl(
                    model=model,
                    dataset=ppl_eval_dataset,
                    device=eval_device,
                    batch_size=args.ppl_batch_size,
                    desc=f"PPL after {args.format} quant",
                    show_progress=True,
                )
                print(f"[PPL] after_quantization format={args.format} loss={quant_loss:.4f} ppl={quant_ppl:.4f}")
            except Exception as exc:
                print(f"[WARN] Failed to evaluate PPL for quantized model: {exc}")

        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "quantize_stats.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": args.format,
                    "backend": quant_backend,
                    "message": msg,
                    "source_model": args.model_dir,
                    "ppl_after_quantization": quant_ppl,
                    "loss_after_quantization": quant_loss,
                },
                f,
                indent=2,
            )
        print(f"[INFO] Saved {args.format} quantized model to: {args.output_dir}")
        print(f"[INFO] Backend: {quant_backend}")
        return

    if args.format == "fp8":
        raise RuntimeError(
            f"FP8 quantization requires optimum-quanto in this implementation. Error: {msg}"
        )

    print(f"[WARN] {msg}")
    print("[WARN] Falling back to torch dynamic INT8 quantization (Linear-only, CPU inference).")
    qmodel = int8_dynamic_fallback(model)
    quant_ppl = None
    quant_loss = None
    if ppl_eval_dataset is not None:
        try:
            quant_loss, quant_ppl = evaluate_causal_lm_loss_and_ppl(
                model=qmodel,
                dataset=ppl_eval_dataset,
                device=torch.device("cpu"),
                batch_size=args.ppl_batch_size,
                desc="PPL after int8 quant (dynamic fallback)",
                show_progress=True,
            )
            print(f"[PPL] after_quantization format=int8 loss={quant_loss:.4f} ppl={quant_ppl:.4f}")
        except Exception as exc:
            print(f"[WARN] Failed to evaluate PPL for int8 fallback model: {exc}")

    torch.save(qmodel.state_dict(), os.path.join(args.output_dir, "model_int8_dynamic.pt"))
    config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "quantize_stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "int8",
                "backend": "torch_dynamic_fallback",
                "message": msg,
                "source_model": args.model_dir,
                "artifact": "model_int8_dynamic.pt",
                "ppl_after_quantization": quant_ppl,
                "loss_after_quantization": quant_loss,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Saved fallback INT8 artifact to: {args.output_dir}/model_int8_dynamic.pt")


if __name__ == "__main__":
    main()
