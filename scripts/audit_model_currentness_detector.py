#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.io import read_jsonl
from context_rot.datasets.schema import BenchmarkItem
from context_rot.eval.graders import parse_model_json
from kv_value_ablation_probe import parse_torch_dtype, resolve_device, sync_torch_device, text_timestamp_candidate_blocks


INSTRUCTION = """You are auditing assistant memory records.
Use only the block text and timestamps. A stale block is an older memory record for the same requested memory key when a newer conflicting record exists.
Return JSON only with keys: stale_ids, current_ids, abstain, confidence, reasoning_summary.
Use exact block IDs. If the currentness relation is unclear, set abstain to true and leave stale_ids empty."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a model-based stale/current memory detector.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--distractor-ratio", type=float, default=0.5)
    parser.add_argument("--variants", default="current_first,stale_first,chronological,scrambled")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--prompt-style", choices=["plain", "chat"], default="plain")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    variants = {value for value in args.variants.split(",") if value}
    items = [
        item
        for item in read_jsonl(ROOT / args.dataset)
        if float(item.metadata.get("distractor_ratio", -1.0)) == args.distractor_ratio
        and str(item.metadata.get("variant", "")) in variants
    ]
    items = items[args.offset : args.offset + args.limit if args.limit is not None else None]
    if not items:
        raise SystemExit("No matching items found.")

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model_kwargs: dict[str, Any] = {"local_files_only": True, "dtype": parse_torch_dtype(args.dtype)}
    if args.low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if device.type != "cpu":
        model.to(device)
    model.eval()

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    totals = {"items": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0, "abstain": 0}
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "item_id",
            "gold_ids",
            "predicted_ids",
            "tp",
            "fp",
            "fn",
            "exact_match",
            "abstain",
            "confidence",
            "latency_ms",
            "raw_output",
            "reasoning_summary",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            started = time.perf_counter()
            raw_output = generate_detection(model, tokenizer, item, args.prompt_style, args.max_new_tokens, device)
            sync_torch_device(device)
            latency_ms = (time.perf_counter() - started) * 1000
            parsed = parse_model_json(raw_output)
            parsed = parsed if isinstance(parsed, dict) else {}
            parsed = supplement_currentness_fields(parsed, raw_output)
            predicted = normalize_id_list(parsed.get("stale_ids"))
            if not predicted:
                predicted = stale_ids_from_current_ids(item, normalize_id_list(parsed.get("current_ids")))
            gold = set(item.stale_ids)
            predicted_set = set(predicted)
            tp = len(predicted_set & gold)
            fp = len(predicted_set - gold)
            fn = len(gold - predicted_set)
            exact = predicted_set == gold
            abstain = bool(parsed.get("abstain", False)) or (not predicted and not normalize_id_list(parsed.get("current_ids")))
            row = {
                "item_id": item.id,
                "gold_ids": " ".join(sorted(gold)),
                "predicted_ids": " ".join(sorted(predicted_set)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "exact_match": exact,
                "abstain": abstain,
                "confidence": parsed.get("confidence", ""),
                "latency_ms": latency_ms,
                "raw_output": raw_output,
                "reasoning_summary": parsed.get("reasoning_summary", ""),
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            totals["items"] += 1
            totals["exact"] += int(exact)
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            totals["abstain"] += int(abstain)

    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    exact_rate = totals["exact"] / max(1, totals["items"])
    abstain_rate = totals["abstain"] / max(1, totals["items"])
    print(
        "items",
        totals["items"],
        "precision",
        f"{precision:.3f}",
        "recall",
        f"{recall:.3f}",
        "exact",
        f"{exact_rate:.3f}",
        "abstain",
        f"{abstain_rate:.3f}",
        "tp/fp/fn",
        f"{totals['tp']}/{totals['fp']}/{totals['fn']}",
    )
    print(f"wrote {output_path}")


def generate_detection(
    model: Any,
    tokenizer: Any,
    item: BenchmarkItem,
    prompt_style: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    prompt = render_detection_prompt(item)
    if prompt_style == "chat":
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("prompt-style=chat requires tokenizer chat_template")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def render_detection_prompt(item: BenchmarkItem) -> str:
    lines = [INSTRUCTION, "", f"Question: {item.query}", "", "Blocks:"]
    for block in item.context_blocks:
        lines.append(f"[block_id={block.id}; timestamp={block.timestamp}; type={block.type}]")
        lines.append(block.text)
    lines.append("")
    lines.append("JSON:")
    return "\n".join(lines)


def normalize_id_list(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = re.split(r"[\s,]+", value.strip())
    elif isinstance(value, list):
        parts = [str(part) for part in value]
    else:
        parts = []
    return [part for part in parts if part]


def supplement_currentness_fields(parsed: dict[str, Any], raw_output: str) -> dict[str, Any]:
    supplemented = dict(parsed)
    for field in ["stale_ids", "current_ids"]:
        if supplemented.get(field):
            continue
        match = re.search(rf"\b{field}\s*:\s*\[([^\]]*)\]", raw_output, flags=re.IGNORECASE)
        if match:
            supplemented[field] = re.findall(r"[A-Za-z0-9_.:-]+", match.group(1))
    return supplemented


def stale_ids_from_current_ids(item: BenchmarkItem, current_ids: list[str]) -> list[str]:
    current_set = set(current_ids)
    if not current_set:
        return []
    candidate_ids = [block.id for block in text_timestamp_candidate_blocks(item)]
    if not candidate_ids:
        return []
    if not current_set <= set(candidate_ids):
        return []
    return [block_id for block_id in candidate_ids if block_id not in current_set]


if __name__ == "__main__":
    main()
