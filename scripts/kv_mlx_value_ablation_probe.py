#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as mlx_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.io import read_jsonl
from context_rot.datasets.schema import BenchmarkItem
from context_rot.eval.graders import grade_output
from kv_value_ablation_probe import (  # type: ignore
    OUTPUT_INSTRUCTION,
    block_id_aliases,
    detected_text_timestamp_pairs,
    parse_probe_output,
    remap_parsed_evidence_ids,
    render_block_parts,
)


@dataclass
class MlxEncodedCase:
    prefix_ids: list[int]
    suffix_ids: list[int]
    block_ranges: dict[str, tuple[int, int]]
    block_aliases: dict[str, str]
    alias_to_block_id: dict[str, str]


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX KV value-ablation probe for stale memory spans.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--distractor-ratio", type=float, default=0.95)
    parser.add_argument("--variants", default="long_context")
    parser.add_argument(
        "--policies",
        default="full_cache,drop_stale_prompt,zero_text_timestamp_conflict_values_layers_8_23_boost_current_factor_1_5",
    )
    parser.add_argument("--render-mode", choices=["ledger", "temporal_blind"], default="temporal_blind")
    parser.add_argument("--parse-mode", choices=["json", "json_or_answer"], default="json_or_answer")
    parser.add_argument("--block-id-mode", choices=["original", "neutral"], default="neutral")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    variants = {value for value in args.variants.split(",") if value}
    matching_items = [
        item
        for item in read_jsonl(ROOT / args.dataset)
        if float(item.metadata.get("distractor_ratio", -1.0)) == args.distractor_ratio
        and str(item.metadata.get("variant", "")) in variants
    ]
    items = matching_items[args.offset : args.offset + args.limit]
    if not items:
        raise SystemExit("No matching items found.")

    model, tokenizer = load(args.model)
    policies = [value for value in args.policies.split(",") if value]

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            encodings: dict[str, MlxEncodedCase] = {
                "full_prefix": encode_case_mlx(
                    tokenizer,
                    item,
                    render_mode=args.render_mode,
                    block_id_mode=args.block_id_mode,
                )
            }
            if "drop_stale_prompt" in policies:
                encodings["drop_stale_prompt"] = encode_case_mlx(
                    tokenizer,
                    item,
                    exclude_block_ids=set(detected_text_timestamp_stale_ids(item)),
                    render_mode=args.render_mode,
                    block_id_mode=args.block_id_mode,
                )
            prefills: dict[str, list[Any]] = {}
            prefill_times: dict[str, float] = {}
            for key, encoded in encodings.items():
                start = time.perf_counter()
                prefills[key] = prefill_cache(model, encoded.prefix_ids)
                prefill_times[key] = (time.perf_counter() - start) * 1000

            for policy in policies:
                encoded_key = policy if policy in encodings and policy != "full_cache" else "full_prefix"
                encoded = encodings[encoded_key]
                start = time.perf_counter()
                cache = clone_mlx_cache(prefills[encoded_key])
                mx.eval([c.state for c in cache])
                cache_clone_ms = (time.perf_counter() - start) * 1000

                edit_ms = 0.0
                if policy != "full_cache" and policy != "drop_stale_prompt":
                    start = time.perf_counter()
                    apply_mlx_policy(cache, encoded, item, policy)
                    mx.eval([c.state for c in cache])
                    edit_ms = (time.perf_counter() - start) * 1000

                start = time.perf_counter()
                raw_output = generate_from_mlx_cache(
                    model,
                    tokenizer,
                    cache,
                    encoded.suffix_ids,
                    args.max_new_tokens,
                )
                decode_ms = (time.perf_counter() - start) * 1000
                parsed = parse_probe_output(raw_output, item, args.parse_mode)
                parsed = remap_parsed_evidence_ids(parsed, encoded.alias_to_block_id)
                model_id = f"mlx:{args.model}"
                grade = grade_output(item, policy, model_id, parsed)
                row = {
                    "item_id": item.id,
                    "domain": item.domain,
                    "question_type": item.question_type,
                    "variant": item.metadata.get("variant"),
                    "distractor_ratio": item.metadata.get("distractor_ratio"),
                    "render_mode": args.render_mode,
                    "prompt_style": "plain",
                    "parse_mode": args.parse_mode,
                    "block_id_mode": args.block_id_mode,
                    "policy": policy,
                    "model_id": model_id,
                    "prefix_tokens": len(encoded.prefix_ids),
                    "full_prefix_tokens": len(encodings["full_prefix"].prefix_ids),
                    "suffix_tokens": len(encoded.suffix_ids),
                    "stale_token_count": token_count_for_ids(encoded, detected_text_timestamp_stale_ids(item)),
                    "gold_token_count": token_count_for_ids(encoded, item.gold_evidence_ids),
                    "latency_ms": decode_ms,
                    "prefill_ms": prefill_times[encoded_key],
                    "full_prefill_ms": prefill_times["full_prefix"],
                    "cache_clone_ms": cache_clone_ms,
                    "cache_edit_ms": edit_ms,
                    "decode_ms": decode_ms,
                    "raw_output": raw_output,
                    "parsed_output": parsed,
                    **grade.model_dump(),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    print(f"wrote {output_path}")


def encode_case_mlx(
    tokenizer: Any,
    item: BenchmarkItem,
    exclude_block_ids: set[str] | None = None,
    render_mode: str = "temporal_blind",
    block_id_mode: str = "neutral",
) -> MlxEncodedCase:
    exclude_block_ids = exclude_block_ids or set()
    prefix_start = (
        "You answer questions from a cached agent memory prefix.\n"
        "Resolve conflicts using the memory contents and timestamps.\n"
        "Context blocks:\n"
    )
    prefix_ids = token_ids(tokenizer, prefix_start)
    block_ranges: dict[str, tuple[int, int]] = {}
    block_aliases = block_id_aliases(item, block_id_mode)
    visible_block_aliases: dict[str, str] = {}
    for block in item.context_blocks:
        if block.id in exclude_block_ids:
            continue
        visible_block_aliases[block.id] = block_aliases[block.id]
        header, body, suffix = render_block_parts(block, render_mode, rendered_block_id=block_aliases[block.id])
        block_ids = token_ids(tokenizer, f"{header}{body}{suffix}")
        start = len(prefix_ids)
        prefix_ids.extend(block_ids)
        block_ranges[block.id] = (start, len(prefix_ids))
    suffix_ids = token_ids(tokenizer, f"\nQuestion: {item.query}\n{OUTPUT_INSTRUCTION}\nJSON:")
    return MlxEncodedCase(
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        block_ranges=block_ranges,
        block_aliases=visible_block_aliases,
        alias_to_block_id={alias: block_id for block_id, alias in visible_block_aliases.items()},
    )


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def detected_text_timestamp_stale_ids(item: BenchmarkItem) -> list[str]:
    return [stale_id for stale_id, _ in detected_text_timestamp_pairs(item)]


def prefill_cache(model: Any, prefix_ids: list[int]) -> list[Any]:
    cache = mlx_cache.make_prompt_cache(model)
    with mx.stream(mx.default_stream(mx.default_device())):
        _ = model(mx.array(prefix_ids, dtype=mx.uint32)[None], cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def clone_mlx_cache(cache: list[Any]) -> list[Any]:
    cloned: list[Any] = []
    for entry in cache:
        new_entry = type(entry)()
        if hasattr(entry, "state"):
            state = entry.state
            if isinstance(state, tuple) and len(state) == 2:
                new_entry.state = tuple(clone_array(value) for value in state)
            else:
                new_entry.state = state
        cloned.append(new_entry)
    return cloned


def clone_array(value: Any) -> Any:
    return value + mx.zeros_like(value)


def apply_mlx_policy(cache: list[Any], encoded: MlxEncodedCase, item: BenchmarkItem, policy: str) -> None:
    match = re.fullmatch(
        r"zero_text_timestamp_conflict_values_layers_(\d+)_(\d+)_boost_current_factor_([0-9]+(?:_[0-9]+)?)",
        policy,
    )
    if not match:
        raise KeyError(f"Unsupported MLX policy: {policy}")
    start_layer = int(match.group(1))
    end_layer = int(match.group(2))
    factor = float(match.group(3).replace("_", "."))
    for stale_id, current_id in detected_text_timestamp_pairs(item):
        zero_block_values(cache, encoded, stale_id, start_layer, end_layer)
        scale_block_values(cache, encoded, current_id, start_layer, end_layer, factor)


def zero_block_values(cache: list[Any], encoded: MlxEncodedCase, block_id: str, start_layer: int, end_layer: int) -> None:
    if block_id not in encoded.block_ranges:
        return
    start, end = encoded.block_ranges[block_id]
    for layer_idx in range(start_layer, min(end_layer + 1, len(cache))):
        values = getattr(cache[layer_idx], "values", None)
        if values is not None:
            values[..., start:end, :] = mx.zeros_like(values[..., start:end, :])


def scale_block_values(
    cache: list[Any],
    encoded: MlxEncodedCase,
    block_id: str,
    start_layer: int,
    end_layer: int,
    factor: float,
) -> None:
    if block_id not in encoded.block_ranges:
        return
    start, end = encoded.block_ranges[block_id]
    for layer_idx in range(start_layer, min(end_layer + 1, len(cache))):
        values = getattr(cache[layer_idx], "values", None)
        if values is not None:
            values[..., start:end, :] = values[..., start:end, :] * factor


def generate_from_mlx_cache(
    model: Any,
    tokenizer: Any,
    cache: list[Any],
    suffix_ids: list[int],
    max_new_tokens: int,
) -> str:
    logits = model(mx.array(suffix_ids, dtype=mx.uint32)[None], cache=cache)
    mx.eval(logits, [c.state for c in cache])
    next_logits = logits[:, -1, :]
    generated: list[int] = []
    eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    for _ in range(max_new_tokens):
        next_id = int(mx.argmax(next_logits, axis=-1).item())
        if next_id in eos_ids:
            break
        generated.append(next_id)
        decoded = tokenizer.decode(generated)
        if "}" in decoded and "{" in decoded and decoded.rfind("}") > decoded.find("{"):
            break
        logits = model(mx.array([next_id], dtype=mx.uint32)[None], cache=cache)
        mx.eval(logits, [c.state for c in cache])
        next_logits = logits[:, -1, :]
    return tokenizer.decode(generated)


def token_count_for_ids(encoded: MlxEncodedCase, block_ids: list[str]) -> int:
    total = 0
    for block_id in block_ids:
        if block_id in encoded.block_ranges:
            start, end = encoded.block_ranges[block_id]
            total += end - start
    return total


if __name__ == "__main__":
    main()
