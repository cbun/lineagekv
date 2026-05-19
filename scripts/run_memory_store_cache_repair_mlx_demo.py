#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.eval.graders import grade_output
from context_rot.memory import CacheEditPlan, RenderedMemoryPrompt, SQLiteMemoryLedger
from demo_memory_store_cache_repair import seed_demo_memories
from kv_mlx_value_ablation_probe import clone_mlx_cache, generate_from_mlx_cache, prefill_cache, token_ids
from kv_value_ablation_probe import parse_probe_output, remap_parsed_evidence_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a memory-store lineage cache-repair demo on MLX.")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--output", default="results/kv/kv_mlx_memory_store_lineage_demo.json")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--layer-start", type=int, default=8)
    parser.add_argument("--layer-end", type=int, default=23)
    parser.add_argument("--current-value-scale", type=float, default=1.0)
    parser.add_argument("--parse-mode", choices=["json", "json_or_answer"], default="json_or_answer")
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    ledger = SQLiteMemoryLedger(":memory:")
    try:
        seed_demo_memories(ledger)
        query_key = "calendar.weekly_review.location"
        query = "Where should I join the weekly review now?"
        encode = lambda text: token_ids(tokenizer, text)
        full_rendered = ledger.render_prompt(
            query=query,
            queried_memory_key=query_key,
            encode=encode,
            include_history=True,
            neutral_aliases=True,
        )
        current_rendered = ledger.render_prompt(
            query=query,
            queried_memory_key=query_key,
            encode=encode,
            include_history=False,
            neutral_aliases=True,
        )
        plan = ledger.cache_edit_plan(
            full_rendered,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            current_value_scale=args.current_value_scale,
        )
        item = ledger.to_benchmark_item(
            item_id="memory_store_lineage_demo_weekly_review",
            query=query,
            memory_key=query_key,
        )

        full_prefix_ids = token_ids(tokenizer, full_rendered.prefix_text)
        full_suffix_ids = token_ids(tokenizer, full_rendered.suffix_text)
        current_prefix_ids = token_ids(tokenizer, current_rendered.prefix_text)
        current_suffix_ids = token_ids(tokenizer, current_rendered.suffix_text)

        full_prefill, full_prefill_ms = timed_prefill(model, full_prefix_ids)
        current_prefill, current_prefill_ms = timed_prefill(model, current_prefix_ids)

        rows = [
            run_policy(
                policy="full_cache",
                model=model,
                tokenizer=tokenizer,
                item=item,
                rendered=full_rendered,
                prefill=full_prefill,
                suffix_ids=full_suffix_ids,
                prefill_ms=full_prefill_ms,
                max_new_tokens=args.max_new_tokens,
                parse_mode=args.parse_mode,
            ),
            run_policy(
                policy="drop_stale_prompt",
                model=model,
                tokenizer=tokenizer,
                item=item,
                rendered=current_rendered,
                prefill=current_prefill,
                suffix_ids=current_suffix_ids,
                prefill_ms=current_prefill_ms,
                max_new_tokens=args.max_new_tokens,
                parse_mode=args.parse_mode,
            ),
            run_policy(
                policy="ledger_cache_repair",
                model=model,
                tokenizer=tokenizer,
                item=item,
                rendered=full_rendered,
                prefill=full_prefill,
                suffix_ids=full_suffix_ids,
                prefill_ms=full_prefill_ms,
                max_new_tokens=args.max_new_tokens,
                parse_mode=args.parse_mode,
                plan=plan,
            ),
        ]
    finally:
        ledger.close()

    output = {
        "model": f"mlx:{args.model}",
        "query": query,
        "cache_edit_plan": plan.model_dump(),
        "full_rendered": full_rendered.model_dump(),
        "current_only_rendered": current_rendered.model_dump(),
        "item": item.model_dump(),
        "rows": rows,
        "assertions": {
            "ledger_plan_consumed_by_mlx": bool(plan.edits),
            "stale_history_preserved": any(block.id in item.stale_ids for block in item.context_blocks),
            "cache_repair_preserves_full_prefix_tokens": rows[2]["prefix_tokens"] == rows[0]["prefix_tokens"],
            "drop_stale_reduces_prefix_tokens": rows[1]["prefix_tokens"] < rows[0]["prefix_tokens"],
        },
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {output_path}")


def timed_prefill(model: Any, prefix_ids: list[int]) -> tuple[list[Any], float]:
    start = time.perf_counter()
    cache = prefill_cache(model, prefix_ids)
    return cache, (time.perf_counter() - start) * 1000


def run_policy(
    *,
    policy: str,
    model: Any,
    tokenizer: Any,
    item: Any,
    rendered: RenderedMemoryPrompt,
    prefill: list[Any],
    suffix_ids: list[int],
    prefill_ms: float,
    max_new_tokens: int,
    parse_mode: str,
    plan: CacheEditPlan | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    cache = clone_mlx_cache(prefill)
    mx.eval([entry.state for entry in cache])
    cache_clone_ms = (time.perf_counter() - start) * 1000

    edit_ms = 0.0
    if plan is not None:
        start = time.perf_counter()
        apply_ledger_plan(cache, plan)
        mx.eval([entry.state for entry in cache])
        edit_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    raw_output = generate_from_mlx_cache(model, tokenizer, cache, suffix_ids, max_new_tokens)
    decode_ms = (time.perf_counter() - start) * 1000
    parsed = parse_probe_output(raw_output, item, parse_mode)
    parsed = remap_parsed_evidence_ids(parsed, rendered.alias_to_source_id)
    grade = grade_output(item, policy, f"mlx:{getattr(model, 'model_type', 'memory-store-demo')}", parsed)
    return {
        "policy": policy,
        "prefix_tokens": len(token_ids(tokenizer, rendered.prefix_text)),
        "suffix_tokens": len(suffix_ids),
        "prefill_ms": prefill_ms,
        "cache_clone_ms": cache_clone_ms,
        "cache_edit_ms": edit_ms,
        "decode_ms": decode_ms,
        "raw_output": raw_output,
        "parsed_output": parsed,
        **grade.model_dump(),
    }


def apply_ledger_plan(cache: list[Any], plan: CacheEditPlan) -> None:
    for edit in plan.edits:
        for layer_idx in range(edit.layer_start, min(edit.layer_end + 1, len(cache))):
            values = getattr(cache[layer_idx], "values", None)
            if values is None:
                continue
            if edit.zero_values:
                values[..., edit.stale_start : edit.stale_end, :] = mx.zeros_like(
                    values[..., edit.stale_start : edit.stale_end, :]
                )
            if edit.current_value_scale != 1.0:
                values[..., edit.current_start : edit.current_end, :] = (
                    values[..., edit.current_start : edit.current_end, :] * edit.current_value_scale
                )


if __name__ == "__main__":
    main()
