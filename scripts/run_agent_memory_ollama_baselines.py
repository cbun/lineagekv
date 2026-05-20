#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl
from context_rot.datasets.schema import BenchmarkItem, ContextBlock, CompressionResult, ContextBundle
from context_rot.eval.graders import grade_output, parse_model_json
from context_rot.models.ollama_adapter import OllamaModel


OUTPUT_INSTRUCTION = (
    'Return JSON only with keys: "answer", "evidence_ids", "used_stale_fact", '
    '"confidence", "abstain", "reasoning_summary". Put context block IDs only in evidence_ids. '
    "Keep reasoning_summary under 8 words."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run black-box Ollama prompt baselines on agent-memory items.")
    parser.add_argument("--dataset", default="data/generated/agent_memory_action_test32.jsonl")
    parser.add_argument("--output", default="results/kv/kv_ollama_agent_memory_action_baselines.jsonl")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--models", default="qwen3.5:27b,gemma3n:e2b")
    parser.add_argument("--policies", default="full_cache,drop_stale_prompt")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    items = read_jsonl(ROOT / args.dataset)[args.offset : args.offset + args.limit]
    models = [value for value in args.models.split(",") if value]
    policies = [value for value in args.policies.split(",") if value]
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for model_name in models:
            model = OllamaModel(model_name, max_tokens=args.max_tokens, timeout_seconds=args.timeout_seconds)
            for item in items:
                aliases = block_id_aliases(item)
                for policy in policies:
                    rendered_blocks = policy_blocks(item, policy)
                    prompt = render_prompt(item, rendered_blocks, aliases)
                    compression = compression_result(item, policy, rendered_blocks, prompt)
                    started = time.perf_counter()
                    raw_output = model.generate(prompt, item, compression)
                    latency_ms = (time.perf_counter() - started) * 1000
                    parsed = normalize_parsed_fields(parse_model_json(raw_output))
                    parsed = remap_evidence_ids(parsed, aliases)
                    grade = grade_output(item, policy, f"ollama:{model_name}", parsed)
                    row = {
                        "item_id": item.id,
                        "domain": item.domain,
                        "question_type": item.question_type,
                        "policy": policy,
                        "model_id": f"ollama:{model_name}",
                        "latency_ms": latency_ms,
                        "prompt_tokens_estimate": estimate_tokens(prompt),
                        "raw_output": raw_output,
                        "parsed_output": parsed,
                        **grade.model_dump(),
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        model_name,
                        policy,
                        item.id,
                        "correct",
                        row["correct"],
                        "stale",
                        row["used_stale_fact"],
                    )
    summary_path = ROOT / (args.summary_output or default_summary_path(args.output))
    write_summary(summary_path, summarize(rows))
    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")


def policy_blocks(item: BenchmarkItem, policy: str) -> list[ContextBlock]:
    if policy == "full_cache":
        return item.context_blocks
    if policy == "drop_stale_prompt":
        stale = set(item.stale_ids)
        return [block for block in item.context_blocks if block.id not in stale]
    raise KeyError(policy)


def render_prompt(item: BenchmarkItem, blocks: list[ContextBlock], aliases: dict[str, str]) -> str:
    lines = [
        "You answer questions from a cached agent memory prefix.",
        "Resolve conflicts using the memory contents and timestamps.",
        "Context blocks:",
    ]
    for block in blocks:
        lines.append(f"[block_id={aliases[block.id]}; timestamp={block.timestamp}; type={block.type}]")
        lines.append(block.text)
    lines.extend(["", f"Question: {item.query}", OUTPUT_INSTRUCTION, "JSON:"])
    return "\n".join(lines)


def block_id_aliases(item: BenchmarkItem) -> dict[str, str]:
    return {block.id: f"m{idx:03d}" for idx, block in enumerate(item.context_blocks)}


def remap_evidence_ids(parsed: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    alias_to_block = {alias: block_id for block_id, alias in aliases.items()}
    remapped = dict(parsed)
    evidence_ids = parsed.get("evidence_ids")
    if isinstance(evidence_ids, str):
        evidence_ids = [evidence_ids]
    if isinstance(evidence_ids, list):
        remapped["evidence_ids"] = [alias_to_block.get(str(evidence_id), str(evidence_id)) for evidence_id in evidence_ids]
    return remapped


def normalize_parsed_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    normalized = dict(parsed)
    for typo in ["eviidence_ids", "evidence_id", "evidences_ids"]:
        if "evidence_ids" not in normalized and typo in normalized:
            normalized["evidence_ids"] = normalized[typo]
    for typo in ["reasoning_summaary", "reasoning"]:
        if "reasoning_summary" not in normalized and typo in normalized:
            normalized["reasoning_summary"] = normalized[typo]
    return normalized


def compression_result(item: BenchmarkItem, policy: str, blocks: list[ContextBlock], prompt: str) -> CompressionResult:
    bundle = ContextBundle(
        item_id=item.id,
        strategy=policy,
        context_blocks=blocks,
        rendered_context=prompt,
        input_tokens_estimate=estimate_tokens(prompt),
    )
    return CompressionResult(
        item_id=item.id,
        strategy=policy,
        bundle=bundle,
        excluded_block_ids=[block_id for block_id in item.stale_ids if policy == "drop_stale_prompt"],
        original_tokens_estimate=estimate_tokens(prompt),
        compression_ratio=1.0,
        latency_ms=0.0,
    )


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["policy"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model_id, policy), policy_rows in sorted(grouped.items()):
        output.append(
            {
                "model_id": model_id,
                "policy": policy,
                "cases": len(policy_rows),
                "accuracy": mean_bool(policy_rows, "correct"),
                "evidence_accuracy": mean_bool(policy_rows, "evidence_correct"),
                "stale_use_rate": mean_bool(policy_rows, "used_stale_fact"),
                "clean_action_rate": mean(row.get("correct") and not row.get("used_stale_fact") for row in policy_rows),
                "avg_latency_ms": sum(float(row["latency_ms"]) for row in policy_rows) / max(1, len(policy_rows)),
            }
        )
    return output


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return mean(bool(row.get(key)) for row in rows)


def mean(values: Any) -> float:
    values = list(values)
    return sum(1.0 if value else 0.0 for value in values) / max(1, len(values))


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def default_summary_path(output: str) -> str:
    path = Path(output)
    return str(path.with_name(f"{path.stem}_summary.csv"))


if __name__ == "__main__":
    main()
