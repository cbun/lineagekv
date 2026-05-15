from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from context_rot.compressors import CompressionConfig, build_compressor
from context_rot.datasets.io import read_jsonl
from context_rot.datasets.schema import BenchmarkItem, to_plain
from context_rot.eval.graders import grade_output, parse_model_json
from context_rot.eval.prompting import render_prompt
from context_rot.models import build_model


def run_eval(
    dataset_path: str | Path,
    output_path: str | Path,
    strategies: list[str],
    model_id: str = "heuristic",
    limit: int | None = None,
    token_budget: int = 1200,
    retrieval_top_k: int = 8,
    seed: int = 20260515,
    mlx_model: str | None = None,
    transformers_model: str | None = None,
    ollama_model: str | None = None,
) -> Path:
    items = read_jsonl(dataset_path)
    if limit is not None:
        items = stratified_prefix(items, limit)
    config = CompressionConfig(token_budget=token_budget, retrieval_top_k=retrieval_top_k, random_seed=seed)
    model = build_model(
        model_id=model_id,
        mlx_model=mlx_model,
        transformers_model=transformers_model,
        ollama_model=ollama_model,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    git_commit = _git_commit()
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            for strategy in strategies:
                compressor = build_compressor(strategy, config)
                compression = compressor.compress(item)
                prompt = render_prompt(item, compression)
                started = time.perf_counter()
                raw_output = model.generate(prompt=prompt, item=item, compression=compression)
                model_latency_ms = (time.perf_counter() - started) * 1000
                parsed = parse_model_json(raw_output)
                grade = grade_output(item, strategy, model.model_id, parsed)
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "git_commit": git_commit,
                    "item_id": item.id,
                    "domain": item.domain,
                    "question_type": item.question_type,
                    "model_id": model.model_id,
                    "strategy": strategy,
                    "distractor_ratio": item.metadata.get("distractor_ratio"),
                    "variant": item.metadata.get("variant"),
                    "contradiction_count": item.metadata.get("contradiction_count"),
                    "duplicate_wrong_count": item.metadata.get("duplicate_wrong_count"),
                    "relevant_position": item.metadata.get("relevant_position"),
                    "context_block_count": len(item.context_blocks),
                    "selected_block_count": len(compression.bundle.context_blocks),
                    "original_tokens_estimate": compression.original_tokens_estimate,
                    "input_tokens_estimate": compression.bundle.input_tokens_estimate,
                    "compression_ratio": compression.compression_ratio,
                    "compression_latency_ms": compression.latency_ms,
                    "model_latency_ms": model_latency_ms,
                    "total_latency_ms": compression.latency_ms + model_latency_ms,
                    "raw_output": raw_output,
                    "parsed_output": parsed,
                    **{key: value for key, value in to_plain(grade).items() if key not in {"item_id", "strategy", "model_id"}},
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    return output_path


def stratified_prefix(items: list[BenchmarkItem], limit: int) -> list[BenchmarkItem]:
    if limit >= len(items):
        return items
    buckets: dict[tuple[str, float], list[BenchmarkItem]] = {}
    for item in items:
        ratio = item.metadata.get("distractor_ratio", 0.0)
        key = (item.domain, float(ratio if ratio is not None else 0.0))
        buckets.setdefault(key, []).append(item)
    selected: list[BenchmarkItem] = []
    keys = sorted(buckets)
    while len(selected) < limit and keys:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip()
