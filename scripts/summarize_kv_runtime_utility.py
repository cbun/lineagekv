#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize KV runtime columns. cold_ms estimates a fresh prefill+decode path. "
            "warm_reuse_ms estimates the incremental path when the relevant prefill cache already exists."
        )
    )
    parser.add_argument("--results", required=True, help="Comma-separated JSONL result paths.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_rows([ROOT / path for path in args.results.split(",") if path])
    if not rows:
        raise SystemExit("No result rows found.")
    output_rows = summarize(rows)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    for row in output_rows:
        print(
            row["policy"],
            "n",
            row["cases"],
            "cold_ms",
            f"{row['avg_cold_ms']:.1f}",
            "warm_reuse_ms",
            f"{row['avg_warm_reuse_ms']:.1f}",
            "edit_ms",
            f"{row['avg_cache_edit_ms']:.3f}",
            "prefix_tokens",
            f"{row['avg_prefix_tokens']:.1f}",
        )
    print(f"wrote {output_path}")


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    output_rows = []
    for policy, policy_rows in sorted(grouped.items()):
        output_rows.append(
            {
                "policy": policy,
                "cases": len(policy_rows),
                "avg_prefix_tokens": mean(policy_rows, "prefix_tokens"),
                "avg_full_prefix_tokens": mean(policy_rows, "full_prefix_tokens"),
                "avg_stale_token_count": mean(policy_rows, "stale_token_count"),
                "avg_prefill_ms": mean(policy_rows, "prefill_ms"),
                "avg_full_prefill_ms": mean(policy_rows, "full_prefill_ms"),
                "avg_cache_clone_ms": mean(policy_rows, "cache_clone_ms"),
                "avg_cache_edit_ms": mean(policy_rows, "cache_edit_ms"),
                "avg_decode_ms": mean(policy_rows, "decode_ms", fallback_key="latency_ms"),
                "avg_cold_ms": mean_total(policy_rows, include_prefill=True),
                "avg_warm_reuse_ms": mean_total(policy_rows, include_prefill=False),
                "accuracy": mean_bool(policy_rows, "correct"),
                "evidence_accuracy": mean_bool(policy_rows, "evidence_correct"),
                "stale_usage_rate": mean_bool(policy_rows, "used_stale_fact"),
            }
        )
    return output_rows


def mean(rows: list[dict[str, Any]], key: str, fallback_key: str | None = None) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None and fallback_key is not None:
            value = row.get(fallback_key)
        values.append(float(value or 0.0))
    return sum(values) / len(values)


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 if row.get(key) else 0.0 for row in rows) / len(rows)


def mean_total(rows: list[dict[str, Any]], include_prefill: bool) -> float:
    totals = []
    for row in rows:
        total = float(row.get("cache_clone_ms") or 0.0)
        total += float(row.get("cache_edit_ms") or 0.0)
        total += float(row.get("decode_ms", row.get("latency_ms")) or 0.0)
        if include_prefill:
            total += float(row.get("prefill_ms") or 0.0)
        totals.append(total)
    return sum(totals) / len(totals)


if __name__ == "__main__":
    main()
