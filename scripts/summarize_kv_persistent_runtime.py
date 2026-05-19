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
            "Compare persistent-cache update latency. Assumes the full-context prefill cache already exists; "
            "cache-edit policies can reuse it, while prompt deletion must prefill a cleaned prompt."
        )
    )
    parser.add_argument("--results", required=True, help="Comma-separated result JSONL paths.")
    parser.add_argument("--baseline", default="drop_stale_prompt")
    parser.add_argument("--comparators", required=True, help="Comma-separated cache-edit policies.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_rows([ROOT / path for path in args.results.split(",") if path])
    comparators = [value for value in args.comparators.split(",") if value]
    paired = complete_items(rows, [args.baseline, *comparators])
    if not paired:
        raise SystemExit("No complete paired items found.")

    output_rows: list[dict[str, Any]] = []
    for comparator in comparators:
        output_rows.append(compare_policy(paired, args.baseline, comparator))

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    for row in output_rows:
        print(
            row["comparator"],
            "n",
            row["cases"],
            "prompt_update_ms",
            f"{row['avg_prompt_deletion_update_ms']:.1f}",
            "cache_update_ms",
            f"{row['avg_cache_repair_update_ms']:.1f}",
            "speedup",
            f"{row['speedup_ratio']:.2f}x",
            "prefill_saved_ms",
            f"{row['avg_prefill_saved_ms']:.1f}",
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


def complete_items(rows: list[dict[str, Any]], policies: list[str]) -> list[dict[str, dict[str, Any]]]:
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_item[str(row["item_id"])][str(row["policy"])] = row
    return [policy_rows for policy_rows in by_item.values() if all(policy in policy_rows for policy in policies)]


def compare_policy(
    paired: list[dict[str, dict[str, Any]]],
    baseline: str,
    comparator: str,
) -> dict[str, Any]:
    prompt_update_ms = []
    cache_update_ms = []
    prefill_saved_ms = []
    correct_delta = []
    evidence_delta = []
    stale_delta = []
    for policy_rows in paired:
        base = policy_rows[baseline]
        comp = policy_rows[comparator]
        prompt_update = value(base, "prefill_ms") + value(base, "cache_clone_ms") + value(base, "decode_ms", "latency_ms")
        cache_update = value(comp, "cache_clone_ms") + value(comp, "cache_edit_ms") + value(comp, "decode_ms", "latency_ms")
        prompt_update_ms.append(prompt_update)
        cache_update_ms.append(cache_update)
        prefill_saved_ms.append(value(base, "prefill_ms") - value(comp, "cache_clone_ms") - value(comp, "cache_edit_ms"))
        correct_delta.append(as_float(comp.get("correct")) - as_float(base.get("correct")))
        evidence_delta.append(as_float(comp.get("evidence_correct")) - as_float(base.get("evidence_correct")))
        stale_delta.append(as_float(comp.get("used_stale_fact")) - as_float(base.get("used_stale_fact")))
    prompt_mean = mean(prompt_update_ms)
    cache_mean = mean(cache_update_ms)
    return {
        "baseline": baseline,
        "comparator": comparator,
        "cases": len(paired),
        "avg_prompt_deletion_update_ms": prompt_mean,
        "avg_cache_repair_update_ms": cache_mean,
        "avg_prefill_saved_ms": mean(prefill_saved_ms),
        "speedup_ratio": prompt_mean / cache_mean if cache_mean else 0.0,
        "accuracy_delta": mean(correct_delta),
        "evidence_delta": mean(evidence_delta),
        "stale_usage_delta": mean(stale_delta),
    }


def value(row: dict[str, Any], key: str, fallback_key: str | None = None) -> float:
    raw = row.get(key)
    if raw is None and fallback_key is not None:
        raw = row.get(fallback_key)
    return float(raw or 0.0)


def as_float(raw: Any) -> float:
    return 1.0 if raw else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
