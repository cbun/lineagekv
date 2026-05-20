#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


METRICS = ("correct", "evidence_correct", "used_stale_fact")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired KV-policy deltas with bootstrap intervals.")
    parser.add_argument("--results", required=True, help="Comma-separated JSONL result paths.")
    parser.add_argument("--baseline", default="full_cache")
    parser.add_argument("--comparators", required=True, help="Comma-separated policies to compare against baseline.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = read_rows([ROOT / path for path in args.results.split(",") if path])
    comparators = [value for value in args.comparators.split(",") if value]
    policies = [args.baseline, *comparators]
    paired_items = complete_items(rows, policies)
    if not paired_items:
        raise SystemExit("No complete paired items found.")

    output_rows: list[dict[str, Any]] = []
    output_rows.extend(compare_scope("all", paired_items, args.baseline, comparators, args.bootstrap_samples, args.seed))
    for domain, domain_items in grouped_items(paired_items, "domain").items():
        output_rows.extend(
            compare_scope(f"domain:{domain}", domain_items, args.baseline, comparators, args.bootstrap_samples, args.seed)
        )

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "baseline",
        "comparator",
        "metric",
        "cases",
        "baseline_mean",
        "comparator_mean",
        "delta",
        "ci_low",
        "ci_high",
        "wins",
        "losses",
        "ties",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print_overall(output_rows)
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


def grouped_items(
    paired_items: list[dict[str, dict[str, Any]]], key: str
) -> dict[str, list[dict[str, dict[str, Any]]]]:
    groups: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for policy_rows in paired_items:
        first_row = next(iter(policy_rows.values()))
        groups[str(first_row.get(key, ""))].append(policy_rows)
    return dict(groups)


def compare_scope(
    scope: str,
    paired_items: list[dict[str, dict[str, Any]]],
    baseline: str,
    comparators: list[str],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for comparator in comparators:
        for metric in METRICS:
            baseline_values = [as_float(policy_rows[baseline].get(metric)) for policy_rows in paired_items]
            comparator_values = [as_float(policy_rows[comparator].get(metric)) for policy_rows in paired_items]
            diffs = [right - left for left, right in zip(baseline_values, comparator_values)]
            ci_low, ci_high = bootstrap_ci(diffs, bootstrap_samples, seed)
            output_rows.append(
                {
                    "scope": scope,
                    "baseline": baseline,
                    "comparator": comparator,
                    "metric": metric,
                    "cases": len(diffs),
                    "baseline_mean": mean(baseline_values),
                    "comparator_mean": mean(comparator_values),
                    "delta": mean(diffs),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wins": sum(1 for diff in diffs if diff > 0),
                    "losses": sum(1 for diff in diffs if diff < 0),
                    "ties": sum(1 for diff in diffs if diff == 0),
                }
            )
    return output_rows


def as_float(value: Any) -> float:
    return 1.0 if value else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(diffs: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    low_idx = int(0.025 * (samples - 1))
    high_idx = int(0.975 * (samples - 1))
    return estimates[low_idx], estimates[high_idx]


def print_overall(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["scope"] == "all":
            print(
                row["comparator"],
                row["metric"],
                "n",
                row["cases"],
                "delta",
                f"{row['delta']:.3f}",
                "ci",
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
                "w/l/t",
                f"{row['wins']}/{row['losses']}/{row['ties']}",
            )


if __name__ == "__main__":
    main()
