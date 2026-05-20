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
    parser = argparse.ArgumentParser(description="Summarize KV ablation result JSONL files.")
    parser.add_argument("--results", required=True, help="Comma-separated JSONL result paths.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired-policies", default="")
    args = parser.parse_args()

    rows = read_rows([ROOT / path for path in args.results.split(",") if path])
    if not rows:
        raise SystemExit("No result rows found.")
    output_rows = []
    output_rows.extend(summarize_scope("all_rows", rows))
    for domain, domain_rows in grouped(rows, "domain").items():
        output_rows.extend(summarize_scope(f"domain:{domain}", domain_rows))

    paired_policies = [value for value in args.paired_policies.split(",") if value]
    if paired_policies:
        paired_rows = paired_subset(rows, paired_policies)
        output_rows.extend(summarize_scope("paired:" + "+".join(paired_policies), paired_rows))
        for domain, domain_rows in grouped(paired_rows, "domain").items():
            output_rows.extend(summarize_scope(f"paired_domain:{domain}", domain_rows))

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "policy",
                "cases",
                "accuracy",
                "evidence_accuracy",
                "stale_usage_rate",
                "avg_prefix_tokens",
                "avg_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print_table(output_rows)
    print(f"wrote {output_path}")


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return dict(groups)


def paired_subset(rows: list[dict[str, Any]], policies: list[str]) -> list[dict[str, Any]]:
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_item[str(row["item_id"])][str(row["policy"])] = row
    paired: list[dict[str, Any]] = []
    for policy_rows in by_item.values():
        if all(policy in policy_rows for policy in policies):
            paired.extend(policy_rows[policy] for policy in policies)
    return paired


def summarize_scope(scope: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows = []
    for policy, policy_rows in sorted(grouped(rows, "policy").items()):
        output_rows.append(
            {
                "scope": scope,
                "policy": policy,
                "cases": len(policy_rows),
                "accuracy": mean_bool(policy_rows, "correct"),
                "evidence_accuracy": mean_bool(policy_rows, "evidence_correct"),
                "stale_usage_rate": mean_bool(policy_rows, "used_stale_fact"),
                "avg_prefix_tokens": mean_num(policy_rows, "prefix_tokens"),
                "avg_latency_ms": mean_num(policy_rows, "latency_ms"),
            }
        )
    return output_rows


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 if row.get(key) else 0.0 for row in rows) / len(rows)


def mean_num(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows)


def print_table(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["scope"].startswith("paired"):
            print(
                row["scope"],
                row["policy"],
                "n",
                row["cases"],
                "acc",
                f"{row['accuracy']:.3f}",
                "evidence",
                f"{row['evidence_accuracy']:.3f}",
                "stale",
                f"{row['stale_usage_rate']:.3f}",
            )


if __name__ == "__main__":
    main()
