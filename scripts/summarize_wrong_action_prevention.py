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
    parser = argparse.ArgumentParser(description="Summarize wrong-action prevention from KV/prompt result rows.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--baseline", default="full_cache")
    parser.add_argument("--comparators", required=True, help="Comma-separated comparator policy names.")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--pairwise-output", required=True)
    args = parser.parse_args()

    rows = read_jsonl_paths([ROOT / path for path in args.results.split(",") if path])
    comparators = [value for value in args.comparators.split(",") if value]
    summary_rows = summarize_policies(rows)
    pairwise_rows = summarize_pairwise(rows, baseline=args.baseline, comparators=comparators)
    write_csv(ROOT / args.summary_output, summary_rows)
    write_csv(ROOT / args.pairwise_output, pairwise_rows)
    for row in pairwise_rows:
        print(
            row["comparator"],
            "baseline_wrong",
            row["baseline_wrong_actions"],
            "prevented",
            row["prevented_wrong_actions"],
            "introduced",
            row["introduced_wrong_actions"],
            "prevention_rate",
            f"{row['wrong_action_prevention_rate']:.3f}",
        )
    print(f"wrote {ROOT / args.summary_output}")
    print(f"wrote {ROOT / args.pairwise_output}")


def summarize_policies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row.get("policy", row.get("strategy", "")))].append(row)
    output: list[dict[str, Any]] = []
    for policy in sorted(by_policy):
        policy_rows = by_policy[policy]
        output.append(
            {
                "policy": policy,
                "cases": len(policy_rows),
                "action_success_rate": mean(action_success(row) for row in policy_rows),
                "wrong_action_rate": mean(wrong_action(row) for row in policy_rows),
                "accuracy": mean(bool_value(row, "correct") for row in policy_rows),
                "stale_use_rate": mean(bool_value(row, "used_stale_fact") for row in policy_rows),
                "evidence_accuracy": mean(bool_value(row, "evidence_correct") for row in policy_rows),
            }
        )
    return output


def summarize_pairwise(rows: list[dict[str, Any]], *, baseline: str, comparators: list[str]) -> list[dict[str, Any]]:
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        item_id = str(row.get("item_id", ""))
        model_id = str(row.get("model_id", ""))
        policy = str(row.get("policy", row.get("strategy", "")))
        paired[f"{model_id}::{item_id}"][policy] = row
    output: list[dict[str, Any]] = []
    for comparator in comparators:
        comparable = [policies for policies in paired.values() if baseline in policies and comparator in policies]
        baseline_wrong = [policies for policies in comparable if wrong_action(policies[baseline])]
        baseline_clean = [policies for policies in comparable if not wrong_action(policies[baseline])]
        prevented = sum(1 for policies in baseline_wrong if not wrong_action(policies[comparator]))
        still_wrong = sum(1 for policies in baseline_wrong if wrong_action(policies[comparator]))
        introduced = sum(1 for policies in baseline_clean if wrong_action(policies[comparator]))
        rescued_correct = sum(
            1
            for policies in comparable
            if not bool_value(policies[baseline], "correct") and bool_value(policies[comparator], "correct")
        )
        output.append(
            {
                "baseline": baseline,
                "comparator": comparator,
                "paired_cases": len(comparable),
                "baseline_wrong_actions": len(baseline_wrong),
                "prevented_wrong_actions": prevented,
                "still_wrong_actions": still_wrong,
                "introduced_wrong_actions": introduced,
                "wrong_action_prevention_rate": safe_rate(prevented, len(baseline_wrong)),
                "introduced_wrong_action_rate": safe_rate(introduced, len(baseline_clean)),
                "rescued_correct_answers": rescued_correct,
            }
        )
    return output


def action_success(row: dict[str, Any]) -> bool:
    return bool_value(row, "correct") and not bool_value(row, "used_stale_fact")


def wrong_action(row: dict[str, Any]) -> bool:
    return not action_success(row)


def bool_value(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def mean(values: Any) -> float:
    values = list(values)
    return sum(float(bool(value)) for value in values) / max(1, len(values))


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_paths(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
