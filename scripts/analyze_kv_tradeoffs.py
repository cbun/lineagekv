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
    parser = argparse.ArgumentParser(description="Analyze paired KV intervention tradeoffs.")
    parser.add_argument("--results", required=True, help="Comma-separated JSONL result paths.")
    parser.add_argument("--baseline", default="full_cache")
    parser.add_argument("--comparators", required=True, help="Comma-separated comparator policies.")
    parser.add_argument("--output-policy", required=True)
    parser.add_argument("--output-comparison", required=True)
    args = parser.parse_args()

    rows = read_rows([ROOT / path for path in args.results.split(",") if path])
    comparators = [value for value in args.comparators.split(",") if value]
    policies = [args.baseline, *comparators]
    paired_items = complete_items(rows, policies)
    if not paired_items:
        raise SystemExit("No complete paired items found.")

    policy_rows = policy_summary(paired_items, policies)
    comparison_rows = comparison_summary(paired_items, args.baseline, comparators)
    write_csv(ROOT / args.output_policy, policy_rows, POLICY_FIELDS)
    write_csv(ROOT / args.output_comparison, comparison_rows, COMPARISON_FIELDS)
    print_tables(policy_rows, comparison_rows)


POLICY_FIELDS = [
    "policy",
    "cases",
    "accuracy",
    "clean_accuracy",
    "evidence_accuracy",
    "clean_evidence_accuracy",
    "stale_usage_rate",
    "correct_stale_rate",
]

COMPARISON_FIELDS = [
    "baseline",
    "comparator",
    "cases",
    "baseline_accuracy",
    "comparator_accuracy",
    "accuracy_delta",
    "baseline_clean_accuracy",
    "comparator_clean_accuracy",
    "clean_accuracy_delta",
    "baseline_evidence_accuracy",
    "comparator_evidence_accuracy",
    "evidence_delta",
    "baseline_stale_usage_rate",
    "comparator_stale_usage_rate",
    "stale_usage_delta",
    "baseline_correct_cases",
    "preserved_correct_cases",
    "answer_preservation_rate",
    "baseline_incorrect_cases",
    "rescued_answer_cases",
    "answer_rescue_rate",
    "baseline_stale_cases",
    "cleaned_stale_cases",
    "stale_cleanup_rate",
    "stale_regression_cases",
    "stale_regression_rate",
    "accuracy_loss_per_stale_point_removed",
]


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


def policy_summary(paired_items: list[dict[str, dict[str, Any]]], policies: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        values = [policy_rows[policy] for policy_rows in paired_items]
        rows.append(
            {
                "policy": policy,
                "cases": len(values),
                "accuracy": mean(bool_value(row, "correct") for row in values),
                "clean_accuracy": mean(clean_correct(row) for row in values),
                "evidence_accuracy": mean(bool_value(row, "evidence_correct") for row in values),
                "clean_evidence_accuracy": mean(clean_evidence_correct(row) for row in values),
                "stale_usage_rate": mean(bool_value(row, "used_stale_fact") for row in values),
                "correct_stale_rate": mean(correct_stale(row) for row in values),
            }
        )
    return rows


def comparison_summary(
    paired_items: list[dict[str, dict[str, Any]]],
    baseline: str,
    comparators: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparator in comparators:
        base_rows = [policy_rows[baseline] for policy_rows in paired_items]
        comp_rows = [policy_rows[comparator] for policy_rows in paired_items]
        base_accuracy = mean(bool_value(row, "correct") for row in base_rows)
        comp_accuracy = mean(bool_value(row, "correct") for row in comp_rows)
        base_clean_accuracy = mean(clean_correct(row) for row in base_rows)
        comp_clean_accuracy = mean(clean_correct(row) for row in comp_rows)
        base_evidence = mean(bool_value(row, "evidence_correct") for row in base_rows)
        comp_evidence = mean(bool_value(row, "evidence_correct") for row in comp_rows)
        base_stale = mean(bool_value(row, "used_stale_fact") for row in base_rows)
        comp_stale = mean(bool_value(row, "used_stale_fact") for row in comp_rows)

        baseline_correct = [idx for idx, row in enumerate(base_rows) if bool_value(row, "correct")]
        baseline_incorrect = [idx for idx, row in enumerate(base_rows) if not bool_value(row, "correct")]
        baseline_stale = [idx for idx, row in enumerate(base_rows) if bool_value(row, "used_stale_fact")]
        baseline_clean = [idx for idx, row in enumerate(base_rows) if not bool_value(row, "used_stale_fact")]

        preserved_correct = sum(1 for idx in baseline_correct if bool_value(comp_rows[idx], "correct"))
        rescued_answers = sum(1 for idx in baseline_incorrect if bool_value(comp_rows[idx], "correct"))
        cleaned_stale = sum(1 for idx in baseline_stale if not bool_value(comp_rows[idx], "used_stale_fact"))
        stale_regressions = sum(1 for idx in baseline_clean if bool_value(comp_rows[idx], "used_stale_fact"))

        stale_removed = max(0.0, base_stale - comp_stale)
        accuracy_loss = max(0.0, base_accuracy - comp_accuracy)
        rows.append(
            {
                "baseline": baseline,
                "comparator": comparator,
                "cases": len(paired_items),
                "baseline_accuracy": base_accuracy,
                "comparator_accuracy": comp_accuracy,
                "accuracy_delta": comp_accuracy - base_accuracy,
                "baseline_clean_accuracy": base_clean_accuracy,
                "comparator_clean_accuracy": comp_clean_accuracy,
                "clean_accuracy_delta": comp_clean_accuracy - base_clean_accuracy,
                "baseline_evidence_accuracy": base_evidence,
                "comparator_evidence_accuracy": comp_evidence,
                "evidence_delta": comp_evidence - base_evidence,
                "baseline_stale_usage_rate": base_stale,
                "comparator_stale_usage_rate": comp_stale,
                "stale_usage_delta": comp_stale - base_stale,
                "baseline_correct_cases": len(baseline_correct),
                "preserved_correct_cases": preserved_correct,
                "answer_preservation_rate": safe_rate(preserved_correct, len(baseline_correct)),
                "baseline_incorrect_cases": len(baseline_incorrect),
                "rescued_answer_cases": rescued_answers,
                "answer_rescue_rate": safe_rate(rescued_answers, len(baseline_incorrect)),
                "baseline_stale_cases": len(baseline_stale),
                "cleaned_stale_cases": cleaned_stale,
                "stale_cleanup_rate": safe_rate(cleaned_stale, len(baseline_stale)),
                "stale_regression_cases": stale_regressions,
                "stale_regression_rate": safe_rate(stale_regressions, len(baseline_clean)),
                "accuracy_loss_per_stale_point_removed": safe_div(accuracy_loss, stale_removed),
            }
        )
    return rows


def bool_value(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key))


def clean_correct(row: dict[str, Any]) -> bool:
    return bool_value(row, "correct") and not bool_value(row, "used_stale_fact")


def clean_evidence_correct(row: dict[str, Any]) -> bool:
    return bool_value(row, "correct") and bool_value(row, "evidence_correct")


def correct_stale(row: dict[str, Any]) -> bool:
    return bool_value(row, "correct") and bool_value(row, "used_stale_fact")


def mean(values: Any) -> float:
    values = list(values)
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_tables(policy_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    for row in policy_rows:
        print(
            row["policy"],
            "n",
            row["cases"],
            "acc",
            f"{row['accuracy']:.3f}",
            "clean_acc",
            f"{row['clean_accuracy']:.3f}",
            "evidence",
            f"{row['evidence_accuracy']:.3f}",
            "stale",
            f"{row['stale_usage_rate']:.3f}",
        )
    for row in comparison_rows:
        print(
            row["comparator"],
            "preserve",
            f"{row['answer_preservation_rate']:.3f}",
            "cleanup",
            f"{row['stale_cleanup_rate']:.3f}",
            "acc_loss_per_stale_removed",
            f"{row['accuracy_loss_per_stale_point_removed']:.3f}",
        )


if __name__ == "__main__":
    main()
