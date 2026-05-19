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
sys.path.insert(0, str(ROOT))

from context_rot.datasets.io import read_jsonl
from context_rot.eval.graders import grade_output
from scripts.kv_value_ablation_probe import detected_lineage_current_ids, detected_text_timestamp_current_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic evidence-ID repair from memory lineage/currentness metadata. "
            "This measures a system provenance layer, not model-generated citation quality."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True, help="Comma-separated KV result JSONL paths.")
    parser.add_argument("--output", required=True, help="Per-row repaired output CSV.")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument(
        "--detector",
        choices=["text_timestamp", "lineage"],
        default="text_timestamp",
        help="Programmatic current-block detector used to replace evidence_ids.",
    )
    args = parser.parse_args()

    items = {item.id: item for item in read_jsonl(ROOT / args.dataset)}
    rows = read_rows([ROOT / path for path in args.results.split(",") if path])
    repaired_rows = [repair_row(row, items, args.detector) for row in rows]

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "policy",
        "correct",
        "original_evidence_correct",
        "original_joint_correct",
        "repaired_evidence_correct",
        "repaired_joint_correct",
        "detector_covered",
        "used_stale_fact",
        "original_evidence_ids",
        "repaired_evidence_ids",
        "gold_evidence_ids",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(repaired_rows)

    summary_rows = summarize(repaired_rows)
    summary_path = ROOT / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "cases",
                "accuracy",
                "original_evidence_accuracy",
                "original_joint_accuracy",
                "repaired_evidence_accuracy",
                "repaired_joint_accuracy",
                "detector_coverage",
                "stale_usage_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        print(
            row["policy"],
            "n",
            row["cases"],
            "acc",
            f"{row['accuracy']:.3f}",
            "orig_ev",
            f"{row['original_evidence_accuracy']:.3f}",
            "repair_ev",
            f"{row['repaired_evidence_accuracy']:.3f}",
            "repair_joint",
            f"{row['repaired_joint_accuracy']:.3f}",
        )
    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def repair_row(row: dict[str, Any], items: dict[str, Any], detector: str) -> dict[str, Any]:
    item_id = str(row["item_id"])
    item = items[item_id]
    current_ids = current_block_ids(item, detector)
    parsed = dict(row.get("parsed_output") or {})
    original_evidence_ids = parsed.get("evidence_ids", [])
    if isinstance(original_evidence_ids, str):
        original_evidence_ids = [original_evidence_ids]

    repaired = dict(parsed)
    if current_ids:
        repaired["evidence_ids"] = current_ids
    repaired_grade = grade_output(item, str(row["policy"]), str(row.get("model_id", "")), repaired)

    return {
        "item_id": item_id,
        "policy": row["policy"],
        "correct": bool(row.get("correct")),
        "original_evidence_correct": bool(row.get("evidence_correct")),
        "original_joint_correct": bool(row.get("correct")) and bool(row.get("evidence_correct")),
        "repaired_evidence_correct": bool(repaired_grade.evidence_correct),
        "repaired_joint_correct": bool(repaired_grade.correct) and bool(repaired_grade.evidence_correct),
        "detector_covered": bool(current_ids),
        "used_stale_fact": bool(row.get("used_stale_fact")),
        "original_evidence_ids": "|".join(str(value) for value in original_evidence_ids),
        "repaired_evidence_ids": "|".join(str(value) for value in repaired.get("evidence_ids", [])),
        "gold_evidence_ids": "|".join(str(value) for value in item.gold_evidence_ids),
    }


def current_block_ids(item: Any, detector: str) -> list[str]:
    if detector == "text_timestamp":
        return detected_text_timestamp_current_ids(item)
    if detector == "lineage":
        return detected_lineage_current_ids(item)
    raise KeyError(detector)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for policy, policy_rows in sorted(by_policy.items()):
        summary_rows.append(
            {
                "policy": policy,
                "cases": len(policy_rows),
                "accuracy": mean_bool(policy_rows, "correct"),
                "original_evidence_accuracy": mean_bool(policy_rows, "original_evidence_correct"),
                "original_joint_accuracy": mean_bool(policy_rows, "original_joint_correct"),
                "repaired_evidence_accuracy": mean_bool(policy_rows, "repaired_evidence_correct"),
                "repaired_joint_accuracy": mean_bool(policy_rows, "repaired_joint_correct"),
                "detector_coverage": mean_bool(policy_rows, "detector_covered"),
                "stale_usage_rate": mean_bool(policy_rows, "used_stale_fact"),
            }
        )
    return summary_rows


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 if row.get(key) else 0.0 for row in rows) / len(rows)


if __name__ == "__main__":
    main()
