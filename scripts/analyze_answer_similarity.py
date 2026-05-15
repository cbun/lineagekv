#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "will",
    "with",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify generated answers as current-like, stale-like, or neither using lexical F1."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True, help="Comma-separated KV result JSONL paths.")
    parser.add_argument("--output-rows", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=0.08)
    args = parser.parse_args()

    items = {item.id: item for item in read_jsonl(ROOT / args.dataset)}
    rows = []
    for path_text in args.results.split(","):
        if not path_text:
            continue
        with (ROOT / path_text).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                result = json.loads(line)
                item = items[str(result["item_id"])]
                rows.append(score_row(result, item, args.threshold, args.margin))

    write_rows(ROOT / args.output_rows, rows)
    summary = summarize(rows)
    write_summary(ROOT / args.output_summary, summary)
    print_summary(summary)


def score_row(result: dict[str, Any], item: Any, threshold: float, margin: float) -> dict[str, Any]:
    answer = str(result.get("metrics", {}).get("answer", ""))
    current = str(item.metadata.get("target_value", item.gold_answer))
    stale = str(item.metadata.get("stale_value", ""))
    current_f1 = token_f1(answer, current)
    stale_f1 = token_f1(answer, stale)
    label = classify(current_f1, stale_f1, threshold, margin)
    return {
        "item_id": result["item_id"],
        "policy": result["policy"],
        "exact_correct": bool(result.get("correct")),
        "evidence_correct": bool(result.get("evidence_correct")),
        "used_stale_fact": bool(result.get("used_stale_fact")),
        "current_f1": current_f1,
        "stale_f1": stale_f1,
        "f1_margin": current_f1 - stale_f1,
        "answer_label": label,
        "answer": answer,
        "current": current,
        "stale": stale,
    }


def token_f1(answer: str, target: str) -> float:
    answer_tokens = content_tokens(answer)
    target_tokens = content_tokens(target)
    if not answer_tokens or not target_tokens:
        return 0.0
    overlap = sum(min(answer_tokens.get(token, 0), target_tokens.get(token, 0)) for token in target_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / sum(answer_tokens.values())
    recall = overlap / sum(target_tokens.values())
    return 2 * precision * recall / (precision + recall)


def content_tokens(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in STOPWORDS:
            continue
        token = light_stem(token)
        counts[token] = counts.get(token, 0) + 1
    return counts


def light_stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def classify(current_f1: float, stale_f1: float, threshold: float, margin: float) -> str:
    if current_f1 >= threshold and current_f1 >= stale_f1 + margin:
        return "current_like"
    if stale_f1 >= threshold and stale_f1 >= current_f1 + margin:
        return "stale_like"
    return "ambiguous_or_other"


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    summary = []
    for policy, policy_rows in sorted(grouped.items()):
        cases = len(policy_rows)
        summary.append(
            {
                "policy": policy,
                "cases": cases,
                "exact_accuracy": mean_bool(policy_rows, "exact_correct"),
                "evidence_accuracy": mean_bool(policy_rows, "evidence_correct"),
                "stale_usage_rate": mean_bool(policy_rows, "used_stale_fact"),
                "current_like_rate": mean_label(policy_rows, "current_like"),
                "stale_like_rate": mean_label(policy_rows, "stale_like"),
                "ambiguous_or_other_rate": mean_label(policy_rows, "ambiguous_or_other"),
                "avg_current_f1": mean_float(policy_rows, "current_f1"),
                "avg_stale_f1": mean_float(policy_rows, "stale_f1"),
                "avg_f1_margin": mean_float(policy_rows, "f1_margin"),
            }
        )
    return summary


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 for row in rows if row[key]) / len(rows)


def mean_label(rows: list[dict[str, Any]], label: str) -> float:
    return sum(1.0 for row in rows if row["answer_label"] == label) / len(rows)


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "policy",
                "exact_correct",
                "evidence_correct",
                "used_stale_fact",
                "current_f1",
                "stale_f1",
                "f1_margin",
                "answer_label",
                "answer",
                "current",
                "stale",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "cases",
                "exact_accuracy",
                "evidence_accuracy",
                "stale_usage_rate",
                "current_like_rate",
                "stale_like_rate",
                "ambiguous_or_other_rate",
                "avg_current_f1",
                "avg_stale_f1",
                "avg_f1_margin",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            row["policy"],
            "n",
            row["cases"],
            "current_like",
            f"{row['current_like_rate']:.3f}",
            "stale_like",
            f"{row['stale_like_rate']:.3f}",
            "avg_current_f1",
            f"{row['avg_current_f1']:.3f}",
            "avg_stale_f1",
            f"{row['avg_stale_f1']:.3f}",
        )


if __name__ == "__main__":
    main()
