#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict and near-gold evidence citations in model result JSONL.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output", default=None)
    parser.add_argument("--near-radius", type=int, default=2)
    args = parser.parse_args()

    items = {item.id: item for item in read_jsonl(ROOT / args.dataset)}
    details: list[dict[str, Any]] = []
    with (ROOT / args.results).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item = items.get(row["item_id"])
            if not item or not item.gold_evidence_ids:
                continue
            cited_ids = _evidence_ids(row.get("parsed_output", {}))
            cited_set = set(cited_ids)
            gold_set = set(item.gold_evidence_ids)
            stale_set = set(item.stale_ids)
            cited_stale = bool(cited_set & stale_set)
            near_pairs = _near_pairs(item, cited_set, gold_set, args.near_radius)
            strict_cover = gold_set <= cited_set and not cited_stale
            strict_any = bool(gold_set & cited_set) and not cited_stale
            near_any = bool(near_pairs) and not cited_stale
            near_cover = _near_cover(item, cited_set, gold_set, args.near_radius) and not cited_stale
            details.append(
                {
                    "item_id": item.id,
                    "strategy": row["strategy"],
                    "correct": bool(row.get("correct", False)),
                    "evidence_correct": bool(row.get("evidence_correct", False)),
                    "strict_cover": strict_cover,
                    "strict_any": strict_any,
                    "near_any": near_any,
                    "near_cover": near_cover,
                    "correct_and_strict_cover": bool(row.get("correct", False)) and strict_cover,
                    "correct_and_near_any": bool(row.get("correct", False)) and near_any,
                    "gold_evidence_ids": sorted(gold_set),
                    "cited_evidence_ids": sorted(cited_set),
                    "near_pairs": near_pairs,
                    "input_tokens_estimate": row.get("input_tokens_estimate"),
                }
            )
    detail_df = pd.DataFrame(details)
    if detail_df.empty:
        raise SystemExit("No auditable rows found.")
    summary = (
        detail_df.groupby("strategy", sort=False)
        .agg(
            cases=("item_id", "count"),
            accuracy=("correct", "mean"),
            strict_cover=("strict_cover", "mean"),
            strict_any=("strict_any", "mean"),
            near_any=("near_any", "mean"),
            near_cover=("near_cover", "mean"),
            correct_and_strict_cover=("correct_and_strict_cover", "mean"),
            correct_and_near_any=("correct_and_near_any", "mean"),
            avg_input_tokens=("input_tokens_estimate", "mean"),
        )
        .reset_index()
    )
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    if args.details_output:
        details_path = ROOT / args.details_output
        serializable = detail_df.copy()
        for column in ["gold_evidence_ids", "cited_evidence_ids", "near_pairs"]:
            serializable[column] = serializable[column].map(json.dumps)
        serializable.to_csv(details_path, index=False)
    print(summary.to_string(index=False))


def _evidence_ids(parsed_output: Any) -> list[str]:
    if not isinstance(parsed_output, dict):
        return []
    value = parsed_output.get("evidence_ids", [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [_normalize_evidence_id(str(eid)) for eid in value]


def _normalize_evidence_id(evidence_id: str) -> str:
    evidence_id = evidence_id.strip()
    evidence_id = re.sub(r"^id=", "", evidence_id)
    return evidence_id.strip("\"' ")


def _near_pairs(item, cited_set: set[str], gold_set: set[str], radius: int) -> list[dict[str, Any]]:
    positions = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    pairs: list[dict[str, Any]] = []
    for cited_id in sorted(cited_set):
        if cited_id not in positions:
            continue
        for gold_id in sorted(gold_set):
            if gold_id not in positions:
                continue
            distance = abs(positions[cited_id] - positions[gold_id])
            if distance <= radius:
                pairs.append({"cited": cited_id, "gold": gold_id, "distance": distance})
    return pairs


def _near_cover(item, cited_set: set[str], gold_set: set[str], radius: int) -> bool:
    positions = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    if not gold_set:
        return False
    for gold_id in gold_set:
        if gold_id not in positions:
            return False
        if not any(cited_id in positions and abs(positions[cited_id] - positions[gold_id]) <= radius for cited_id in cited_set):
            return False
    return True


if __name__ == "__main__":
    main()
