#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.io import read_jsonl
from context_rot.datasets.schema import BenchmarkItem
from kv_value_ablation_probe import detected_conflict_stale_ids, extract_value_claim
from summarize_kv_results import summarize_scope


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a structurally gated KV edit policy.")
    parser.add_argument("--datasets", required=True, help="Comma-separated datasets for items and calibration.")
    parser.add_argument("--results", required=True, help="Comma-separated KV result JSONL files.")
    parser.add_argument("--output", default="results/kv/kv_selective_policy_summary.csv")
    parser.add_argument("--decision-output", default="")
    parser.add_argument("--baseline-policy", default="full_cache")
    parser.add_argument("--edit-policy", default="zero_duplicate_aware_conflict_values_last_quarter")
    parser.add_argument("--simulated-policy", default="selective_structural_gate_q4")
    parser.add_argument("--distractor-ratio", type=float, default=0.9)
    parser.add_argument("--variants", default="")
    args = parser.parse_args()

    allowed_variants = {value for value in args.variants.split(",") if value}
    items: dict[str, BenchmarkItem] = {}
    calibration: dict[str, Counter[str]] = defaultdict(Counter)
    for dataset in [value for value in args.datasets.split(",") if value]:
        for item in read_jsonl(ROOT / dataset):
            if float(item.metadata.get("distractor_ratio", -1.0)) != args.distractor_ratio:
                continue
            if allowed_variants and str(item.metadata.get("variant", "")) not in allowed_variants:
                continue
            items[item.id] = item
            for signature, label_pattern in structural_patterns(item, set(item.stale_ids)):
                calibration[signature][label_pattern] += 1

    result_rows = read_result_rows([ROOT / value for value in args.results.split(",") if value])
    rows_by_item_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in result_rows:
        rows_by_item_policy[row["item_id"]][row["policy"]] = row

    simulated_rows = []
    decision_rows = []
    decisions = Counter()
    for item_id, item in items.items():
        policy_rows = rows_by_item_policy.get(item_id, {})
        baseline = policy_rows.get(args.baseline_policy)
        edit = policy_rows.get(args.edit_policy)
        if baseline is None or edit is None:
            continue
        predicted_stale_ids = set(
            detected_conflict_stale_ids(item, use_current_text_cues=False, prefer_singleton_conflicts=True)
        )
        certifiable = is_certifiable(item, predicted_stale_ids, calibration)
        selected = edit if certifiable else baseline
        decisions["edit" if certifiable else "fallback"] += 1
        simulated = dict(selected)
        simulated["policy"] = args.simulated_policy
        simulated["selected_source_policy"] = selected["policy"]
        simulated["certified_edit"] = certifiable
        simulated_rows.append(simulated)
        decision_rows.append(
            {
                "item_id": item_id,
                "domain": item.domain,
                "variant": item.metadata.get("variant"),
                "selected_source_policy": selected["policy"],
                "certified_edit": certifiable,
                "correct": selected.get("correct"),
                "evidence_correct": selected.get("evidence_correct"),
                "used_stale_fact": selected.get("used_stale_fact"),
            }
        )

    output_rows = summarize_scope("simulated", simulated_rows)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scope",
            "policy",
            "cases",
            "accuracy",
            "evidence_accuracy",
            "stale_usage_rate",
            "avg_prefix_tokens",
            "avg_latency_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    if args.decision_output:
        decision_path = ROOT / args.decision_output
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        with decision_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "item_id",
                "domain",
                "variant",
                "selected_source_policy",
                "certified_edit",
                "correct",
                "evidence_correct",
                "used_stale_fact",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(decision_rows)

    for row in output_rows:
        print(
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
    print("decisions", dict(decisions))
    print(f"wrote {output_path}")
    if args.decision_output:
        print(f"wrote {decision_path}")


def read_result_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def is_certifiable(
    item: BenchmarkItem,
    predicted_stale_ids: set[str],
    calibration: dict[str, Counter[str]],
) -> bool:
    patterns = structural_patterns(item, predicted_stale_ids)
    if not patterns:
        return False
    for signature, predicted_pattern in patterns:
        labels = calibration.get(signature, Counter())
        if len(labels) != 1:
            return False
        if predicted_pattern not in labels:
            return False
    return True


def structural_patterns(item: BenchmarkItem, stale_ids: set[str]) -> list[tuple[str, str]]:
    groups: dict[tuple[str, str], list[tuple[str, int, str, str]]] = defaultdict(list)
    for position, block in enumerate(item.context_blocks):
        claim = extract_value_claim(block)
        if claim is None:
            continue
        entity, prop, value = claim
        groups[(entity, prop)].append((block.id, position, block.timestamp, value))

    patterns = []
    for claims in groups.values():
        values = {value for _, _, _, value in claims}
        if len(values) < 2:
            continue
        sorted_timestamps = sorted({timestamp for _, _, timestamp, _ in claims})
        timestamp_rank = {timestamp: rank for rank, timestamp in enumerate(sorted_timestamps)}
        value_entries: list[tuple[tuple[int, int, int, int, int], str]] = []
        for value in values:
            value_claims = [
                (block_id, position, timestamp)
                for block_id, position, timestamp, claim_value in claims
                if claim_value == value
            ]
            positions = [position for _, position, _ in value_claims]
            ranks = [timestamp_rank[timestamp] for _, _, timestamp in value_claims]
            role = (len(value_claims), min(ranks), max(ranks), min(positions), max(positions))
            block_ids = {block_id for block_id, _, _ in value_claims}
            if block_ids <= stale_ids:
                label = "S"
            elif block_ids.isdisjoint(stale_ids):
                label = "C"
            else:
                label = "M"
            value_entries.append((role, label))
        value_entries.sort(key=lambda item: item[0])
        signature = json.dumps([role for role, _ in value_entries], separators=(",", ":"))
        label_pattern = "".join(label for _, label in value_entries)
        patterns.append((signature, label_pattern))
    return patterns


if __name__ == "__main__":
    main()
