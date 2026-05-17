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
from kv_value_ablation_probe import extract_value_claim


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether structural conflict features identify stale spans."
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated JSONL datasets to audit.",
    )
    parser.add_argument("--output", default="results/kv/kv_structural_identifiability_audit.csv")
    parser.add_argument("--distractor-ratio", type=float, default=0.9)
    parser.add_argument("--variants", default="")
    args = parser.parse_args()

    allowed_variants = {value for value in args.variants.split(",") if value}
    examples_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset in [value for value in args.datasets.split(",") if value]:
        for item in read_jsonl(ROOT / dataset):
            if float(item.metadata.get("distractor_ratio", -1.0)) != args.distractor_ratio:
                continue
            if allowed_variants and str(item.metadata.get("variant", "")) not in allowed_variants:
                continue
            for example in item_structural_examples(item, dataset):
                examples_by_signature[example["signature"]].append(example)

    rows = []
    total_examples = 0
    majority_correct = 0
    ambiguous_signatures = 0
    ambiguous_examples = 0
    for signature, examples in sorted(examples_by_signature.items()):
        labels = Counter(example["label_pattern"] for example in examples)
        majority_label, majority_count = labels.most_common(1)[0]
        total_examples += len(examples)
        majority_correct += majority_count
        is_ambiguous = len(labels) > 1
        ambiguous_signatures += int(is_ambiguous)
        ambiguous_examples += len(examples) if is_ambiguous else 0
        rows.append(
            {
                "signature": signature,
                "examples": len(examples),
                "label_patterns": json.dumps(dict(sorted(labels.items())), sort_keys=True),
                "ambiguous": is_ambiguous,
                "majority_label": majority_label,
                "majority_count": majority_count,
                "example_items": " ".join(example["item_id"] for example in examples[:8]),
                "example_variants": " ".join(str(example["variant"]) for example in examples[:8]),
                "example_datasets": " ".join(example["dataset"] for example in examples[:8]),
            }
        )

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "signature",
            "examples",
            "label_patterns",
            "ambiguous",
            "majority_label",
            "majority_count",
            "example_items",
            "example_variants",
            "example_datasets",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    upper_bound = majority_correct / max(1, total_examples)
    print(
        "signatures",
        len(rows),
        "examples",
        total_examples,
        "ambiguous_signatures",
        ambiguous_signatures,
        "ambiguous_examples",
        ambiguous_examples,
        "structural_majority_exact_upper_bound",
        f"{upper_bound:.3f}",
    )
    print(f"wrote {output_path}")


def item_structural_examples(item: BenchmarkItem, dataset: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[tuple[str, int, str, str]]] = defaultdict(list)
    for position, block in enumerate(item.context_blocks):
        claim = extract_value_claim(block)
        if claim is None:
            continue
        entity, prop, value = claim
        groups[(entity, prop)].append((block.id, position, block.timestamp, value))

    examples = []
    stale_ids = set(item.stale_ids)
    for claims in groups.values():
        values = {value for _, _, _, value in claims}
        if len(values) < 2:
            continue
        sorted_timestamps = sorted({timestamp for _, _, timestamp, _ in claims})
        timestamp_rank = {timestamp: rank for rank, timestamp in enumerate(sorted_timestamps)}
        value_entries: list[tuple[tuple[int, int, int, int, int], str]] = []
        for value in values:
            value_claims = [(block_id, position, timestamp) for block_id, position, timestamp, claim_value in claims if claim_value == value]
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
        examples.append(
            {
                "dataset": dataset,
                "item_id": item.id,
                "variant": item.metadata.get("variant"),
                "signature": signature,
                "label_pattern": label_pattern,
            }
        )
    return examples


if __name__ == "__main__":
    main()
