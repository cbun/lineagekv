#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.io import read_jsonl
from context_rot.datasets.schema import BenchmarkItem
from kv_value_ablation_probe import detected_text_timestamp_stale_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether text+timestamp stale targeting depends on labels, status, "
            "lineage edges, revision metadata, item metadata, block text, or timestamps."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--distractor-ratio", type=float, default=0.0)
    parser.add_argument("--variants", default="current_first,stale_first")
    args = parser.parse_args()

    allowed_variants = {value for value in args.variants.split(",") if value}
    rows = []
    totals = {
        "items": 0,
        "label_scrub_exact": 0,
        "status_scrub_exact": 0,
        "lineage_removed_exact": 0,
        "block_metadata_removed_exact": 0,
        "item_metadata_removed_exact": 0,
        "all_metadata_removed_exact": 0,
        "block_text_removed_empty": 0,
        "timestamp_tied_empty": 0,
    }
    for item in read_jsonl(ROOT / args.dataset):
        if float(item.metadata.get("distractor_ratio", -1.0)) != args.distractor_ratio:
            continue
        if allowed_variants and str(item.metadata.get("variant", "")) not in allowed_variants:
            continue
        row = audit_item(item)
        for key in totals:
            if key == "items":
                continue
            totals[key] += int(row[key])
        totals["items"] += 1
        rows.append(row)

    if not rows:
        raise SystemExit("No matching items found.")
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("items", totals["items"])
    for key in [
        "label_scrub_exact",
        "status_scrub_exact",
        "lineage_removed_exact",
        "block_metadata_removed_exact",
        "item_metadata_removed_exact",
        "all_metadata_removed_exact",
        "block_text_removed_empty",
        "timestamp_tied_empty",
    ]:
        print(key, f"{totals[key] / totals['items']:.3f}", f"{totals[key]}/{totals['items']}")
    print(f"wrote {output_path}")


def audit_item(item: BenchmarkItem) -> dict[str, Any]:
    gold = set(item.stale_ids)
    predictions = {
        "original": set(detected_text_timestamp_stale_ids(item)),
        "label_scrubbed": set(detected_text_timestamp_stale_ids(item.model_copy(update={"stale_ids": []}))),
        "status_scrubbed": set(detected_text_timestamp_stale_ids(status_scrubbed_item(item))),
        "lineage_removed": set(detected_text_timestamp_stale_ids(lineage_removed_item(item))),
        "block_metadata_removed": set(detected_text_timestamp_stale_ids(block_metadata_removed_item(item))),
        "item_metadata_removed": set(detected_text_timestamp_stale_ids(item.model_copy(update={"metadata": {}}))),
        "all_metadata_removed": set(
            detected_text_timestamp_stale_ids(
                block_metadata_removed_item(status_scrubbed_item(item)).model_copy(
                    update={"stale_ids": [], "metadata": {}}
                )
            )
        ),
        "block_text_removed": set(detected_text_timestamp_stale_ids(block_text_removed_item(item))),
        "timestamp_tied": set(detected_text_timestamp_stale_ids(timestamp_tied_item(item))),
    }
    return {
        "item_id": item.id,
        "gold_ids": " ".join(sorted(gold)),
        "original_predicted_ids": " ".join(sorted(predictions["original"])),
        "label_scrubbed_predicted_ids": " ".join(sorted(predictions["label_scrubbed"])),
        "status_scrubbed_predicted_ids": " ".join(sorted(predictions["status_scrubbed"])),
        "lineage_removed_predicted_ids": " ".join(sorted(predictions["lineage_removed"])),
        "block_metadata_removed_predicted_ids": " ".join(sorted(predictions["block_metadata_removed"])),
        "item_metadata_removed_predicted_ids": " ".join(sorted(predictions["item_metadata_removed"])),
        "all_metadata_removed_predicted_ids": " ".join(sorted(predictions["all_metadata_removed"])),
        "block_text_removed_predicted_ids": " ".join(sorted(predictions["block_text_removed"])),
        "timestamp_tied_predicted_ids": " ".join(sorted(predictions["timestamp_tied"])),
        "label_scrub_exact": predictions["label_scrubbed"] == gold,
        "status_scrub_exact": predictions["status_scrubbed"] == gold,
        "lineage_removed_exact": predictions["lineage_removed"] == gold,
        "block_metadata_removed_exact": predictions["block_metadata_removed"] == gold,
        "item_metadata_removed_exact": predictions["item_metadata_removed"] == gold,
        "all_metadata_removed_exact": predictions["all_metadata_removed"] == gold,
        "block_text_removed_empty": not predictions["block_text_removed"],
        "timestamp_tied_empty": not predictions["timestamp_tied"],
    }


def status_scrubbed_item(item: BenchmarkItem) -> BenchmarkItem:
    return item.model_copy(
        update={"context_blocks": [block.model_copy(update={"status": "unknown"}) for block in item.context_blocks]}
    )


def lineage_removed_item(item: BenchmarkItem) -> BenchmarkItem:
    blocks = []
    for block in item.context_blocks:
        metadata = dict(block.metadata)
        metadata.pop("supersedes", None)
        metadata.pop("superseded_by", None)
        blocks.append(block.model_copy(update={"metadata": metadata, "status": "unknown"}))
    return item.model_copy(update={"context_blocks": blocks, "stale_ids": []})


def block_metadata_removed_item(item: BenchmarkItem) -> BenchmarkItem:
    return item.model_copy(
        update={
            "context_blocks": [
                block.model_copy(update={"metadata": {}, "status": "unknown"})
                for block in item.context_blocks
            ],
            "stale_ids": [],
        }
    )


def block_text_removed_item(item: BenchmarkItem) -> BenchmarkItem:
    return item.model_copy(
        update={
            "context_blocks": [
                block.model_copy(update={"text": "", "metadata": {}, "status": "unknown"})
                for block in item.context_blocks
            ],
            "stale_ids": [],
            "metadata": {},
        }
    )


def timestamp_tied_item(item: BenchmarkItem) -> BenchmarkItem:
    if not item.context_blocks:
        return item
    timestamp = item.context_blocks[0].timestamp
    return item.model_copy(
        update={
            "context_blocks": [
                block.model_copy(update={"timestamp": timestamp, "metadata": {}, "status": "unknown"})
                for block in item.context_blocks
            ],
            "stale_ids": [],
            "metadata": {},
        }
    )


if __name__ == "__main__":
    main()
