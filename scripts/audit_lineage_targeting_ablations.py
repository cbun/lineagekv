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
from kv_value_ablation_probe import detected_lineage_stale_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether lineage targeting depends on stale labels, block status, or explicit lineage edges."
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
        "label_status_scrub_exact": 0,
        "lineage_removed_empty": 0,
        "lineage_removed_exact": 0,
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
        "label_status_scrub_exact",
        "lineage_removed_empty",
        "lineage_removed_exact",
    ]:
        print(key, f"{totals[key] / totals['items']:.3f}", f"{totals[key]}/{totals['items']}")
    print(f"wrote {output_path}")


def audit_item(item: BenchmarkItem) -> dict[str, Any]:
    gold = set(item.stale_ids)
    predictions = {
        "original": set(detected_lineage_stale_ids(item)),
        "label_scrubbed": set(detected_lineage_stale_ids(item.model_copy(update={"stale_ids": []}))),
        "status_scrubbed": set(detected_lineage_stale_ids(status_scrubbed_item(item))),
        "label_status_scrubbed": set(
            detected_lineage_stale_ids(status_scrubbed_item(item).model_copy(update={"stale_ids": []}))
        ),
        "lineage_removed": set(detected_lineage_stale_ids(lineage_removed_item(item))),
    }
    return {
        "item_id": item.id,
        "gold_ids": " ".join(sorted(gold)),
        "original_predicted_ids": " ".join(sorted(predictions["original"])),
        "label_scrubbed_predicted_ids": " ".join(sorted(predictions["label_scrubbed"])),
        "status_scrubbed_predicted_ids": " ".join(sorted(predictions["status_scrubbed"])),
        "label_status_scrubbed_predicted_ids": " ".join(sorted(predictions["label_status_scrubbed"])),
        "lineage_removed_predicted_ids": " ".join(sorted(predictions["lineage_removed"])),
        "label_scrub_exact": predictions["label_scrubbed"] == gold,
        "status_scrub_exact": predictions["status_scrubbed"] == gold,
        "label_status_scrub_exact": predictions["label_status_scrubbed"] == gold,
        "lineage_removed_empty": not predictions["lineage_removed"],
        "lineage_removed_exact": predictions["lineage_removed"] == gold,
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


if __name__ == "__main__":
    main()
