#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl, write_jsonl
from context_rot.datasets.schema import BenchmarkItem, ContextBlock


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add explicit supersession-lineage metadata for ledger-driven KV stale-span tests."
    )
    parser.add_argument("--input", default="data/generated/context_rot_v1_adversarial_conflicts.jsonl")
    parser.add_argument("--output", default="data/generated/context_rot_v1_adversarial_lineage.jsonl")
    args = parser.parse_args()

    items = [lineage_item(item) for item in read_jsonl(ROOT / args.input)]
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} lineage-annotated items to {ROOT / args.output}")


def lineage_item(item: BenchmarkItem) -> BenchmarkItem:
    block_order = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    gold_ids = sorted(item.gold_evidence_ids, key=lambda block_id: block_order.get(block_id, len(block_order)))
    stale_ids = sorted(item.stale_ids, key=lambda block_id: block_order.get(block_id, len(block_order)))

    return item.model_copy(
        update={
            "context_blocks": [lineage_block(block, gold_ids, stale_ids) for block in item.context_blocks],
            "metadata": {
                **item.metadata,
                "lineage_annotated": True,
                "lineage_source": "synthetic_gold_and_stale_ids",
            },
        }
    )


def lineage_block(block: ContextBlock, gold_ids: list[str], stale_ids: list[str]) -> ContextBlock:
    metadata = dict(block.metadata)
    if block.id in gold_ids and stale_ids:
        metadata["supersedes"] = merge_ids(metadata.get("supersedes"), stale_ids)
    if block.id in stale_ids and gold_ids:
        metadata["superseded_by"] = merge_ids(metadata.get("superseded_by"), gold_ids)
    return block.model_copy(update={"metadata": metadata})


def merge_ids(existing: Any, additions: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*metadata_id_list(existing), *additions]:
        if value not in merged:
            merged.append(value)
    return merged


def metadata_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


if __name__ == "__main__":
    main()
