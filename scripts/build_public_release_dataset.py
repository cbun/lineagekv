#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl, write_jsonl
from context_rot.datasets.schema import BenchmarkItem


SOURCES = [
    ("update_nonce_option", "data/generated/agent_memory_update_nonce_option_probe80.jsonl"),
    ("action", "data/generated/agent_memory_action_probe64.jsonl"),
    ("messy_multi_update", "data/generated/agent_memory_messy_probe40.jsonl"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public stale-memory repair benchmark bundle.")
    parser.add_argument("--output", default="data/public/stale_memory_repair_benchmark_v1.jsonl")
    parser.add_argument("--metadata-output", default="data/public/stale_memory_repair_benchmark_v1_metadata.json")
    args = parser.parse_args()

    items: list[BenchmarkItem] = []
    source_counts: Counter[str] = Counter()
    for source_name, path in SOURCES:
        source_items = read_jsonl(ROOT / path)
        source_counts[source_name] += len(source_items)
        for idx, item in enumerate(source_items):
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "release_benchmark": "stale_memory_repair_v1",
                    "release_source": source_name,
                    "release_source_path": path,
                    "release_source_item_id": item.id,
                    "release_index_within_source": idx,
                }
            )
            items.append(
                item.model_copy(
                    update={
                        "id": f"{source_name}__{item.id}",
                        "metadata": metadata,
                    }
                )
            )

    written = write_jsonl(items, ROOT / args.output)
    metadata = {
        "name": "stale_memory_repair_benchmark_v1",
        "records": written,
        "sources": dict(source_counts),
        "source_paths": {source_name: path for source_name, path in SOURCES},
        "description": (
            "Composite stale/current assistant-memory benchmark for cache-repair research. "
            "Rows preserve gold current evidence IDs, stale IDs, distractor IDs, timestamps, "
            "and memory metadata needed for lineage and text+timestamp currentness audits."
        ),
        "license_note": "Generated benchmark data authored inside this repository; no private user data.",
    }
    output_path = ROOT / args.metadata_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {written} rows to {ROOT / args.output}")
    print(f"wrote {ROOT / args.metadata_output}")


if __name__ == "__main__":
    main()
