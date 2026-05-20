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
    (
        "assistant_memory_public_v1",
        "data/public/stale_memory_repair_benchmark_v1.jsonl",
        "generated assistant-memory stale/current traces",
    ),
    (
        "public_git_history_doc_nonce",
        "data/generated/git_doc_nonce_option_audited_history_four_repo_probe80.jsonl",
        "real public Git history document revisions from express, axios, flask, and requests",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mixed universality trace benchmark.")
    parser.add_argument("--output", default="data/public/stale_memory_universality_traces_v1.jsonl")
    parser.add_argument(
        "--metadata-output",
        default="data/public/stale_memory_universality_traces_v1_metadata.json",
    )
    args = parser.parse_args()

    items: list[BenchmarkItem] = []
    source_counts: Counter[str] = Counter()
    source_descriptions: dict[str, str] = {}
    source_paths: dict[str, str] = {}
    for source_name, path, description in SOURCES:
        source_items = read_jsonl(ROOT / path)
        source_counts[source_name] += len(source_items)
        source_descriptions[source_name] = description
        source_paths[source_name] = path
        for idx, item in enumerate(source_items):
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "universality_benchmark": "stale_memory_universality_traces_v1",
                    "universality_source": source_name,
                    "universality_source_path": path,
                    "universality_source_description": description,
                    "universality_source_item_id": item.id,
                    "universality_index_within_source": idx,
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
        "name": "stale_memory_universality_traces_v1",
        "records": written,
        "sources": dict(source_counts),
        "source_paths": source_paths,
        "source_descriptions": source_descriptions,
        "description": (
            "Mixed stale/current trace benchmark for universality checks. It combines the public "
            "assistant-memory release with less-generated public Git-history document-revision traces."
        ),
        "license_note": (
            "Assistant-memory rows are generated in this repository. Git-history rows are derived "
            "from public repository histories already present under external/git_history."
        ),
    }
    output_path = ROOT / args.metadata_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {written} rows to {ROOT / args.output}")
    print(f"wrote {ROOT / args.metadata_output}")


if __name__ == "__main__":
    main()
