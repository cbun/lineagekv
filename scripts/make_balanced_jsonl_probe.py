#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl, write_jsonl
from context_rot.datasets.schema import BenchmarkItem


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a round-robin balanced JSONL probe from multiple inputs.")
    parser.add_argument("--input", action="append", required=True, help="Input JSONL. Repeat once per source.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-input", type=int, required=True)
    args = parser.parse_args()

    input_items = [read_jsonl(ROOT / path)[: args.per_input] for path in args.input]
    items = round_robin(input_items)
    count = write_jsonl(items, ROOT / args.output)
    counts = ", ".join(f"{Path(path).name}={len(rows)}" for path, rows in zip(args.input, input_items))
    print(f"wrote {count} balanced items to {ROOT / args.output} ({counts})")


def round_robin(groups: list[list[BenchmarkItem]]) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    max_len = max((len(group) for group in groups), default=0)
    for idx in range(max_len):
        for group in groups:
            if idx < len(group):
                items.append(group[idx])
    return items


if __name__ == "__main__":
    main()
