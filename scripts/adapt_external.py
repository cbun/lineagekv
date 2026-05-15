#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import write_jsonl
from context_rot.datasets.load_external import adapt_locomo, adapt_longmemeval_s, adapt_nolima


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt fetched external datasets into BenchmarkItem JSONL.")
    parser.add_argument("--longmemeval-limit", type=int, default=50)
    parser.add_argument("--locomo-limit", type=int, default=100)
    parser.add_argument("--nolima-limit", type=int, default=50)
    args = parser.parse_args()

    outputs = [
        (
            "data/processed/longmemeval_s_v0.jsonl",
            adapt_longmemeval_s(
                ROOT / "data/raw/kellyhongg__cleaned-longmemeval-s/longmemeval_s_cleaned.csv",
                limit=args.longmemeval_limit,
            ),
        ),
        (
            "data/processed/locomo_v0.jsonl",
            adapt_locomo(ROOT / "data/raw/Percena__locomo-mc10/raw/locomo10.json", limit=args.locomo_limit),
        ),
        (
            "data/processed/nolima_v0.jsonl",
            adapt_nolima(ROOT / "data/raw/amodaresi__NoLiMa", limit=args.nolima_limit),
        ),
    ]
    for rel_path, items in outputs:
        count = write_jsonl(items, ROOT / rel_path)
        print(f"wrote {count} adapted cases to {ROOT / rel_path}")


if __name__ == "__main__":
    main()
