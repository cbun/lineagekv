#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.generate_synthetic import DISTRACTOR_RATIOS, VARIANTS, generate_items
from context_rot.datasets.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic context-rot benchmark cases.")
    parser.add_argument("--output", default="data/generated/context_rot_v1.jsonl")
    parser.add_argument("--base-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--distractor-ratios", default=",".join(str(v) for v in DISTRACTOR_RATIOS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()

    ratios = [float(value) for value in args.distractor_ratios.split(",") if value]
    variants = [value for value in args.variants.split(",") if value]
    items = generate_items(base_limit=args.base_limit, distractor_ratios=ratios, variants=variants, seed=args.seed)
    count = write_jsonl(items, ROOT / args.output)
    base_count = len({item.metadata["base_index"] for item in items})
    print(f"wrote {count} cases from {base_count} base scenarios to {ROOT / args.output}")


if __name__ == "__main__":
    main()
