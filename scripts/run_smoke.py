#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.generate_synthetic import generate_items
from context_rot.datasets.io import write_jsonl
from context_rot.eval.runner import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and run a small smoke test.")
    parser.add_argument("--dataset", default="data/generated/context_rot_smoke.jsonl")
    parser.add_argument("--output", default="results/runs/context_rot_smoke_heuristic.jsonl")
    parser.add_argument("--base-limit", type=int, default=4)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    items = generate_items(base_limit=args.base_limit)
    write_jsonl(items, dataset)
    run_eval(
        dataset_path=dataset,
        output_path=ROOT / args.output,
        strategies=["full", "recency", "retrieval", "random_control", "hybrid"],
        model_id="heuristic",
        limit=args.limit,
        token_budget=800,
    )
    print(f"smoke dataset: {dataset}")
    print(f"smoke results: {ROOT / args.output}")


if __name__ == "__main__":
    main()
