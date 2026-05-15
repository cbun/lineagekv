#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.eval.runner import run_eval


DEFAULT_STRATEGIES = [
    "full",
    "recency",
    "retrieval",
    "dedup_retrieval",
    "recency_weighted_retrieval",
    "contradiction_oracle",
    "structured_state",
    "hybrid",
    "random_control",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run context-rot benchmark evaluation.")
    parser.add_argument("--dataset", default="data/generated/context_rot_v1.jsonl")
    parser.add_argument("--output", default="results/runs/context_rot_v1_heuristic.jsonl")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--model", default="heuristic", choices=["heuristic", "mlx", "transformers", "ollama"])
    parser.add_argument("--mlx-model", default=None)
    parser.add_argument("--transformers-model", default=None)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    strategies = [value for value in args.strategies.split(",") if value]
    path = run_eval(
        dataset_path=ROOT / args.dataset,
        output_path=ROOT / args.output,
        strategies=strategies,
        model_id=args.model,
        limit=args.limit,
        token_budget=args.token_budget,
        retrieval_top_k=args.retrieval_top_k,
        seed=args.seed,
        mlx_model=args.mlx_model,
        transformers_model=args.transformers_model,
        ollama_model=args.ollama_model,
    )
    print(f"wrote results to {path}")


if __name__ == "__main__":
    main()
