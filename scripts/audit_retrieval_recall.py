#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.compressors import CompressionConfig, build_compressor
from context_rot.datasets.io import read_jsonl
from context_rot.eval.runner import stratified_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pre-model gold-evidence recall for retrieval strategies.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--strategies", default="retrieval,windowed_retrieval,evidence_oracle,random_control")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--retrieval-top-k", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--output", default=None)
    parser.add_argument("--details-output", default=None)
    args = parser.parse_args()

    items = read_jsonl(ROOT / args.dataset)
    if args.limit is not None:
        items = stratified_prefix(items, args.limit)
    config = CompressionConfig(
        token_budget=args.token_budget,
        retrieval_top_k=args.retrieval_top_k,
        random_seed=args.seed,
    )
    strategies = [strategy for strategy in args.strategies.split(",") if strategy]
    details = []
    for item in items:
        gold_ids = set(item.gold_evidence_ids)
        if not gold_ids:
            continue
        for strategy in strategies:
            compression = build_compressor(strategy, config).compress(item)
            selected_ids = {block.id for block in compression.bundle.context_blocks}
            overlap = sorted(gold_ids & selected_ids)
            details.append(
                {
                    "item_id": item.id,
                    "strategy": strategy,
                    "gold_evidence_count": len(gold_ids),
                    "selected_block_count": len(selected_ids),
                    "input_tokens_estimate": compression.bundle.input_tokens_estimate,
                    "exact_recall": gold_ids <= selected_ids,
                    "partial_recall": bool(overlap),
                    "overlap_count": len(overlap),
                    "gold_evidence_ids": sorted(gold_ids),
                    "selected_gold_evidence_ids": overlap,
                }
            )
    detail_df = pd.DataFrame(details)
    if detail_df.empty:
        raise SystemExit("No rows with gold evidence IDs found.")
    summary = (
        detail_df.groupby("strategy", sort=False)
        .agg(
            cases=("item_id", "count"),
            exact_recall=("exact_recall", "mean"),
            partial_recall=("partial_recall", "mean"),
            avg_overlap_count=("overlap_count", "mean"),
            avg_selected_blocks=("selected_block_count", "mean"),
            avg_input_tokens=("input_tokens_estimate", "mean"),
        )
        .reset_index()
    )
    if args.output:
        output_path = ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
    if args.details_output:
        details_path = ROOT / args.details_output
        details_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = detail_df.copy()
        for column in ["gold_evidence_ids", "selected_gold_evidence_ids"]:
            serializable[column] = serializable[column].map(json.dumps)
        serializable.to_csv(details_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
