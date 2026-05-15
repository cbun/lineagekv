#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.compressors import CompressionConfig, build_compressor
from context_rot.datasets.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Show one benchmark item under several context strategies.")
    parser.add_argument("--dataset", default="data/generated/context_rot_v1.jsonl")
    parser.add_argument("--item-id", default=None)
    parser.add_argument("--strategies", default="full,recency_weighted_retrieval,hybrid,random_control")
    parser.add_argument("--token-budget", type=int, default=1200)
    args = parser.parse_args()

    items = read_jsonl(ROOT / args.dataset)
    item = next((candidate for candidate in items if candidate.id == args.item_id), items[0])
    print(f"Item: {item.id}")
    print(f"Domain: {item.domain}")
    print(f"Query: {item.query}")
    print(f"Gold: {item.gold_answer}")
    print(f"Gold evidence: {', '.join(item.gold_evidence_ids) or '(none)'}")
    print()

    config = CompressionConfig(token_budget=args.token_budget)
    for strategy in [value for value in args.strategies.split(",") if value]:
        result = build_compressor(strategy, config).compress(item)
        selected_ids = [block.id for block in result.bundle.context_blocks]
        stale_selected = sorted(set(selected_ids) & set(item.stale_ids))
        print(f"== {strategy}")
        print(f"tokens: {result.bundle.input_tokens_estimate} / original {result.original_tokens_estimate}")
        print(f"compression_ratio: {result.compression_ratio:.3f}")
        print(f"selected_ids: {', '.join(selected_ids)}")
        print(f"stale_selected: {', '.join(stale_selected) or '(none)'}")
        if strategy in {"hybrid", "structured_state"}:
            first = result.bundle.context_blocks[0]
            if first.type == "structured_state":
                state = json.loads(first.text)
                print(f"structured_relevant_evidence_ids: {state.get('relevant_evidence_ids', [])}")
        print()


if __name__ == "__main__":
    main()
