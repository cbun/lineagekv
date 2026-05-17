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
    parser = argparse.ArgumentParser(
        description="Convert git-doc phrase lineage items into current-block evidence-choice QA items."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-items", type=int, default=80)
    args = parser.parse_args()

    items = []
    for item in read_jsonl(ROOT / args.input):
        items.append(evidence_choice_item(item))
        if len(items) >= args.max_items:
            break
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} git-doc evidence-choice lineage items to {ROOT / args.output}")


def evidence_choice_item(item: BenchmarkItem) -> BenchmarkItem:
    if not item.gold_evidence_ids:
        raise ValueError(f"item {item.id} has no gold evidence id")
    if not item.stale_ids:
        raise ValueError(f"item {item.id} has no stale ids")

    current_block_id = item.gold_evidence_ids[0]
    stale_block_id = item.stale_ids[0]
    repo_name = str(item.metadata.get("repo_name", "repository"))
    path = str(item.metadata.get("path", "documentation"))
    current_ref = str(item.metadata.get("current_ref", "current revision"))
    phrase = str(item.metadata.get("target_phrase", item.metadata.get("target_value", item.gold_answer)))
    query = (
        f"In {repo_name} {path}, which context block is current after revision {current_ref} "
        f"for the changed documentation phrase related to {phrase!r}? "
        "Answer with only the block_id."
    )
    return item.model_copy(
        update={
            "id": item.id.replace("git_doc_phrase_", "git_doc_evidence_choice_", 1),
            "question_type": "doc_revision_evidence_choice",
            "query": query,
            "gold_answer": current_block_id,
            "metadata": {
                **item.metadata,
                "evidence_choice": True,
                "phrase_item_id": item.id,
                "target_phrase": phrase,
                "target_value": current_block_id,
                "stale_value": stale_block_id,
            },
        }
    )


if __name__ == "__main__":
    main()
