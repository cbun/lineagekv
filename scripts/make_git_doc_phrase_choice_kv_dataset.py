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
        description="Convert git-doc phrase lineage items into candidate-phrase currentness choice items."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-items", type=int, default=80)
    args = parser.parse_args()

    items = []
    for item in read_jsonl(ROOT / args.input):
        items.extend(phrase_choice_items(item))
        if len(items) >= args.max_items:
            items = items[: args.max_items]
            break
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} git-doc phrase-choice lineage items to {ROOT / args.output}")


def phrase_choice_items(item: BenchmarkItem) -> list[BenchmarkItem]:
    return [
        phrase_choice_item(item, candidate_order="current_first"),
        phrase_choice_item(item, candidate_order="stale_first"),
    ]


def phrase_choice_item(item: BenchmarkItem, candidate_order: str = "current_first") -> BenchmarkItem:
    current_phrase = str(item.metadata.get("target_phrase", item.metadata.get("target_value", item.gold_answer)))
    stale_phrase = str(item.metadata.get("stale_phrase", item.metadata.get("stale_value", "")))
    if not current_phrase or not stale_phrase:
        raise ValueError(f"item {item.id} is missing phrase-level target/stale metadata")
    if candidate_order == "current_first":
        candidates = [current_phrase, stale_phrase]
    elif candidate_order == "stale_first":
        candidates = [stale_phrase, current_phrase]
    else:
        raise KeyError(candidate_order)

    repo_name = str(item.metadata.get("repo_name", "repository"))
    path = str(item.metadata.get("path", "documentation"))
    current_ref = str(item.metadata.get("current_ref", "current revision"))
    query = (
        f"In {repo_name} {path}, which candidate phrase is current after revision {current_ref}? "
        f"Candidate phrases: {candidates[0]!r}; {candidates[1]!r}. "
        "Answer with only the current phrase."
    )
    return item.model_copy(
        update={
            "id": phrase_choice_item_id(item.id, candidate_order),
            "question_type": "doc_revision_phrase_choice",
            "query": query,
            "gold_answer": current_phrase,
            "metadata": {
                **item.metadata,
                "phrase_choice": True,
                "phrase_item_id": item.id,
                "candidate_order": candidate_order,
                "candidate_phrases": candidates,
                "target_phrase": current_phrase,
                "stale_phrase": stale_phrase,
                "target_value": current_phrase,
                "stale_value": stale_phrase,
            },
        }
    )


def phrase_choice_item_id(item_id: str, candidate_order: str) -> str:
    return f"{item_id.replace('git_doc_phrase_', 'git_doc_phrase_choice_', 1)}_{candidate_order}"


if __name__ == "__main__":
    main()
