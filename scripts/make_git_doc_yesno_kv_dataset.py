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
        description="Convert git-doc phrase lineage items into balanced yes/no currentness checks."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-items", type=int, default=80)
    args = parser.parse_args()

    items = []
    for item in read_jsonl(ROOT / args.input):
        items.extend(yesno_items(item))
        if len(items) >= args.max_items:
            items = items[: args.max_items]
            break
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} git-doc yes/no lineage items to {ROOT / args.output}")


def yesno_items(item: BenchmarkItem) -> list[BenchmarkItem]:
    current_phrase = str(item.metadata.get("target_phrase", item.metadata.get("target_value", item.gold_answer)))
    stale_phrase = str(item.metadata.get("stale_phrase", item.metadata.get("stale_value", "")))
    if not current_phrase or not stale_phrase:
        raise ValueError(f"item {item.id} is missing phrase-level target/stale metadata")
    return [
        yesno_item(item, candidate_phrase=current_phrase, gold_answer="YES", id_suffix="candidate_current"),
        yesno_item(item, candidate_phrase=stale_phrase, gold_answer="NO", id_suffix="candidate_stale"),
    ]


def yesno_item(item: BenchmarkItem, candidate_phrase: str, gold_answer: str, id_suffix: str) -> BenchmarkItem:
    repo_name = str(item.metadata.get("repo_name", "repository"))
    path = str(item.metadata.get("path", "documentation"))
    current_ref = str(item.metadata.get("current_ref", "current revision"))
    query = (
        f"In {repo_name} {path}, is the candidate changed phrase {candidate_phrase!r} current after "
        f"revision {current_ref}? Answer with only YES or NO."
    )
    stale_answer = "NO" if gold_answer == "YES" else "YES"
    return item.model_copy(
        update={
            "id": yesno_item_id(item.id, id_suffix),
            "question_type": "doc_revision_yesno",
            "query": query,
            "gold_answer": gold_answer,
            "metadata": {
                **item.metadata,
                "yesno_currentness": True,
                "phrase_item_id": item.id,
                "candidate_phrase": candidate_phrase,
                "target_phrase": item.metadata.get("target_phrase", item.metadata.get("target_value", item.gold_answer)),
                "stale_phrase": item.metadata.get("stale_phrase", item.metadata.get("stale_value", "")),
                "target_value": gold_answer,
                "stale_value": stale_answer,
            },
        }
    )


def yesno_item_id(item_id: str, id_suffix: str) -> str:
    return f"{item_id.replace('git_doc_phrase_', 'git_doc_yesno_', 1)}_{id_suffix}"


if __name__ == "__main__":
    main()
