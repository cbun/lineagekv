#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl, write_jsonl
from context_rot.datasets.schema import BenchmarkItem


LABELS = ("ALPHA", "BRAVO")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert git-doc phrase lineage items into two-option currentness QA items."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-items", type=int, default=80)
    parser.add_argument(
        "--label-mode",
        choices=["hash", "both"],
        default="hash",
        help="hash emits one deterministic label assignment per item; both emits both current-label assignments.",
    )
    args = parser.parse_args()

    items = []
    for item in read_jsonl(ROOT / args.input):
        items.extend(option_level_items(item, label_mode=args.label_mode))
        if len(items) >= args.max_items:
            items = items[: args.max_items]
            break
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} git-doc option lineage items to {ROOT / args.output}")


def option_level_items(item: BenchmarkItem, label_mode: str = "hash") -> list[BenchmarkItem]:
    if label_mode == "hash":
        current_label, stale_label = option_labels(item.id)
        return [option_level_item(item, current_label=current_label, stale_label=stale_label)]
    if label_mode == "both":
        return [
            option_level_item(item, current_label="ALPHA", stale_label="BRAVO", id_suffix="alpha_current"),
            option_level_item(item, current_label="BRAVO", stale_label="ALPHA", id_suffix="bravo_current"),
        ]
    raise KeyError(label_mode)


def option_level_item(
    item: BenchmarkItem,
    current_label: str | None = None,
    stale_label: str | None = None,
    id_suffix: str | None = None,
) -> BenchmarkItem:
    current_phrase = str(item.metadata.get("target_value", item.gold_answer))
    stale_phrase = str(item.metadata.get("stale_value", ""))
    if not current_phrase or not stale_phrase:
        raise ValueError(f"item {item.id} is missing phrase-level target/stale metadata")

    if current_label is None or stale_label is None:
        current_label, stale_label = option_labels(item.id)
    if current_label == stale_label:
        raise ValueError("current and stale labels must differ")
    options = {
        current_label: current_phrase,
        stale_label: stale_phrase,
    }
    repo_name = str(item.metadata.get("repo_name", "repository"))
    path = str(item.metadata.get("path", "documentation"))
    current_ref = str(item.metadata.get("current_ref", "current revision"))
    option_text = " ".join(f"{label} = {options[label]!r}." for label in LABELS)
    query = (
        f"In {repo_name} {path}, which option is current after revision {current_ref}? "
        f"{option_text} Answer with only the option label."
    )
    return item.model_copy(
        update={
            "id": option_item_id(item.id, id_suffix),
            "question_type": "doc_revision_option",
            "query": query,
            "gold_answer": current_label,
            "metadata": {
                **item.metadata,
                "option_level": True,
                "phrase_item_id": item.id,
                "option_labels": dict(options),
                "target_phrase": current_phrase,
                "stale_phrase": stale_phrase,
                "target_value": current_label,
                "stale_value": stale_label,
            },
        }
    )


def option_item_id(item_id: str, id_suffix: str | None = None) -> str:
    base = item_id.replace("git_doc_phrase_", "git_doc_option_", 1)
    if id_suffix:
        return f"{base}_{id_suffix}"
    return base


def option_labels(item_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    if digest[0] % 2 == 0:
        return "ALPHA", "BRAVO"
    return "BRAVO", "ALPHA"


if __name__ == "__main__":
    main()
