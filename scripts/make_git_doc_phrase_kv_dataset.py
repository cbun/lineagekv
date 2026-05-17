#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl, write_jsonl
from context_rot.datasets.schema import BenchmarkItem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert git-doc sentence lineage items into short changed-phrase QA items."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-items", type=int, default=80)
    args = parser.parse_args()

    items = []
    for item in read_jsonl(ROOT / args.input):
        phrase_item = phrase_level_item(item)
        if phrase_item is None:
            continue
        items.append(phrase_item)
        if len(items) >= args.max_items:
            break
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} git-doc phrase lineage items to {ROOT / args.output}")


def phrase_level_item(item: BenchmarkItem) -> BenchmarkItem | None:
    old_sentence = str(item.metadata.get("stale_value", ""))
    new_sentence = str(item.metadata.get("target_value", item.gold_answer))
    old_phrase, new_phrase = changed_phrases(old_sentence, new_sentence)
    if not useful_phrase_pair(old_phrase, new_phrase):
        return None

    repo_name = str(item.metadata.get("repo_name", "repository"))
    path = str(item.metadata.get("path", "documentation"))
    current_ref = str(item.metadata.get("current_ref", "current revision"))
    query = (
        f"In {repo_name} {path}, which short changed phrase is current after revision {current_ref}? "
        "Answer with only the current changed phrase."
    )
    return item.model_copy(
        update={
            "id": item.id.replace("git_doc_", "git_doc_phrase_", 1),
            "question_type": "doc_revision_phrase",
            "query": query,
            "gold_answer": new_phrase,
            "metadata": {
                **item.metadata,
                "phrase_level": True,
                "sentence_target_value": new_sentence,
                "sentence_stale_value": old_sentence,
                "target_value": new_phrase,
                "stale_value": old_phrase,
            },
        }
    )


def changed_phrases(old_sentence: str, new_sentence: str) -> tuple[str, str]:
    old_tokens = phrase_tokens(old_sentence)
    new_tokens = phrase_tokens(new_sentence)
    matcher = SequenceMatcher(None, old_tokens, new_tokens)
    old_parts: list[str] = []
    new_parts: list[str] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_parts.extend(old_tokens[old_start:old_end])
        new_parts.extend(new_tokens[new_start:new_end])
    return cleanup_phrase(old_parts), cleanup_phrase(new_parts)


def phrase_tokens(sentence: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9._'-]*", sentence)


def cleanup_phrase(tokens: list[str]) -> str:
    return " ".join(tokens).strip(" ,.;:-")


def useful_phrase_pair(old_phrase: str, new_phrase: str) -> bool:
    if not old_phrase or not new_phrase:
        return False
    if old_phrase.lower() == new_phrase.lower():
        return False
    if not 2 <= len(old_phrase) <= 90 or not 2 <= len(new_phrase) <= 90:
        return False
    if len(new_phrase.split()) > 10 or len(old_phrase.split()) > 10:
        return False
    if any(char in old_phrase + new_phrase for char in "{}<>|"):
        return False
    return True


if __name__ == "__main__":
    main()
