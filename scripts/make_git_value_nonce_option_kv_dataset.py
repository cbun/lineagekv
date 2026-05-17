#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl, write_jsonl
from context_rot.datasets.schema import BenchmarkItem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert git value lineage items into nonce-coded option currentness items."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-items", type=int, default=80)
    args = parser.parse_args()

    items = []
    for item in read_jsonl(ROOT / args.input):
        items.extend(nonce_option_items(item))
        if len(items) >= args.max_items:
            items = items[: args.max_items]
            break
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} git-value nonce-option lineage items to {ROOT / args.output}")


def nonce_option_items(item: BenchmarkItem) -> list[BenchmarkItem]:
    return [
        nonce_option_item(item, assignment="code_a_current"),
        nonce_option_item(item, assignment="code_b_current"),
    ]


def nonce_option_item(item: BenchmarkItem, assignment: str = "code_a_current") -> BenchmarkItem:
    current_value = str(item.metadata.get("target_value", item.gold_answer))
    stale_value = str(item.metadata.get("stale_value", ""))
    if not current_value or not stale_value:
        raise ValueError(f"item {item.id} is missing target/stale value metadata")

    code_a, code_b = nonce_codes(item.id)
    if assignment == "code_a_current":
        options = {code_a: current_value, code_b: stale_value}
        current_code = code_a
        stale_code = code_b
    elif assignment == "code_b_current":
        options = {code_a: stale_value, code_b: current_value}
        current_code = code_b
        stale_code = code_a
    else:
        raise KeyError(assignment)

    repo_name = str(item.metadata.get("repo_name", "repository"))
    path = str(item.metadata.get("path", "file"))
    key_path = str(item.metadata.get("key_path", "value"))
    current_ref = str(item.metadata.get("current_ref", "current revision"))
    option_text = " ".join(f"{code} = {value!r}." for code, value in options.items())
    query = (
        f"In {repo_name} {path}::{key_path}, which option code names the current value "
        f"after revision {current_ref}? {option_text} Answer with only the option code."
    )
    return item.model_copy(
        update={
            "id": nonce_option_item_id(item.id, assignment),
            "question_type": "git_value_nonce_option",
            "query": query,
            "gold_answer": current_code,
            "metadata": {
                **item.metadata,
                "nonce_option": True,
                "source_item_id": item.id,
                "nonce_assignment": assignment,
                "option_labels": dict(options),
                "target_value": current_code,
                "stale_value": stale_code,
                "target_raw_value": current_value,
                "stale_raw_value": stale_value,
            },
        }
    )


def nonce_codes(item_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    encoded = base64.b32encode(digest).decode("ascii").lower()
    return f"c{encoded[:5]}", f"c{encoded[5:10]}"


def nonce_option_item_id(item_id: str, assignment: str) -> str:
    return f"{item_id.replace('git_history_', 'git_value_nonce_', 1)}_{assignment}"


if __name__ == "__main__":
    main()
