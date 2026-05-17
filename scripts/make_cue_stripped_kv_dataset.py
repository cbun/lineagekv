#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl, write_jsonl
from context_rot.datasets.schema import BenchmarkItem, ContextBlock


CLAIM_KEYS = {"entity", "property", "value"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a cue-stripped copy of the synthetic KV dataset for detector stress tests."
    )
    parser.add_argument("--input", default="data/generated/context_rot_v1.jsonl")
    parser.add_argument("--output", default="data/generated/context_rot_v1_cue_stripped.jsonl")
    parser.add_argument(
        "--keep-claim-metadata",
        action="store_true",
        help="Keep block-level entity/property/value metadata. By default only parseable neutral text remains.",
    )
    args = parser.parse_args()

    items = [
        cue_stripped_item(item, drop_claim_metadata=not args.keep_claim_metadata)
        for item in read_jsonl(ROOT / args.input)
    ]
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} cue-stripped items to {ROOT / args.output}")


def cue_stripped_item(item: BenchmarkItem, drop_claim_metadata: bool = True) -> BenchmarkItem:
    return item.model_copy(
        update={
            "context_blocks": [cue_stripped_block(block, drop_claim_metadata) for block in item.context_blocks],
            "metadata": {**item.metadata, "cue_stripped": True, "claim_metadata_dropped": drop_claim_metadata},
        }
    )


def cue_stripped_block(block: ContextBlock, drop_claim_metadata: bool = True) -> ContextBlock:
    metadata = dict(block.metadata)
    entity = metadata.get("entity")
    prop = metadata.get("property")
    value = metadata.get("value")
    text = block.text
    if entity and prop and value:
        text = f"Note for {entity}: {prop} was '{value}'."
        if drop_claim_metadata:
            metadata = {key: value for key, value in metadata.items() if key not in CLAIM_KEYS}
    elif "supports" in metadata and entity:
        text = f"Decision record for {entity}: see the dated notes in this bundle."
    else:
        text = strip_lexical_cues(text)
    return block.model_copy(update={"text": text, "metadata": metadata})


def strip_lexical_cues(text: str) -> str:
    replacements: list[tuple[str, str]] = [
        ("Current decision", "Decision note"),
        ("current decision", "decision note"),
        ("superseded duplicates", "older duplicates"),
        ("superseded", "older"),
        ("supersedes", "replaces"),
        ("Earlier note", "Note"),
        ("earlier value", "older value"),
    ]
    stripped = text
    for old, new in replacements:
        stripped = stripped.replace(old, new)
    return stripped


if __name__ == "__main__":
    main()
