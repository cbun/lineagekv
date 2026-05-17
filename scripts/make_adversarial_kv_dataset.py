#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl, write_jsonl
from context_rot.datasets.schema import BenchmarkItem, ContextBlock


ADVERSARIAL_VARIANTS = [
    "repeated_current",
    "singleton_stale_late",
    "both_values_repeated",
    "paraphrased_stale_duplicates",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cue-stripped adversarial KV conflict cases.")
    parser.add_argument("--input", default="data/generated/context_rot_v1_cue_stripped.jsonl")
    parser.add_argument("--output", default="data/generated/context_rot_v1_adversarial_conflicts.jsonl")
    parser.add_argument("--base-variant", default="current_late")
    parser.add_argument("--distractor-ratio", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    base_items = [
        item
        for item in read_jsonl(ROOT / args.input)
        if item.metadata.get("variant") == args.base_variant
        and float(item.metadata.get("distractor_ratio", -1.0)) == args.distractor_ratio
    ][: args.limit]
    items: list[BenchmarkItem] = []
    for item in base_items:
        for variant in ADVERSARIAL_VARIANTS:
            items.append(build_adversarial_item(item, variant))
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} adversarial items to {ROOT / args.output}")


def build_adversarial_item(item: BenchmarkItem, variant: str) -> BenchmarkItem:
    entity = str(item.metadata["entity"])
    prop = str(item.metadata["property"])
    current_value = str(item.metadata["target_value"])
    stale_value = str(item.metadata["stale_value"])
    base_index = int(item.metadata["base_index"])
    base_time = min(block.parsed_timestamp for block in item.context_blocks)
    distractors = [block for block in item.context_blocks if block.id in set(item.distractor_ids)]

    current = claim_block(item, f"b{base_index:03d}_adv_current", 1, entity, prop, current_value, is_stale=False)
    stale = claim_block(item, f"b{base_index:03d}_adv_stale", 0, entity, prop, stale_value, is_stale=True)

    if variant == "repeated_current":
        current_blocks = [
            current.model_copy(update={"id": f"b{base_index:03d}_adv_current_dup{idx + 1:02d}", "timestamp": iso(base_time + timedelta(days=3, minutes=idx))})
            for idx in range(3)
        ]
        blocks = [stale, *distractors, *current_blocks]
        gold_ids = [current_blocks[-1].id]
        stale_ids = [stale.id]
    elif variant == "singleton_stale_late":
        late_stale = stale.model_copy(update={"timestamp": iso(base_time + timedelta(days=8))})
        blocks = [current, *distractors, late_stale]
        gold_ids = [current.id]
        stale_ids = [late_stale.id]
    elif variant == "both_values_repeated":
        current_blocks = [
            current.model_copy(update={"id": f"b{base_index:03d}_adv_current_dup{idx + 1:02d}", "timestamp": iso(base_time + timedelta(days=2, minutes=idx))})
            for idx in range(2)
        ]
        stale_blocks = [
            stale.model_copy(update={"id": f"b{base_index:03d}_adv_stale_dup{idx + 1:02d}", "timestamp": iso(base_time + timedelta(days=8, minutes=idx))})
            for idx in range(2)
        ]
        blocks = [*current_blocks, *distractors, *stale_blocks]
        gold_ids = [current_blocks[-1].id]
        stale_ids = [block.id for block in stale_blocks]
    elif variant == "paraphrased_stale_duplicates":
        stale_blocks = [
            claim_block(
                item,
                f"b{base_index:03d}_adv_stale_para{idx + 1:02d}",
                8 + idx,
                entity,
                prop,
                stale_paraphrase(stale_value, idx),
                is_stale=True,
            )
            for idx in range(3)
        ]
        blocks = [current, *distractors, *stale_blocks]
        gold_ids = [current.id]
        stale_ids = [block.id for block in stale_blocks]
    else:
        raise KeyError(variant)

    return item.model_copy(
        update={
            "id": f"{item.domain}_{base_index:03d}_{variant}_d{int(float(item.metadata['distractor_ratio']) * 100):02d}",
            "context_blocks": blocks,
            "gold_evidence_ids": gold_ids,
            "stale_ids": stale_ids,
            "distractor_ids": [block.id for block in distractors],
            "metadata": {
                **item.metadata,
                "variant": variant,
                "adversarial_conflict": True,
                "target_value": current_value,
                "stale_value": stale_value,
                "duplicate_wrong_count": len(stale_ids) - 1,
            },
        }
    )


def claim_block(
    item: BenchmarkItem,
    block_id: str,
    day_offset: int,
    entity: str,
    prop: str,
    value: str,
    is_stale: bool,
) -> ContextBlock:
    base_time = min(block.parsed_timestamp for block in item.context_blocks)
    return ContextBlock(
        id=block_id,
        timestamp=iso(base_time + timedelta(days=day_offset)),
        type=str(item.metadata.get("block_type", item.domain)),
        trust="high",
        status="unknown",
        text=f"Note for {entity}: {prop} was '{value}'.",
        metadata={"adversarial_stale": is_stale},
    )


def stale_paraphrase(value: str, idx: int) -> str:
    suffixes = ["old wording", "prior note", "legacy phrasing"]
    return f"{value} ({suffixes[idx]})"


def iso(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
