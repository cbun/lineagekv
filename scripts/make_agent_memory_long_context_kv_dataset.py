#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl, write_jsonl
from context_rot.datasets.schema import BenchmarkItem, ContextBlock


MEMORY_TOPICS = [
    ("profile.timezone", "Scheduling defaults to Central Time unless a calendar invite says otherwise."),
    ("assistant.voice.default", "Voice replies should stay calm, concise, and low-drama."),
    ("repo.install.package_manager", "Use pnpm for JavaScript dependency installation."),
    ("ops.escalation.primary", "Urgent incidents escalate through OpsGenie Primary."),
    ("sales.crm.default_view", "Open the revenue pipeline in the expansion view by default."),
    ("writing.status_update.format", "Weekly status updates should use bullets with blockers first."),
    ("calendar.focus_block.window", "Deep-work blocks usually sit between 10:00 and 12:00."),
    ("analytics.metric.owner", "North-star dashboard questions go to the analytics lead."),
    ("support.refund.policy", "Refund requests over $250 require manager review."),
    ("engineering.review.channel", "Code-review coordination happens in #eng-reviews."),
    ("finance.vendor.tax_form", "Vendor onboarding requires a current W-9 before payment."),
    ("design.asset.source", "Product screenshots should come from the latest staging build."),
    ("research.paper_library", "Paper notes belong in the shared research index."),
    ("security.secret_storage", "Secrets belong in the managed vault, not in chat logs."),
    ("profile.travel.seat", "Default flight seat preference is aisle when available."),
    ("ops.deploy.window", "Routine deploys should avoid Friday afternoon."),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inflate action-memory cases into long persistent-cache cases.")
    parser.add_argument("--source", default="data/generated/agent_memory_action_test32.jsonl")
    parser.add_argument("--output", default="data/generated/agent_memory_long_context_test8.jsonl")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--distractor-blocks", type=int, default=96)
    parser.add_argument("--sentences-per-block", type=int, default=5)
    parser.add_argument("--variant", default="long_context")
    args = parser.parse_args()

    source_items = read_jsonl(ROOT / args.source)[: args.limit]
    items = [
        inflate_item(
            item,
            distractor_blocks=args.distractor_blocks,
            sentences_per_block=args.sentences_per_block,
            variant=args.variant,
        )
        for item in source_items
    ]
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} long-context memory items to {ROOT / args.output}")


def inflate_item(
    item: BenchmarkItem,
    distractor_blocks: int,
    sentences_per_block: int,
    variant: str,
) -> BenchmarkItem:
    target_ids = set(item.gold_evidence_ids) | set(item.stale_ids)
    target_blocks = [copy_block(block) for block in item.context_blocks if block.id in target_ids]
    original_distractors = [copy_block(block) for block in item.context_blocks if block.id not in target_ids]
    synthetic = [
        synthetic_memory_block(item.id, idx, sentences_per_block)
        for idx in range(distractor_blocks)
    ]
    context_blocks = interleave_long_context(target_blocks, original_distractors, synthetic, item.id)
    metadata = dict(item.metadata)
    metadata.update(
        {
            "source_item_id": item.id,
            "variant": variant,
            "distractor_ratio": 0.95,
            "long_context_distractor_blocks": distractor_blocks,
            "sentences_per_distractor_block": sentences_per_block,
        }
    )
    return BenchmarkItem(
        id=f"{item.id}_{variant}_d{distractor_blocks}_s{sentences_per_block}",
        domain="agent_memory_long_context",
        question_type=item.question_type,
        context_blocks=context_blocks,
        query=item.query,
        gold_answer=item.gold_answer,
        gold_evidence_ids=item.gold_evidence_ids,
        stale_ids=item.stale_ids,
        distractor_ids=[block.id for block in context_blocks if block.id not in target_ids],
        metadata=metadata,
    )


def copy_block(block: ContextBlock) -> ContextBlock:
    return ContextBlock(**block.model_dump())


def interleave_long_context(
    target_blocks: list[ContextBlock],
    original_distractors: list[ContextBlock],
    synthetic: list[ContextBlock],
    item_id: str,
) -> list[ContextBlock]:
    digest = int(hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:8], 16)
    insert_a = digest % max(1, len(synthetic) // 3)
    insert_b = (len(synthetic) * 2 // 3) + (digest % max(1, len(synthetic) // 4))
    blocks = synthetic[:insert_a]
    if original_distractors:
        blocks.append(original_distractors[0])
    blocks.append(target_blocks[0])
    blocks.extend(synthetic[insert_a:insert_b])
    if len(original_distractors) > 1:
        blocks.append(original_distractors[1])
    blocks.append(target_blocks[1])
    blocks.extend(synthetic[insert_b:])
    return blocks


def synthetic_memory_block(item_id: str, idx: int, sentences_per_block: int) -> ContextBlock:
    topic_key, base_text = MEMORY_TOPICS[idx % len(MEMORY_TOPICS)]
    timestamp_month = 1 + (idx % 3)
    timestamp_day = 1 + (idx % 27)
    text_parts = [
        f"Memory key: {topic_key}. {base_text}",
    ]
    for repeat_idx in range(1, sentences_per_block):
        text_parts.append(
            f"Related note {repeat_idx}: this memory is background context for unrelated workflow "
            f"{idx:03d} and should not affect other memory keys."
        )
    return ContextBlock(
        id=f"{item_id}_long_distractor_{idx:03d}",
        timestamp=f"2024-{timestamp_month:02d}-{timestamp_day:02d}T09:00:00+00:00",
        type="agent_memory",
        trust="medium",
        status="unknown",
        text=" ".join(text_parts),
        metadata={"memory_key": topic_key, "long_context_distractor": True},
    )


if __name__ == "__main__":
    main()
