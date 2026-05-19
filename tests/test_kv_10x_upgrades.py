from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.schema import BenchmarkItem, ContextBlock
from kv_value_ablation_probe import (
    detected_text_timestamp_current_ids,
    detected_text_timestamp_stale_ids,
    latest_block_ids_per_memory_key,
    prompt_policy_item,
)
from make_agent_memory_action_kv_dataset import all_items as action_memory_items
from make_agent_memory_long_context_kv_dataset import inflate_item
from make_agent_memory_messy_kv_dataset import all_items as messy_memory_items
from summarize_kv_persistent_runtime import compare_policy


def test_long_context_inflation_preserves_target_lineage_and_adds_distractors() -> None:
    item = BenchmarkItem(
        id="case",
        domain="agent_memory_action",
        question_type="agent_memory_action_value",
        context_blocks=[
            ContextBlock(
                id="current",
                timestamp="2024-03-01T00:00:00+00:00",
                type="agent_memory",
                text="Memory key: profile.editor.default. The current editor is Zed.",
                metadata={"memory_key": "profile.editor.default", "supersedes": ["stale"]},
            ),
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="agent_memory",
                text="Memory key: profile.editor.default. The old editor was Vim.",
                metadata={"memory_key": "profile.editor.default", "superseded_by": ["current"]},
            ),
        ],
        query="For memory key 'profile.editor.default', what current editor should be used?",
        gold_answer="Zed",
        gold_evidence_ids=["current"],
        stale_ids=["stale"],
        metadata={"variant": "current_first", "distractor_ratio": 0.5},
    )

    inflated = inflate_item(item, distractor_blocks=6, sentences_per_block=2, variant="long_context")

    assert inflated.domain == "agent_memory_long_context"
    assert inflated.gold_evidence_ids == ["current"]
    assert inflated.stale_ids == ["stale"]
    assert len(inflated.context_blocks) == 8
    assert len(inflated.distractor_ids) == 6
    assert inflated.metadata["source_item_id"] == "case"
    current = next(block for block in inflated.context_blocks if block.id == "current")
    stale = next(block for block in inflated.context_blocks if block.id == "stale")
    assert current.metadata["supersedes"] == ["stale"]
    assert stale.metadata["superseded_by"] == ["current"]


def test_persistent_runtime_comparison_charges_prompt_deletion_for_cleaned_prefill() -> None:
    paired = [
        {
            "drop_stale_prompt": {
                "correct": True,
                "evidence_correct": True,
                "used_stale_fact": False,
                "prefill_ms": 100.0,
                "cache_clone_ms": 2.0,
                "decode_ms": 20.0,
            },
            "cache_edit": {
                "correct": True,
                "evidence_correct": True,
                "used_stale_fact": False,
                "cache_clone_ms": 3.0,
                "cache_edit_ms": 1.0,
                "decode_ms": 20.0,
            },
        }
    ]

    row = compare_policy(paired, "drop_stale_prompt", "cache_edit")

    assert row["avg_prompt_deletion_update_ms"] == 122.0
    assert row["avg_cache_repair_update_ms"] == 24.0
    assert row["avg_prefill_saved_ms"] == 96.0
    assert row["speedup_ratio"] == 122.0 / 24.0


def test_memory_lineage_protocol_resolves_current_sources_without_stale_labels() -> None:
    items = action_memory_items()[:4] + messy_memory_items()[:4]
    for item in items:
        assert detected_text_timestamp_current_ids(item) == item.gold_evidence_ids
        assert set(detected_text_timestamp_stale_ids(item)) == set(item.stale_ids)
        latest_ids = latest_block_ids_per_memory_key(item)
        assert set(item.gold_evidence_ids).issubset(latest_ids)
        assert not (set(item.stale_ids) & latest_ids)

        summary_item = prompt_policy_item("summarize_text_timestamp_current_prompt", item)
        assert [block.id for block in summary_item.context_blocks] == item.gold_evidence_ids
        assert all(block.metadata.get("synthetic_summary") for block in summary_item.context_blocks)
