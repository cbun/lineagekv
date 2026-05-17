from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.schema import BenchmarkItem, ContextBlock
from context_rot.datasets.generate_synthetic import generate_items
from audit_git_doc_phrase_quality import audit_item
from analyze_kv_identifiability import item_structural_examples
from kv_value_ablation_probe import (
    EncodedCase,
    apply_policy,
    detected_conflict_stale_ids,
    detected_lineage_pairs,
    detected_lineage_stale_ids,
    detected_query_ref_stale_ids,
    detected_text_timestamp_current_ids,
    detected_text_timestamp_pairs,
    detected_text_timestamp_stale_ids,
    encode_case,
    move_encoded_case,
    parse_torch_dtype,
    parse_probe_output,
    prompt_policy_excluded_block_ids,
    prompt_policy_item,
    latest_block_ids_per_memory_key,
    remap_parsed_evidence_ids,
    resolve_device,
)
from make_agent_memory_update_kv_dataset import SCENARIOS, agent_memory_item
from make_agent_memory_action_kv_dataset import SCENARIOS as ACTION_SCENARIOS, all_items as action_memory_items
from make_agent_memory_messy_kv_dataset import SCENARIOS as MESSY_SCENARIOS, all_items as messy_memory_items
from make_adversarial_kv_dataset import ADVERSARIAL_VARIANTS, build_adversarial_item
from make_cue_stripped_kv_dataset import cue_stripped_item
from make_git_doc_evidence_choice_kv_dataset import evidence_choice_item
from make_git_doc_history_kv_dataset import build_doc_items
from make_git_doc_nonce_option_kv_dataset import nonce_codes, nonce_option_items
from make_git_doc_option_kv_dataset import LABELS, option_level_item, option_level_items
from make_git_doc_phrase_choice_kv_dataset import phrase_choice_items
from make_git_doc_phrase_kv_dataset import changed_phrases, phrase_level_item
from make_git_doc_yesno_kv_dataset import yesno_items
from make_git_value_nonce_option_kv_dataset import nonce_option_items as value_nonce_option_items
from make_git_history_kv_dataset import build_items, parse_spec
from make_lineage_kv_dataset import lineage_item
from make_balanced_jsonl_probe import round_robin


def test_timestamp_only_detector_fails_when_stale_duplicates_are_latest() -> None:
    item = next(
        candidate
        for candidate in generate_items(base_limit=1)
        if candidate.id == "user_preference_001_wrong_duplicate_10x_d90"
    )

    assert set(detected_conflict_stale_ids(item, use_current_text_cues=True)) == set(item.stale_ids)
    assert set(detected_conflict_stale_ids(item, use_current_text_cues=False)) == {"b001_current"}
    assert set(
        detected_conflict_stale_ids(item, use_current_text_cues=False, prefer_singleton_conflicts=True)
    ) == set(item.stale_ids)


def test_agent_memory_update_items_are_balanced_and_text_timestamp_targetable() -> None:
    items = [
        agent_memory_item(scenario, order=order, assignment=assignment)
        for scenario in SCENARIOS
        for order in ["current_first", "stale_first"]
        for assignment in ["code_a_current", "code_b_current"]
    ]

    assert len(items) == 80
    assert {item.domain for item in items} == {"agent_memory"}
    assert {item.metadata["variant"] for item in items} == {"current_first", "stale_first"}
    assert {item.metadata["nonce_assignment"] for item in items} == {"code_a_current", "code_b_current"}

    sample = next(item for item in items if item.metadata["slot"] == "default departure airport")
    assert sample.gold_answer in sample.metadata["option_labels"]
    assert sample.metadata["option_labels"][sample.gold_answer] == "AUS"
    assert sample.metadata["option_labels"][sample.metadata["stale_value"]] == "SFO"
    assert set(detected_lineage_stale_ids(sample.model_copy(update={"stale_ids": []}))) == set(sample.stale_ids)

    scrubbed = sample.model_copy(
        update={
            "context_blocks": [
                block.model_copy(update={"metadata": {}, "status": "unknown"})
                for block in sample.context_blocks
            ],
            "metadata": {},
            "stale_ids": [],
        }
    )
    assert set(detected_text_timestamp_stale_ids(scrubbed)) == set(sample.stale_ids)


def test_agent_memory_action_items_have_locked_splits_and_visible_key_targeting() -> None:
    items = action_memory_items()

    assert len(items) == 64
    assert {item.domain for item in items} == {"agent_memory_action"}
    assert {item.metadata["split"] for item in items} == {"dev", "test"}
    assert {item.metadata["variant"] for item in items} == {"current_first", "stale_first"}
    assert {item.metadata["query_style"] for item in items} == {"direct_value", "task_action"}

    dev_keys = {item.metadata["memory_key"] for item in items if item.metadata["split"] == "dev"}
    test_keys = {item.metadata["memory_key"] for item in items if item.metadata["split"] == "test"}
    assert len(dev_keys) == 8
    assert len(test_keys) == 8
    assert not (dev_keys & test_keys)
    assert {scenario["split"] for scenario in ACTION_SCENARIOS} == {"dev", "test"}

    sample = next(item for item in items if item.metadata["split"] == "test" and item.metadata["query_style"] == "task_action")
    assert len(sample.context_blocks) == 4
    assert len(sample.distractor_ids) == 2
    assert set(detected_lineage_stale_ids(sample.model_copy(update={"stale_ids": []}))) == set(sample.stale_ids)

    scrubbed_blocks = []
    for block in sample.context_blocks:
        metadata = dict(block.metadata)
        metadata.pop("supersedes", None)
        metadata.pop("superseded_by", None)
        metadata.pop("memory_key", None)
        scrubbed_blocks.append(block.model_copy(update={"status": "unknown", "metadata": metadata}))
    scrubbed = sample.model_copy(update={"stale_ids": [], "metadata": {}, "context_blocks": scrubbed_blocks})
    assert set(detected_text_timestamp_stale_ids(scrubbed)) == set(sample.stale_ids)

    no_text = scrubbed.model_copy(
        update={"context_blocks": [block.model_copy(update={"text": ""}) for block in scrubbed.context_blocks]}
    )
    assert detected_text_timestamp_stale_ids(no_text) == []

    tied_time = scrubbed.model_copy(
        update={
            "context_blocks": [
                block.model_copy(update={"timestamp": "2024-03-15T09:00:00+00:00"})
                for block in scrubbed.context_blocks
            ]
        }
    )
    assert detected_text_timestamp_stale_ids(tied_time) == []


def test_agent_memory_messy_items_have_multi_update_locked_splits_and_visible_key_targeting() -> None:
    items = messy_memory_items()

    assert len(items) == 40
    assert {item.domain for item in items} == {"agent_memory_messy"}
    assert {item.metadata["split"] for item in items} == {"dev", "test"}
    assert {item.metadata["variant"] for item in items} == {"chronological", "scrambled"}
    assert {item.metadata["query_style"] for item in items} == {"direct_value", "task_action"}

    dev_keys = {item.metadata["memory_key"] for item in items if item.metadata["split"] == "dev"}
    test_keys = {item.metadata["memory_key"] for item in items if item.metadata["split"] == "test"}
    assert len(dev_keys) == 5
    assert len(test_keys) == 5
    assert not (dev_keys & test_keys)
    assert {scenario["split"] for scenario in MESSY_SCENARIOS} == {"dev", "test"}

    sample = next(item for item in items if item.metadata["split"] == "test" and item.metadata["variant"] == "scrambled")
    assert len(sample.context_blocks) == 5
    assert len(sample.stale_ids) == 2
    assert len(sample.distractor_ids) == 2
    assert set(detected_lineage_stale_ids(sample.model_copy(update={"stale_ids": []}))) == set(sample.stale_ids)

    scrubbed_blocks = []
    for block in sample.context_blocks:
        metadata = dict(block.metadata)
        metadata.pop("supersedes", None)
        metadata.pop("superseded_by", None)
        metadata.pop("memory_key", None)
        scrubbed_blocks.append(block.model_copy(update={"status": "unknown", "metadata": metadata}))
    scrubbed = sample.model_copy(update={"stale_ids": [], "metadata": {}, "context_blocks": scrubbed_blocks})
    assert set(detected_text_timestamp_stale_ids(scrubbed)) == set(sample.stale_ids)
    assert detected_text_timestamp_current_ids(scrubbed) == sample.gold_evidence_ids

    tied_time = scrubbed.model_copy(
        update={
            "context_blocks": [
                block.model_copy(update={"timestamp": "2024-03-20T09:00:00+00:00"})
                for block in scrubbed.context_blocks
            ]
        }
    )
    assert detected_text_timestamp_stale_ids(tied_time) == []


def test_cue_stripped_dataset_removes_current_cues_but_keeps_parseable_claims() -> None:
    item = next(
        candidate
        for candidate in generate_items(base_limit=1)
        if candidate.id == "user_preference_001_current_late_d90"
    )

    stripped = cue_stripped_item(item)
    joined_text = "\n".join(block.text for block in stripped.context_blocks).lower()
    assert "current decision" not in joined_text
    assert "supersedes" not in joined_text
    assert "entity" not in stripped.context_blocks[0].metadata
    assert set(detected_conflict_stale_ids(stripped, use_current_text_cues=True)) == set(item.stale_ids)


def test_duplicate_aware_detector_recovers_cue_stripped_high_rot_cases() -> None:
    variants = {"wrong_duplicate_10x", "current_late", "current_early_stale_late", "stale_repeated_after"}
    items = [
        cue_stripped_item(candidate)
        for candidate in generate_items(base_limit=5, distractor_ratios=[0.9], variants=sorted(variants))
    ]

    assert len(items) == 20
    for item in items:
        predicted = detected_conflict_stale_ids(
            item,
            use_current_text_cues=False,
            prefer_singleton_conflicts=True,
        )
        assert set(predicted) == set(item.stale_ids)


def test_adversarial_conflicts_break_duplicate_aware_detector() -> None:
    base = cue_stripped_item(
        next(
            candidate
            for candidate in generate_items(base_limit=1, distractor_ratios=[0.9], variants=["current_late"])
            if candidate.id == "user_preference_001_current_late_d90"
        )
    )

    for variant in ADVERSARIAL_VARIANTS:
        item = build_adversarial_item(base, variant)
        predicted = set(
            detected_conflict_stale_ids(
                item,
                use_current_text_cues=False,
                prefer_singleton_conflicts=True,
            )
        )
        assert predicted != set(item.stale_ids)


def test_lineage_detector_recovers_adversarial_spans_without_status_or_labels() -> None:
    base = cue_stripped_item(
        next(
            candidate
            for candidate in generate_items(base_limit=1, distractor_ratios=[0.9], variants=["current_late"])
            if candidate.id == "user_preference_001_current_late_d90"
        )
    )
    item = build_adversarial_item(base, "both_values_repeated")
    lineage = lineage_item(item)

    assert [block.status for block in lineage.context_blocks] == [block.status for block in item.context_blocks]
    for block in lineage.context_blocks:
        if block.id in item.stale_ids:
            assert block.metadata["superseded_by"] == item.gold_evidence_ids
        if block.id in item.gold_evidence_ids:
            assert block.metadata["supersedes"] == item.stale_ids

    assert detected_lineage_stale_ids(item.model_copy(update={"stale_ids": []})) == []
    scrambled_labels = lineage.model_copy(update={"stale_ids": []})
    assert set(detected_lineage_stale_ids(scrambled_labels)) == set(item.stale_ids)


def test_generic_lineage_layer_head_policy_zeros_only_target_values() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 2, 6, 1))
            self.values = torch.ones((1, 2, 6, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "zero_lineage_conflict_values_layer_1_head_0")

    assert torch.all(cache.layers[1].values[:, 0, 1:3, :] == 0)
    assert torch.all(cache.layers[1].values[:, 1, 1:3, :] == 1)
    assert torch.all(cache.layers[0].values == 1)
    assert torch.all(cache.layers[1].values[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[1].keys == 1)


def test_generic_lineage_layer_range_policy_zeros_target_band() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 2, 6, 1))
            self.values = torch.ones((1, 2, 6, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_range_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "zero_lineage_conflict_values_layers_1_2")

    assert torch.all(cache.layers[0].values == 1)
    assert torch.all(cache.layers[1].values[:, :, 1:3, :] == 0)
    assert torch.all(cache.layers[2].values[:, :, 1:3, :] == 0)
    assert torch.all(cache.layers[1].values[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[2].values[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[1].keys == 1)
    assert torch.all(cache.layers[2].keys == 1)


def test_query_ref_layer_range_policy_zeros_target_band_without_lineage() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 2, 6, 1))
            self.values = torch.ones((1, 2, 6, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="query_ref_policy_test",
        domain="git_doc_history",
        question_type="doc_revision_nonce_option",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="git_doc_revision",
                text="Old value",
                metadata={"repo": "demo", "path": "README.md", "ref": "v1"},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="git_doc_revision",
                text="New value",
                metadata={"repo": "demo", "path": "README.md", "ref": "v2"},
            ),
        ],
        query="In demo README.md, which option code names the current phrase after revision v2?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "zero_query_ref_conflict_values_layers_1_2")

    assert torch.all(cache.layers[0].values == 1)
    assert torch.all(cache.layers[1].values[:, :, 1:3, :] == 0)
    assert torch.all(cache.layers[2].values[:, :, 1:3, :] == 0)
    assert torch.all(cache.layers[1].values[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[2].values[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[1].keys == 1)
    assert torch.all(cache.layers[2].keys == 1)


def test_text_timestamp_layer_range_policy_zeros_target_band_without_metadata() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 2, 6, 1))
            self.values = torch.ones((1, 2, 6, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="text_timestamp_policy_test",
        domain="git_doc_history",
        question_type="doc_revision_nonce_option",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="git_doc_revision",
                text="The client retries failed requests twice.",
                metadata={},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="git_doc_revision",
                text="The client retries failed requests four times.",
                metadata={},
            ),
        ],
        query=(
            "In demo README.md, which option code names the current phrase after revision v2? "
            "caaaaa = 'four times'. cbbbbb = 'twice'. Answer with only the option code."
        ),
        gold_answer="caaaaa",
        gold_evidence_ids=["current"],
        stale_ids=[],
        metadata={},
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "zero_text_timestamp_conflict_values_layers_1_2")

    assert torch.all(cache.layers[0].values == 1)
    assert torch.all(cache.layers[1].values[:, :, 1:3, :] == 0)
    assert torch.all(cache.layers[2].values[:, :, 1:3, :] == 0)
    assert torch.all(cache.layers[1].values[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[2].values[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[1].keys == 1)
    assert torch.all(cache.layers[2].keys == 1)


def test_text_timestamp_soft_scale_and_graft_policies_use_timestamp_pairs() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 1, 6, 1))
            self.values = torch.tensor([[[[0.0], [1.0], [2.0], [3.0], [5.0], [6.0]]]])

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="text_timestamp_soft_policy_test",
        domain="git_doc_history",
        question_type="doc_revision_nonce_option",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="git_doc_revision",
                text="The client retries failed requests twice.",
                metadata={},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="git_doc_revision",
                text="The client retries failed requests four times.",
                metadata={},
            ),
        ],
        query=(
            "In demo README.md, which option code names the current phrase after revision v2? "
            "caaaaa = 'four times'. cbbbbb = 'twice'. Answer with only the option code."
        ),
        gold_answer="caaaaa",
        gold_evidence_ids=["current"],
        stale_ids=[],
        metadata={},
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )

    assert detected_text_timestamp_pairs(item) == [("stale", "current")]

    scaled_cache = Cache()
    apply_policy(scaled_cache, encoded, item, "scale_text_timestamp_conflict_values_layers_1_2_factor_0_25")
    assert torch.all(scaled_cache.layers[1].values[:, :, 1:3, :] == torch.tensor([[[[0.25], [0.5]]]]))
    assert torch.all(scaled_cache.layers[2].values[:, :, 1:3, :] == torch.tensor([[[[0.25], [0.5]]]]))
    assert torch.all(scaled_cache.layers[1].values[:, :, 3:5, :] == Layer().values[:, :, 3:5, :])
    assert torch.all(scaled_cache.layers[1].keys == 1)

    graft_cache = Cache()
    apply_policy(graft_cache, encoded, item, "graft_text_timestamp_current_mean_values_layers_1_2")
    assert torch.all(graft_cache.layers[1].values[:, :, 1:3, :] == 4)
    assert torch.all(graft_cache.layers[2].values[:, :, 1:3, :] == 4)
    assert torch.all(graft_cache.layers[1].values[:, :, 3:5, :] == Layer().values[:, :, 3:5, :])
    assert torch.all(graft_cache.layers[1].keys == 1)


def test_text_timestamp_current_boost_policies_use_timestamp_pairs() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 1, 6, 1))
            self.values = torch.ones((1, 1, 6, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="text_timestamp_current_boost_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="agent_memory",
                text="Memory key: profile.editor.default. The default editor used to be Vim.",
                metadata={},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="agent_memory",
                text="Memory key: profile.editor.default. The default editor is now Zed.",
                metadata={},
            ),
        ],
        query="Use the current memory for key 'profile.editor.default'. What editor should the assistant use?",
        gold_answer="Zed",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )

    assert detected_text_timestamp_pairs(item) == [("stale", "current")]
    assert detected_text_timestamp_current_ids(item) == ["current"]

    boost_cache = Cache()
    apply_policy(boost_cache, encoded, item, "scale_text_timestamp_current_values_layers_1_2_factor_1_5")
    assert torch.all(boost_cache.layers[0].values == 1)
    assert torch.all(boost_cache.layers[1].values[:, :, 1:3, :] == 1)
    assert torch.all(boost_cache.layers[2].values[:, :, 1:3, :] == 1)
    assert torch.all(boost_cache.layers[1].values[:, :, 3:5, :] == 1.5)
    assert torch.all(boost_cache.layers[2].values[:, :, 3:5, :] == 1.5)
    assert torch.all(boost_cache.layers[1].keys == 1)

    zero_boost_cache = Cache()
    apply_policy(zero_boost_cache, encoded, item, "zero_text_timestamp_conflict_values_layers_1_2_boost_current_factor_1_5")
    assert torch.all(zero_boost_cache.layers[0].values == 1)
    assert torch.all(zero_boost_cache.layers[1].values[:, :, 1:3, :] == 0)
    assert torch.all(zero_boost_cache.layers[2].values[:, :, 1:3, :] == 0)
    assert torch.all(zero_boost_cache.layers[1].values[:, :, 3:5, :] == 1.5)
    assert torch.all(zero_boost_cache.layers[2].values[:, :, 3:5, :] == 1.5)
    assert torch.all(zero_boost_cache.layers[1].keys == 1)

    key_boost_cache = Cache()
    apply_policy(
        key_boost_cache,
        encoded,
        item,
        "zero_text_timestamp_conflict_keys_layers_1_2_boost_current_values_factor_1_5",
    )
    assert torch.all(key_boost_cache.layers[0].keys == 1)
    assert torch.all(key_boost_cache.layers[1].keys[:, :, 1:3, :] == 0)
    assert torch.all(key_boost_cache.layers[2].keys[:, :, 1:3, :] == 0)
    assert torch.all(key_boost_cache.layers[1].values[:, :, 1:3, :] == 1)
    assert torch.all(key_boost_cache.layers[1].values[:, :, 3:5, :] == 1.5)
    assert torch.all(key_boost_cache.layers[2].values[:, :, 3:5, :] == 1.5)

    kv_boost_cache = Cache()
    apply_policy(
        kv_boost_cache,
        encoded,
        item,
        "zero_text_timestamp_conflict_keys_values_layers_1_2_boost_current_values_factor_1_5",
    )
    assert torch.all(kv_boost_cache.layers[0].keys == 1)
    assert torch.all(kv_boost_cache.layers[1].keys[:, :, 1:3, :] == 0)
    assert torch.all(kv_boost_cache.layers[2].keys[:, :, 1:3, :] == 0)
    assert torch.all(kv_boost_cache.layers[1].values[:, :, 1:3, :] == 0)
    assert torch.all(kv_boost_cache.layers[2].values[:, :, 1:3, :] == 0)
    assert torch.all(kv_boost_cache.layers[1].values[:, :, 3:5, :] == 1.5)
    assert torch.all(kv_boost_cache.layers[2].values[:, :, 3:5, :] == 1.5)


def test_prompt_current_only_policies_exclude_non_current_blocks() -> None:
    item = BenchmarkItem(
        id="prompt_policy_case",
        domain="agent_memory_action",
        question_type="agent_memory_action_value",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="agent_memory",
                text="Memory key: profile.editor.default. The old editor was Vim.",
                metadata={"memory_key": "profile.editor.default"},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="agent_memory",
                text="Memory key: profile.editor.default. The current editor is Zed.",
                metadata={"memory_key": "profile.editor.default", "value": "Zed"},
            ),
            ContextBlock(
                id="distractor",
                timestamp="2024-02-02T00:00:00+00:00",
                type="agent_memory",
                text="Memory key: profile.timezone. Use Central Time.",
                metadata={"memory_key": "profile.timezone"},
            ),
        ],
        query="For memory key 'profile.editor.default', what current editor should be used?",
        gold_answer="Zed",
        gold_evidence_ids=["current"],
        stale_ids=["stale"],
    )

    assert prompt_policy_excluded_block_ids("drop_stale_prompt", item) == {"stale"}
    assert prompt_policy_excluded_block_ids("keep_gold_prompt", item) == {"stale", "distractor"}
    assert prompt_policy_excluded_block_ids("keep_query_key_prompt", item) == {"distractor"}
    assert prompt_policy_excluded_block_ids("keep_latest_per_memory_key_prompt", item) == {"stale"}
    assert prompt_policy_excluded_block_ids("keep_text_timestamp_current_prompt", item) == {"stale", "distractor"}
    assert prompt_policy_excluded_block_ids("summarize_text_timestamp_current_prompt", item) == set()
    assert prompt_policy_excluded_block_ids("full_cache", item) is None
    assert latest_block_ids_per_memory_key(item) == {"current", "distractor"}

    summary_item = prompt_policy_item("summarize_text_timestamp_current_prompt", item)
    assert [block.id for block in summary_item.context_blocks] == ["current"]
    assert summary_item.context_blocks[0].text == "Memory key: profile.editor.default. Current value: Zed."
    assert summary_item.context_blocks[0].metadata["source_block_id"] == "current"


def test_generic_lineage_key_only_range_policy_preserves_values() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 2, 6, 1))
            self.values = torch.ones((1, 2, 6, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_key_only_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "zero_lineage_conflict_keys_layers_1_2")

    assert torch.all(cache.layers[0].keys == 1)
    assert torch.all(cache.layers[1].keys[:, :, 1:3, :] == 0)
    assert torch.all(cache.layers[2].keys[:, :, 1:3, :] == 0)
    assert torch.all(cache.layers[1].keys[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[2].keys[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[1].values == 1)
    assert torch.all(cache.layers[2].values == 1)


def test_probe_runtime_dtype_device_helpers_keep_metadata() -> None:
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 3), dtype=torch.long),
        suffix_ids=torch.ones((1, 2), dtype=torch.long),
        block_ranges={"stale": (0, 1)},
        suffix_ranges={"query": (0, 1)},
        block_header_ranges={"stale": (0, 1)},
        block_body_ranges={"stale": (1, 2)},
        block_aliases={"stale": "m000"},
        alias_to_block_id={"m000": "stale"},
    )

    assert parse_torch_dtype("float16") is torch.float16
    assert resolve_device("cpu").type == "cpu"
    moved = move_encoded_case(encoded, torch.device("cpu"))

    assert moved is encoded
    assert moved.block_ranges == {"stale": (0, 1)}
    assert moved.alias_to_block_id == {"m000": "stale"}


def test_generic_lineage_layer_range_scale_policy_attenuates_target_band() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 2, 6, 1))
            self.values = torch.ones((1, 2, 6, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_scale_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "scale_lineage_conflict_values_layers_1_2_factor_0_25")

    assert torch.all(cache.layers[0].values == 1)
    assert torch.all(cache.layers[1].values[:, :, 1:3, :] == 0.25)
    assert torch.all(cache.layers[2].values[:, :, 1:3, :] == 0.25)
    assert torch.all(cache.layers[1].values[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[2].values[:, :, 3:5, :] == 1)
    assert torch.all(cache.layers[1].keys == 1)
    assert torch.all(cache.layers[2].keys == 1)


def test_generic_lineage_layer_range_span_policy_zeros_only_requested_subspan() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 1, 8, 1))
            self.values = torch.ones((1, 1, 8, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_span_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 8), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 5), "current": (5, 7)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "zero_lineage_conflict_values_layers_1_2_span_second_half")

    assert torch.all(cache.layers[0].values == 1)
    assert torch.all(cache.layers[1].values[:, :, 1:3, :] == 1)
    assert torch.all(cache.layers[1].values[:, :, 3:5, :] == 0)
    assert torch.all(cache.layers[2].values[:, :, 1:3, :] == 1)
    assert torch.all(cache.layers[2].values[:, :, 3:5, :] == 0)
    assert torch.all(cache.layers[1].values[:, :, 5:7, :] == 1)
    assert torch.all(cache.layers[1].keys == 1)


def test_generic_lineage_layer_range_decile_policy_zeros_only_requested_decile() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 1, 14, 1))
            self.values = torch.ones((1, 1, 14, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_decile_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 14), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (2, 12), "current": (12, 14)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "zero_lineage_conflict_values_layers_1_2_span_decile_3")

    assert torch.all(cache.layers[0].values == 1)
    assert torch.all(cache.layers[1].values[:, :, 2:5, :] == 1)
    assert torch.all(cache.layers[1].values[:, :, 5:6, :] == 0)
    assert torch.all(cache.layers[1].values[:, :, 6:12, :] == 1)
    assert torch.all(cache.layers[2].values[:, :, 5:6, :] == 0)
    assert torch.all(cache.layers[1].values[:, :, 12:14, :] == 1)
    assert torch.all(cache.layers[1].keys == 1)


def test_generic_lineage_layer_range_header_body_policies_use_subranges() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 1, 10, 1))
            self.values = torch.ones((1, 1, 10, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_header_body_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 10), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 7), "current": (7, 10)},
        block_header_ranges={"stale": (1, 4), "current": (7, 8)},
        block_body_ranges={"stale": (4, 7), "current": (8, 10)},
    )

    header_cache = Cache()
    body_cache = Cache()
    apply_policy(header_cache, encoded, item, "zero_lineage_conflict_values_layers_1_1_span_header")
    apply_policy(body_cache, encoded, item, "zero_lineage_conflict_values_layers_1_1_span_body")

    assert torch.all(header_cache.layers[1].values[:, :, 1:4, :] == 0)
    assert torch.all(header_cache.layers[1].values[:, :, 4:7, :] == 1)
    assert torch.all(body_cache.layers[1].values[:, :, 1:4, :] == 1)
    assert torch.all(body_cache.layers[1].values[:, :, 4:7, :] == 0)
    assert torch.all(header_cache.layers[1].keys == 1)
    assert torch.all(body_cache.layers[1].keys == 1)


def test_lineage_attention_top_fraction_policy_zeros_only_high_attention_tokens() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 1, 8, 1))
            self.values = torch.ones((1, 1, 8, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_attention_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 8), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 5), "current": (5, 7)},
    )
    cache = Cache()
    attention_scores = torch.tensor([0.0, 0.1, 0.9, 0.2, 0.8, 0.7, 0.6, 0.0])

    apply_policy(
        cache,
        encoded,
        item,
        "zero_lineage_conflict_values_layers_1_2_attn_topfrac_0_50",
        attention_token_scores=attention_scores,
    )

    assert torch.all(cache.layers[0].values == 1)
    assert torch.all(cache.layers[1].values[:, :, [1, 3], :] == 1)
    assert torch.all(cache.layers[1].values[:, :, [2, 4], :] == 0)
    assert torch.all(cache.layers[2].values[:, :, [2, 4], :] == 0)
    assert torch.all(cache.layers[1].values[:, :, 5:7, :] == 1)
    assert torch.all(cache.layers[1].keys == 1)


def test_lineage_current_scale_policy_boosts_current_values_only() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 2, 6, 1))
            self.values = torch.ones((1, 2, 6, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_current_scale_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "scale_lineage_current_values_layers_1_2_factor_1_5")

    assert torch.all(cache.layers[0].values == 1)
    assert torch.all(cache.layers[1].values[:, :, 1:3, :] == 1)
    assert torch.all(cache.layers[2].values[:, :, 1:3, :] == 1)
    assert torch.all(cache.layers[1].values[:, :, 3:5, :] == 1.5)
    assert torch.all(cache.layers[2].values[:, :, 3:5, :] == 1.5)
    assert torch.all(cache.layers[1].keys == 1)
    assert torch.all(cache.layers[2].keys == 1)


def test_lineage_zero_boost_policy_suppresses_stale_and_boosts_current() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 2, 6, 1))
            self.values = torch.ones((1, 2, 6, 1))

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_zero_boost_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "zero_lineage_conflict_values_layers_1_2_boost_current_factor_1_5")

    assert torch.all(cache.layers[0].values == 1)
    assert torch.all(cache.layers[1].values[:, :, 1:3, :] == 0)
    assert torch.all(cache.layers[2].values[:, :, 1:3, :] == 0)
    assert torch.all(cache.layers[1].values[:, :, 3:5, :] == 1.5)
    assert torch.all(cache.layers[2].values[:, :, 3:5, :] == 1.5)
    assert torch.all(cache.layers[1].keys == 1)
    assert torch.all(cache.layers[2].keys == 1)


def test_lineage_current_mean_graft_replaces_stale_values_only() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 2, 6, 1))
            self.values = torch.tensor(
                [[[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]]]]
            )

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_graft_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )
    cache = Cache()

    assert detected_lineage_pairs(item.model_copy(update={"stale_ids": []})) == [("stale", "current")]
    apply_policy(cache, encoded, item, "graft_lineage_current_mean_values_layers_1_2")

    assert torch.all(cache.layers[0].values == Layer().values)
    assert torch.all(cache.layers[1].values[:, 0, 1:3, :] == 3.5)
    assert torch.all(cache.layers[1].values[:, 1, 1:3, :] == 13.5)
    assert torch.all(cache.layers[2].values[:, 0, 1:3, :] == 3.5)
    assert torch.all(cache.layers[2].values[:, 1, 1:3, :] == 13.5)
    assert torch.all(cache.layers[1].values[:, :, 3:5, :] == Layer().values[:, :, 3:5, :])
    assert torch.all(cache.layers[2].values[:, :, 3:5, :] == Layer().values[:, :, 3:5, :])
    assert torch.all(cache.layers[1].keys == 1)
    assert torch.all(cache.layers[2].keys == 1)


def test_lineage_current_resample_graft_preserves_current_token_variation() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 1, 8, 1))
            self.values = torch.tensor([[[[0.0], [1.0], [2.0], [3.0], [10.0], [20.0], [30.0], [40.0]]]])

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_resample_graft_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 8), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 4), "current": (4, 8)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "graft_lineage_current_resample_values_layers_1_2")

    assert torch.all(cache.layers[0].values == Layer().values)
    assert torch.all(cache.layers[1].values[:, :, 1:4, :] == torch.tensor([[[[10.0], [20.0], [30.0]]]]))
    assert torch.all(cache.layers[2].values[:, :, 1:4, :] == torch.tensor([[[[10.0], [20.0], [30.0]]]]))
    assert torch.all(cache.layers[1].values[:, :, 4:8, :] == Layer().values[:, :, 4:8, :])
    assert torch.all(cache.layers[1].keys == 1)


def test_lineage_current_mean_normmatch_graft_keeps_stale_token_norms() -> None:
    class Layer:
        def __init__(self) -> None:
            self.keys = torch.ones((1, 1, 6, 2))
            self.values = torch.tensor(
                [[[[0.0, 0.0], [3.0, 4.0], [0.0, 2.0], [6.0, 8.0], [6.0, 8.0], [0.0, 0.0]]]]
            )

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer(), Layer()]

    item = BenchmarkItem(
        id="lineage_normmatch_graft_policy_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )
    encoded = EncodedCase(
        prefix_ids=torch.ones((1, 6), dtype=torch.long),
        suffix_ids=torch.ones((1, 1), dtype=torch.long),
        block_ranges={"stale": (1, 3), "current": (3, 5)},
    )
    cache = Cache()

    apply_policy(cache, encoded, item, "graft_lineage_current_mean_normmatch_values_layers_1_1")

    expected = torch.tensor([[[[3.0, 4.0], [1.2, 1.6]]]])
    assert torch.all(cache.layers[0].values == Layer().values)
    assert torch.allclose(cache.layers[1].values[:, :, 1:3, :], expected)
    assert torch.allclose(cache.layers[1].values[:, :, 1:3, :].norm(dim=-1), torch.tensor([[[5.0, 2.0]]]))
    assert torch.all(cache.layers[1].values[:, :, 3:5, :] == Layer().values[:, :, 3:5, :])
    assert torch.all(cache.layers[1].keys == 1)


def test_chat_prompt_style_preserves_block_ranges_inside_template() -> None:
    class CharTokenizer:
        chat_template = "fake"

        def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
            return SimpleNamespace(input_ids=[ord(char) for char in text])

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool,
            tokenize: bool = False,
        ) -> str:
            assert not tokenize
            rendered = "".join(f"<{message['role']}>\n{message['content']}</{message['role']}>\n" for message in messages)
            if add_generation_prompt:
                rendered += "<assistant>\n"
            return rendered

    item = BenchmarkItem(
        id="chat_prompt_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["current"]},
            ),
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["stale"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=["stale"],
    )

    encoded = encode_case(CharTokenizer(), item, render_mode="temporal_blind", prompt_style="chat")
    prefix_text = "".join(chr(value) for value in encoded.prefix_ids[0].tolist())
    suffix_text = "".join(chr(value) for value in encoded.suffix_ids[0].tolist())
    stale_start, stale_end = encoded.block_ranges["stale"]
    current_start, current_end = encoded.block_ranges["current"]
    stale_header_start, stale_header_end = encoded.block_header_ranges["stale"]
    stale_body_start, stale_body_end = encoded.block_body_ranges["stale"]

    assert prefix_text.startswith("<user>\n")
    assert "Old value" in prefix_text[stale_start:stale_end]
    assert "New value" in prefix_text[current_start:current_end]
    assert "[block_id=stale;" in prefix_text[stale_header_start:stale_header_end]
    assert prefix_text[stale_body_start:stale_body_end] == "Old value"
    assert suffix_text.endswith("</user>\n<assistant>\n")
    assert "Question: Which value is current?" in suffix_text
    assert encoded.suffix_ranges is not None
    query_start, query_end = encoded.suffix_ranges["query"]
    assert suffix_text[query_start:query_end] == "Which value is current?"


def test_gemma4_prompt_style_uses_text_turn_template() -> None:
    class CharTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
            return SimpleNamespace(input_ids=[ord(char) for char in text])

    item = BenchmarkItem(
        id="gemma4_prompt_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["current"],
        stale_ids=[],
    )

    encoded = encode_case(CharTokenizer(), item, render_mode="temporal_blind", prompt_style="gemma4")
    prefix_text = "".join(chr(value) for value in encoded.prefix_ids[0].tolist())
    suffix_text = "".join(chr(value) for value in encoded.suffix_ids[0].tolist())

    assert prefix_text.startswith("<bos><|turn>system\n")
    assert "<|turn>user\n" in prefix_text
    assert "New value" in prefix_text
    assert suffix_text.endswith("<turn|>\n<|turn>model\n")


def test_neutral_block_id_mode_hides_original_ids_and_remaps_evidence() -> None:
    class CharTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
            return SimpleNamespace(input_ids=[ord(char) for char in text])

    item = BenchmarkItem(
        id="neutral_block_id_test",
        domain="synthetic",
        question_type="unit",
        context_blocks=[
            ContextBlock(
                id="d001_current",
                timestamp="2024-02-01T00:00:00+00:00",
                type="fact",
                text="New value",
                metadata={"supersedes": ["d001_stale"]},
            ),
            ContextBlock(
                id="d001_stale",
                timestamp="2024-01-01T00:00:00+00:00",
                type="fact",
                text="Old value",
                metadata={"superseded_by": ["d001_current"]},
            ),
        ],
        query="Which value is current?",
        gold_answer="new",
        gold_evidence_ids=["d001_current"],
        stale_ids=["d001_stale"],
    )

    encoded = encode_case(CharTokenizer(), item, render_mode="temporal_blind", block_id_mode="neutral")
    prefix_text = "".join(chr(value) for value in encoded.prefix_ids[0].tolist())
    parsed = remap_parsed_evidence_ids(
        {"evidence_ids": ["m000", "1", "m1", "unknown"]},
        encoded.alias_to_block_id or {},
    )

    assert "d001_current" not in prefix_text
    assert "d001_stale" not in prefix_text
    assert "[block_id=m000;" in prefix_text
    assert "[block_id=m001;" in prefix_text
    assert encoded.block_aliases == {"d001_current": "m000", "d001_stale": "m001"}
    assert "d001_stale" in encoded.block_ranges
    assert parsed["evidence_ids"] == ["d001_current", "d001_stale", "d001_stale", "unknown"]

    deleted = encode_case(
        CharTokenizer(),
        item,
        exclude_block_ids={"d001_stale"},
        render_mode="temporal_blind",
        block_id_mode="neutral",
    )
    deleted_parsed = remap_parsed_evidence_ids({"evidence_ids": ["1", "m001"]}, deleted.alias_to_block_id or {})
    assert deleted.block_aliases == {"d001_current": "m000"}
    assert deleted.alias_to_block_id == {"m000": "d001_current"}
    assert deleted_parsed["evidence_ids"] == ["1", "m001"]


def test_json_or_answer_parse_mode_extracts_unique_nonce_code() -> None:
    item = BenchmarkItem(
        id="nonce_parse_test",
        domain="synthetic",
        question_type="doc_revision_nonce_option",
        context_blocks=[],
        query="Which option code is current?",
        gold_answer="cwmoly",
        gold_evidence_ids=["current"],
        stale_ids=["stale"],
        metadata={
            "target_value": "cwmoly",
            "stale_value": "cu574s",
            "option_labels": {"cwmoly": "current phrase", "cu574s": "stale phrase"},
        },
    )

    parsed = parse_probe_output("Option code: cwmoly\n\nReturn JSON only...", item, parse_mode="json_or_answer")
    echoed_prompt = parse_probe_output(
        "cwmoly = current phrase. cu574s = stale phrase. Answer with only the option code.",
        item,
        parse_mode="json_or_answer",
    )

    assert parsed["answer"] == "cwmoly"
    assert parsed["evidence_ids"] == []
    assert echoed_prompt == {}


def test_json_or_answer_repairs_missing_answer_when_reasoning_has_unique_value() -> None:
    item = BenchmarkItem(
        id="malformed_answer_test",
        domain="agent_memory_long_context",
        question_type="agent_memory_action_value",
        context_blocks=[],
        query="Which analytics dashboard is current?",
        gold_answer="Metabase",
        gold_evidence_ids=["current"],
        stale_ids=["stale"],
        metadata={"target_value": "Metabase", "stale_value": "Looker"},
    )

    parsed = parse_probe_output(
        '{"evidence_ids":["current"],"reasoning_summary":"Metabase is current."}',
        item,
        parse_mode="json_or_answer",
    )

    assert parsed["answer"] == "Metabase"
    assert parsed["evidence_ids"] == ["current"]


def test_structural_signature_can_have_opposite_labels() -> None:
    base = cue_stripped_item(
        next(
            candidate
            for candidate in generate_items(base_limit=1, distractor_ratios=[0.9], variants=["current_late"])
            if candidate.id == "user_preference_001_current_late_d90"
        )
    )
    current_late_signature = item_structural_examples(base, "test")[0]
    singleton_stale_late_signature = item_structural_examples(
        build_adversarial_item(base, "singleton_stale_late"),
        "test",
    )[0]

    assert current_late_signature["signature"] == singleton_stale_late_signature["signature"]
    assert current_late_signature["label_pattern"] != singleton_stale_late_signature["label_pattern"]


def test_git_history_dataset_derives_lineage_from_real_commits(tmp_path: Path) -> None:
    repo = tmp_path / "demo_repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")

    commit_version(repo, "1.0.0", "2024-01-01T00:00:00+00:00")
    commit_version(repo, "1.1.0", "2024-02-01T00:00:00+00:00")
    commit_version(repo, "1.2.0", "2024-03-01T00:00:00+00:00")

    items = build_items(
        repo=repo,
        repo_name="demo_repo",
        specs=[parse_spec("package.json::version")],
        variants=["current_first"],
        max_items=10,
        max_events_per_spec=10,
    )

    assert [item.gold_answer for item in items] == ["1.1.0", "1.2.0"]
    first = items[0]
    assert first.context_blocks[0].metadata["supersedes"] == ["g001_stale"]
    assert first.context_blocks[1].metadata["superseded_by"] == ["g001_current"]
    assert {block.status for block in first.context_blocks} == {"unknown"}
    assert set(detected_lineage_stale_ids(first.model_copy(update={"stale_ids": []}))) == {"g001_stale"}


def test_git_value_nonce_option_items_use_balanced_codes(tmp_path: Path) -> None:
    repo = tmp_path / "value_nonce_repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")

    commit_version(repo, "1.0.0", "2024-01-01T00:00:00+00:00")
    commit_version(repo, "1.1.0", "2024-02-01T00:00:00+00:00")

    source_item = build_items(
        repo=repo,
        repo_name="value_nonce_repo",
        specs=[parse_spec("package.json::version")],
        variants=["current_first"],
        max_items=10,
        max_events_per_spec=10,
    )[0]

    code_a_item, code_b_item = value_nonce_option_items(source_item)
    code_a = code_a_item.metadata["stale_value"] if code_a_item.metadata["nonce_assignment"] == "code_b_current" else code_a_item.gold_answer
    code_b = code_b_item.metadata["stale_value"] if code_b_item.metadata["nonce_assignment"] == "code_a_current" else code_b_item.gold_answer

    assert code_a_item.question_type == "git_value_nonce_option"
    assert code_b_item.question_type == "git_value_nonce_option"
    assert code_a_item.gold_answer != code_b_item.gold_answer
    assert code_a_item.metadata["option_labels"][code_a_item.gold_answer] == "1.1.0"
    assert code_b_item.metadata["option_labels"][code_b_item.gold_answer] == "1.1.0"
    assert code_a_item.metadata["option_labels"][code_a_item.metadata["stale_value"]] == "1.0.0"
    assert code_b_item.metadata["option_labels"][code_b_item.metadata["stale_value"]] == "1.0.0"
    assert code_a in code_a_item.metadata["option_labels"]
    assert code_b in code_b_item.metadata["option_labels"]
    assert "Answer with only the option code" in code_a_item.query
    assert code_a_item.context_blocks == source_item.context_blocks
    assert code_b_item.context_blocks == source_item.context_blocks
    assert set(detected_lineage_stale_ids(code_b_item.model_copy(update={"stale_ids": []}))) == {"g001_stale"}


def test_git_doc_history_dataset_pairs_real_sentence_edits(tmp_path: Path) -> None:
    repo = tmp_path / "doc_repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")

    commit_readme(
        repo,
        "The client retries failed requests twice before surfacing an error to callers.",
        "2024-01-01T00:00:00+00:00",
        "v1",
    )
    commit_readme(
        repo,
        "The client retries failed requests four times before surfacing an error to callers.",
        "2024-02-01T00:00:00+00:00",
        "v2",
    )

    items = build_doc_items(
        repo=repo,
        repo_name="doc_repo",
        paths=["README.md"],
        variants=["stale_first"],
        max_items=10,
    )

    assert len(items) == 1
    item = items[0]
    assert item.gold_answer == "The client retries failed requests four times before surfacing an error to callers."
    assert item.metadata["stale_value"] == "The client retries failed requests twice before surfacing an error to callers."
    assert item.context_blocks[0].metadata["superseded_by"] == ["d001_current"]
    assert item.context_blocks[1].metadata["supersedes"] == ["d001_stale"]
    assert "value" not in item.context_blocks[1].metadata
    assert set(detected_lineage_stale_ids(item.model_copy(update={"stale_ids": []}))) == {"d001_stale"}


def test_git_doc_phrase_item_extracts_short_answer_target(tmp_path: Path) -> None:
    repo = tmp_path / "phrase_repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")

    commit_readme(
        repo,
        "The client retries failed requests twice before surfacing an error to callers.",
        "2024-01-01T00:00:00+00:00",
        "v1",
    )
    commit_readme(
        repo,
        "The client retries failed requests four times before surfacing an error to callers.",
        "2024-02-01T00:00:00+00:00",
        "v2",
    )

    sentence_item = build_doc_items(
        repo=repo,
        repo_name="phrase_repo",
        paths=["README.md"],
        variants=["stale_first"],
        max_items=10,
    )[0]
    phrase_item = phrase_level_item(sentence_item)

    assert changed_phrases(
        "The client retries failed requests twice before surfacing an error to callers.",
        "The client retries failed requests four times before surfacing an error to callers.",
    ) == ("twice", "four times")
    assert phrase_item is not None
    assert phrase_item.gold_answer == "four times"
    assert phrase_item.metadata["target_value"] == "four times"
    assert phrase_item.metadata["stale_value"] == "twice"
    assert phrase_item.context_blocks == sentence_item.context_blocks
    assert set(detected_lineage_stale_ids(phrase_item.model_copy(update={"stale_ids": []}))) == {"d001_stale"}
    assert audit_item(phrase_item).passed


def test_git_doc_phrase_quality_rejects_unrelated_hunk_neighbors() -> None:
    items = build_doc_items_from_sentences(
        old_sentence="Added support for res.contentType() literal",
        new_sentence="Fixed SlowBuffer support for res.send().",
    )
    phrase_item = phrase_level_item(items[0])

    assert phrase_item is not None
    audit = audit_item(phrase_item)
    assert not audit.passed
    assert "low_sentence_overlap" in audit.issues


def test_git_doc_phrase_quality_rejects_generic_phrase_pair() -> None:
    items = build_doc_items_from_sentences(
        old_sentence=(
            "This is the available config options for making requests. Only the url is required."
        ),
        new_sentence=(
            "These are the available config options for making requests. Only the url is required."
        ),
    )
    phrase_item = phrase_level_item(items[0])

    assert phrase_item is not None
    audit = audit_item(phrase_item)
    assert not audit.passed
    assert "generic_target_phrase" in audit.issues


def test_git_doc_option_item_uses_labels_without_changing_context(tmp_path: Path) -> None:
    repo = tmp_path / "option_repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")

    commit_readme(
        repo,
        "The client retries failed requests twice before surfacing an error to callers.",
        "2024-01-01T00:00:00+00:00",
        "v1",
    )
    commit_readme(
        repo,
        "The client retries failed requests four times before surfacing an error to callers.",
        "2024-02-01T00:00:00+00:00",
        "v2",
    )

    sentence_item = build_doc_items(
        repo=repo,
        repo_name="option_repo",
        paths=["README.md"],
        variants=["stale_first"],
        max_items=10,
    )[0]
    phrase_item = phrase_level_item(sentence_item)
    assert phrase_item is not None

    option_item = option_level_item(phrase_item)

    assert option_item.question_type == "doc_revision_option"
    assert option_item.gold_answer in LABELS
    assert option_item.metadata["target_value"] == option_item.gold_answer
    assert option_item.metadata["stale_value"] in LABELS
    assert option_item.metadata["stale_value"] != option_item.metadata["target_value"]
    assert option_item.metadata["option_labels"][option_item.gold_answer] == "four times"
    assert option_item.metadata["option_labels"][option_item.metadata["stale_value"]] == "twice"
    assert "Answer with only the option label" in option_item.query
    assert option_item.context_blocks == phrase_item.context_blocks
    assert set(detected_lineage_stale_ids(option_item.model_copy(update={"stale_ids": []}))) == {"d001_stale"}

    both = option_level_items(phrase_item, label_mode="both")
    assert len(both) == 2
    assert {item.gold_answer for item in both} == set(LABELS)
    assert {item.metadata["stale_value"] for item in both} == set(LABELS)
    assert all(item.context_blocks == phrase_item.context_blocks for item in both)
    assert all(item.id.endswith(("alpha_current", "bravo_current")) for item in both)


def test_git_doc_evidence_choice_item_targets_current_block_id(tmp_path: Path) -> None:
    repo = tmp_path / "evidence_choice_repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")

    commit_readme(
        repo,
        "The client retries failed requests twice before surfacing an error to callers.",
        "2024-01-01T00:00:00+00:00",
        "v1",
    )
    commit_readme(
        repo,
        "The client retries failed requests four times before surfacing an error to callers.",
        "2024-02-01T00:00:00+00:00",
        "v2",
    )

    sentence_item = build_doc_items(
        repo=repo,
        repo_name="evidence_choice_repo",
        paths=["README.md"],
        variants=["stale_first"],
        max_items=10,
    )[0]
    phrase_item = phrase_level_item(sentence_item)
    assert phrase_item is not None

    choice_item = evidence_choice_item(phrase_item)

    assert choice_item.question_type == "doc_revision_evidence_choice"
    assert choice_item.gold_answer == "d001_current"
    assert choice_item.metadata["target_value"] == "d001_current"
    assert choice_item.metadata["stale_value"] == "d001_stale"
    assert "Answer with only the block_id" in choice_item.query
    assert choice_item.context_blocks == phrase_item.context_blocks
    assert set(detected_lineage_stale_ids(choice_item.model_copy(update={"stale_ids": []}))) == {"d001_stale"}


def test_git_doc_yesno_items_balance_current_and_stale_candidates(tmp_path: Path) -> None:
    repo = tmp_path / "yesno_repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")

    commit_readme(
        repo,
        "The client retries failed requests twice before surfacing an error to callers.",
        "2024-01-01T00:00:00+00:00",
        "v1",
    )
    commit_readme(
        repo,
        "The client retries failed requests four times before surfacing an error to callers.",
        "2024-02-01T00:00:00+00:00",
        "v2",
    )

    sentence_item = build_doc_items(
        repo=repo,
        repo_name="yesno_repo",
        paths=["README.md"],
        variants=["stale_first"],
        max_items=10,
    )[0]
    phrase_item = phrase_level_item(sentence_item)
    assert phrase_item is not None

    yes_item, no_item = yesno_items(phrase_item)

    assert yes_item.question_type == "doc_revision_yesno"
    assert no_item.question_type == "doc_revision_yesno"
    assert yes_item.gold_answer == "YES"
    assert no_item.gold_answer == "NO"
    assert yes_item.metadata["candidate_phrase"] == "four times"
    assert no_item.metadata["candidate_phrase"] == "twice"
    assert yes_item.metadata["stale_value"] == "NO"
    assert no_item.metadata["stale_value"] == "YES"
    assert "Answer with only YES or NO" in yes_item.query
    assert yes_item.context_blocks == phrase_item.context_blocks
    assert no_item.context_blocks == phrase_item.context_blocks
    assert set(detected_lineage_stale_ids(no_item.model_copy(update={"stale_ids": []}))) == {"d001_stale"}


def test_git_doc_phrase_choice_items_balance_candidate_order(tmp_path: Path) -> None:
    repo = tmp_path / "phrase_choice_repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")

    commit_readme(
        repo,
        "The client retries failed requests twice before surfacing an error to callers.",
        "2024-01-01T00:00:00+00:00",
        "v1",
    )
    commit_readme(
        repo,
        "The client retries failed requests four times before surfacing an error to callers.",
        "2024-02-01T00:00:00+00:00",
        "v2",
    )

    sentence_item = build_doc_items(
        repo=repo,
        repo_name="phrase_choice_repo",
        paths=["README.md"],
        variants=["stale_first"],
        max_items=10,
    )[0]
    phrase_item = phrase_level_item(sentence_item)
    assert phrase_item is not None

    current_first, stale_first = phrase_choice_items(phrase_item)

    assert current_first.question_type == "doc_revision_phrase_choice"
    assert stale_first.question_type == "doc_revision_phrase_choice"
    assert current_first.gold_answer == "four times"
    assert stale_first.gold_answer == "four times"
    assert current_first.metadata["target_value"] == "four times"
    assert stale_first.metadata["target_value"] == "four times"
    assert current_first.metadata["stale_value"] == "twice"
    assert stale_first.metadata["stale_value"] == "twice"
    assert current_first.metadata["candidate_phrases"] == ["four times", "twice"]
    assert stale_first.metadata["candidate_phrases"] == ["twice", "four times"]
    assert "Answer with only the current phrase" in current_first.query
    assert current_first.context_blocks == phrase_item.context_blocks
    assert stale_first.context_blocks == phrase_item.context_blocks
    assert set(detected_lineage_stale_ids(stale_first.model_copy(update={"stale_ids": []}))) == {"d001_stale"}


def test_git_doc_nonce_option_items_use_balanced_codes(tmp_path: Path) -> None:
    repo = tmp_path / "nonce_option_repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")

    commit_readme(
        repo,
        "The client retries failed requests twice before surfacing an error to callers.",
        "2024-01-01T00:00:00+00:00",
        "v1",
    )
    commit_readme(
        repo,
        "The client retries failed requests four times before surfacing an error to callers.",
        "2024-02-01T00:00:00+00:00",
        "v2",
    )

    sentence_item = build_doc_items(
        repo=repo,
        repo_name="nonce_option_repo",
        paths=["README.md"],
        variants=["stale_first"],
        max_items=10,
    )[0]
    phrase_item = phrase_level_item(sentence_item)
    assert phrase_item is not None

    code_a, code_b = nonce_codes(phrase_item.id)
    code_a_item, code_b_item = nonce_option_items(phrase_item)

    assert code_a_item.question_type == "doc_revision_nonce_option"
    assert code_b_item.question_type == "doc_revision_nonce_option"
    assert code_a_item.gold_answer == code_a
    assert code_b_item.gold_answer == code_b
    assert code_a_item.metadata["option_labels"][code_a] == "four times"
    assert code_a_item.metadata["option_labels"][code_b] == "twice"
    assert code_b_item.metadata["option_labels"][code_a] == "twice"
    assert code_b_item.metadata["option_labels"][code_b] == "four times"
    assert code_a_item.metadata["stale_value"] == code_b
    assert code_b_item.metadata["stale_value"] == code_a
    assert "Answer with only the option code" in code_a_item.query
    assert code_a_item.context_blocks == phrase_item.context_blocks
    assert code_b_item.context_blocks == phrase_item.context_blocks
    assert set(detected_lineage_stale_ids(code_b_item.model_copy(update={"stale_ids": []}))) == {"d001_stale"}
    scrubbed_blocks = []
    for block in code_b_item.context_blocks:
        metadata = dict(block.metadata)
        metadata.pop("supersedes", None)
        metadata.pop("superseded_by", None)
        scrubbed_blocks.append(block.model_copy(update={"metadata": metadata, "status": "unknown"}))
    scrubbed_item = code_b_item.model_copy(
        update={"context_blocks": scrubbed_blocks, "metadata": {}, "stale_ids": []}
    )
    assert set(detected_query_ref_stale_ids(scrubbed_item)) == {"d001_stale"}
    fully_scrubbed_blocks = [
        block.model_copy(update={"metadata": {}, "status": "unknown"})
        for block in code_b_item.context_blocks
    ]
    fully_scrubbed_item = code_b_item.model_copy(
        update={"context_blocks": fully_scrubbed_blocks, "metadata": {}, "stale_ids": []}
    )
    assert set(detected_text_timestamp_stale_ids(fully_scrubbed_item)) == {"d001_stale"}
    tied_timestamp_item = fully_scrubbed_item.model_copy(
        update={
            "context_blocks": [
                block.model_copy(update={"timestamp": "2024-01-01T00:00:00+00:00"})
                for block in fully_scrubbed_item.context_blocks
            ]
        }
    )
    assert detected_text_timestamp_stale_ids(tied_timestamp_item) == []


def test_balanced_probe_round_robin_preserves_source_balance() -> None:
    assert round_robin([["a1", "a2"], ["b1"], ["c1", "c2", "c3"]]) == [
        "a1",
        "b1",
        "c1",
        "a2",
        "c2",
        "c3",
    ]


def commit_version(repo: Path, version: str, date: str) -> None:
    (repo / "package.json").write_text(json.dumps({"version": version}) + "\n", encoding="utf-8")
    git(repo, "add", "package.json")
    env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    git(repo, "commit", "-m", f"version {version}", env=env)


def commit_readme(repo: Path, sentence: str, date: str, tag: str) -> None:
    (repo / "README.md").write_text(f"# Demo\n\n{sentence}\n", encoding="utf-8")
    git(repo, "add", "README.md")
    env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    git(repo, "commit", "-m", f"docs {tag}", env=env)
    git(repo, "tag", tag, env=env)


def build_doc_items_from_sentences(old_sentence: str, new_sentence: str) -> list[BenchmarkItem]:
    return [
        BenchmarkItem(
            id="git_doc_demo_readme_md_001_stale_first",
            domain="git_doc_history",
            question_type="doc_revision_sentence",
            context_blocks=[
                ContextBlock(
                    id="d001_stale",
                    timestamp="2024-01-01T00:00:00+00:00",
                    type="git_doc_revision",
                    text=old_sentence,
                    metadata={"superseded_by": ["d001_current"]},
                ),
                ContextBlock(
                    id="d001_current",
                    timestamp="2024-02-01T00:00:00+00:00",
                    type="git_doc_revision",
                    text=new_sentence,
                    metadata={"supersedes": ["d001_stale"]},
                ),
            ],
            query="What is the current documentation sentence?",
            gold_answer=new_sentence,
            gold_evidence_ids=["d001_current"],
            stale_ids=["d001_stale"],
            metadata={
                "repo_name": "demo",
                "path": "README.md",
                "current_ref": "v2",
                "stale_ref": "v1",
                "target_value": new_sentence,
                "stale_value": old_sentence,
                "variant": "stale_first",
                "distractor_ratio": 0.0,
            },
        )
    ]


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> None:
    merged_env = None
    if env:
        import os

        merged_env = {**os.environ, **env}
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=merged_env, stdout=subprocess.DEVNULL)
