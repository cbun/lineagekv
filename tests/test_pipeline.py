from __future__ import annotations

from context_rot.compressors import CompressionConfig, build_compressor
from context_rot.datasets.schema import BenchmarkItem, ContextBlock
from context_rot.datasets.generate_synthetic import generate_items
from context_rot.eval.graders import grade_output, parse_model_json
from context_rot.eval.prompting import render_prompt
from context_rot.models.heuristic_adapter import HeuristicModel
from context_rot.models import build_model
from context_rot.models.ollama_adapter import OllamaModel


def test_generator_produces_expected_v1_size() -> None:
    items = generate_items(base_limit=50)
    assert len(items) == 1250
    assert len({item.metadata["base_index"] for item in items}) == 50
    assert {item.metadata["distractor_ratio"] for item in items} == {0.0, 0.25, 0.5, 0.75, 0.9}


def test_ollama_adapter_registers_without_network_call() -> None:
    model = build_model("ollama", ollama_model="qwen3:8b")
    assert isinstance(model, OllamaModel)
    assert model.model_id == "ollama:qwen3:8b"


def test_hybrid_excludes_stale_duplicate_blocks() -> None:
    item = next(
        candidate
        for candidate in generate_items(base_limit=50)
        if candidate.id == "billing_ops_041_wrong_duplicate_10x_d90"
    )
    full = build_compressor("full", CompressionConfig(token_budget=1200)).compress(item)
    hybrid = build_compressor("hybrid", CompressionConfig(token_budget=1200)).compress(item)
    full_ids = {block.id for block in full.bundle.context_blocks}
    hybrid_ids = {block.id for block in hybrid.bundle.context_blocks}
    assert set(item.stale_ids) & full_ids
    assert not (set(item.stale_ids) & hybrid_ids)
    assert set(item.gold_evidence_ids) & hybrid_ids or hybrid.bundle.context_blocks[0].type == "structured_state"
    assert '"evidence_id": "b041_current"' in hybrid.bundle.context_blocks[0].text
    assert '"value": "hold credit pending usage audit"' in hybrid.bundle.context_blocks[0].text


def test_longmemeval_focused_oracle_uses_metadata_context() -> None:
    item = generate_items(base_limit=1)[0].model_copy(
        update={
            "domain": "external_longmemeval_s",
            "metadata": {"focused_input": "Relevant focused memory says the answer is 3.", "focused_input_tokens": 9},
        }
    )
    result = build_compressor("longmemeval_focused_oracle", CompressionConfig(token_budget=1200)).compress(item)
    assert result.bundle.context_blocks[0].type == "focused_oracle_context"
    assert "Relevant focused memory" in result.bundle.rendered_context


def test_evidence_oracle_selects_gold_blocks() -> None:
    item = generate_items(base_limit=1)[0]
    result = build_compressor("evidence_oracle", CompressionConfig(token_budget=1200)).compress(item)
    selected_ids = {block.id for block in result.bundle.context_blocks}
    assert set(item.gold_evidence_ids).issubset(selected_ids)
    assert not (set(item.stale_ids) & selected_ids)


def test_windowed_retrieval_includes_neighbor_turns() -> None:
    item = BenchmarkItem(
        id="windowed_retrieval_case",
        domain="external_locomo",
        question_type="locomo_category_2",
        context_blocks=[
            ContextBlock(
                id="D1:1",
                timestamp="2023-05-08T00:00:00Z",
                type="dialogue_turn",
                text="Melanie: Take a look at this.",
            ),
            ContextBlock(
                id="D1:2",
                timestamp="2023-05-08T00:00:00Z",
                type="dialogue_turn",
                text="Caroline: Is this your sunrise painting?",
            ),
            ContextBlock(
                id="D1:3",
                timestamp="2023-05-08T00:00:00Z",
                type="dialogue_turn",
                text="Melanie: Yes, I painted it last year.",
            ),
        ],
        query="When did Melanie paint a sunrise?",
        gold_answer="last year",
        gold_evidence_ids=["D1:1"],
    )
    result = build_compressor("windowed_retrieval", CompressionConfig(token_budget=1200, retrieval_top_k=1)).compress(item)
    selected_ids = {block.id for block in result.bundle.context_blocks}
    assert "D1:1" in selected_ids


def test_parser_salvages_truncated_json_and_normalizes_evidence() -> None:
    parsed = parse_model_json('{"answer": "hold credit pending usage audit", "evidence_ids": ["id=b041_current"], "used_stale_fact": false')
    item = next(
        candidate
        for candidate in generate_items(base_limit=50)
        if candidate.id == "billing_ops_041_current_late_d00"
    )
    grade = grade_output(item, "full", "test", parsed)
    assert grade.correct
    assert grade.evidence_correct
    unquoted = parse_model_json('{"answer": "yesterday", "evidence_ids": [id=D1:3], "used_stale_fact": false')
    assert unquoted["evidence_ids"] == ["D1:3"]
    numeric_grade = grade_output(item, "full", "test", 3)
    assert numeric_grade.metrics["answer"] == "3"


def test_prompt_lists_allowed_evidence_ids() -> None:
    item = next(
        candidate
        for candidate in generate_items(base_limit=1)
        if candidate.id == "user_preference_001_current_late_d00"
    )
    compression = build_compressor("full", CompressionConfig(token_budget=1200)).compress(item)
    prompt = render_prompt(item, compression)
    assert "Allowed evidence_ids:" in prompt
    assert "b001_current" in prompt
    assert "include at least one matching ID" in prompt


def test_locomo_prompt_asks_for_minimal_direct_evidence_ids() -> None:
    item = BenchmarkItem(
        id="locomo_prompt_case",
        domain="external_locomo",
        question_type="locomo_category_2",
        context_blocks=[
            ContextBlock(
                id="D1:3",
                timestamp="2023-05-08T00:00:00Z",
                type="dialogue_turn",
                text="Caroline: I went to a support group yesterday.",
            )
        ],
        query="When did Caroline go to the support group?",
        gold_answer="7 May 2023",
        gold_evidence_ids=["D1:3"],
    )
    prompt = render_prompt(item, build_compressor("full", CompressionConfig(token_budget=1200)).compress(item))
    assert "smallest sufficient set" in prompt
    assert "directly support the answer" in prompt


def test_grader_accepts_conservative_semantic_aliases() -> None:
    policy = next(
        candidate
        for candidate in generate_items(base_limit=10)
        if candidate.id == "policy_approval_006_current_late_d25"
    )
    policy_grade = grade_output(policy, "full", "test", {"answer": "No", "evidence_ids": ["b006_current"]})
    assert policy_grade.correct
    kb = next(
        candidate
        for candidate in generate_items(base_limit=35)
        if candidate.id == "knowledge_base_031_current_late_d00"
    )
    kb_grade = grade_output(kb, "full", "test", {"answer": "30 days", "evidence_ids": ["b031_current"]})
    assert kb_grade.correct
    launch = next(
        candidate
        for candidate in generate_items(base_limit=40)
        if candidate.id == "multi_hop_memory_036_current_late_d00"
    )
    launch_grade = grade_output(
        launch,
        "hybrid",
        "test",
        {"answer": "Trailmind launch should be launched in Slack with a short demo clip.", "evidence_ids": ["b036_current"]},
    )
    assert launch_grade.correct


def test_grader_accepts_numeric_values_inside_verbose_gold_answers() -> None:
    item = BenchmarkItem(
        id="numeric_gold_case",
        domain="external_longmemeval_s",
        question_type="long_chat_memory",
        context_blocks=[],
        query="How many hours in total?",
        gold_answer="15 hours for getting to the three destinations (or 30 hours for the round trip)",
        gold_evidence_ids=[],
    )
    assert grade_output(item, "full", "test", {"answer": "15", "evidence_ids": []}).correct
    assert grade_output(item, "full", "test", {"answer": "fifteen hours", "evidence_ids": []}).correct
    assert not grade_output(item, "full", "test", {"answer": "3", "evidence_ids": []}).correct


def test_locomo_grader_handles_partial_answers_and_relative_dates() -> None:
    item = BenchmarkItem(
        id="locomo_test",
        domain="external_locomo",
        question_type="locomo_category_2",
        context_blocks=[
            ContextBlock(
                id="D1:3",
                timestamp="2023-05-08T13:56:00Z",
                type="dialogue_turn",
                text="Caroline: I went to a LGBTQ support group yesterday and it was so powerful.",
            )
        ],
        query="When did Caroline go to the LGBTQ support group?",
        gold_answer="7 May 2023",
        gold_evidence_ids=["D1:3"],
    )
    date_grade = grade_output(item, "evidence_oracle", "test", {"answer": "yesterday", "evidence_ids": ["D1:3"]})
    assert date_grade.correct
    identity_item = item.model_copy(update={"gold_answer": "Transgender woman", "query": "What is Caroline's identity?"})
    identity_grade = grade_output(
        identity_item,
        "evidence_oracle",
        "test",
        {"answer": "transgender", "evidence_ids": ["D1:3"]},
    )
    assert identity_grade.correct


def test_heuristic_scores_hybrid_better_than_full_on_stale_duplicate_case() -> None:
    item = next(
        candidate
        for candidate in generate_items(base_limit=50)
        if candidate.id == "billing_ops_041_wrong_duplicate_10x_d90"
    )
    model = HeuristicModel()
    full = build_compressor("full", CompressionConfig(token_budget=2000)).compress(item)
    hybrid = build_compressor("hybrid", CompressionConfig(token_budget=1200)).compress(item)
    full_grade = grade_output(item, "full", model.model_id, parse_model_json(model.generate("", item, full)))
    hybrid_grade = grade_output(item, "hybrid", model.model_id, parse_model_json(model.generate("", item, hybrid)))
    assert not full_grade.correct
    assert full_grade.used_stale_fact
    assert hybrid_grade.correct
    assert not hybrid_grade.used_stale_fact
