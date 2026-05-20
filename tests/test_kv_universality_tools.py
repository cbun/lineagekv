from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_agent_memory_ollama_baselines import normalize_parsed_fields
from summarize_wrong_action_prevention import summarize_pairwise


def test_wrong_action_prevention_pairs_by_model_and_item() -> None:
    rows = [
        row("model-a", "case", "full_cache", correct=True, stale=True),
        row("model-a", "case", "drop", correct=True, stale=False),
        row("model-b", "case", "full_cache", correct=False, stale=False),
        row("model-b", "case", "drop", correct=True, stale=False),
    ]

    [summary] = summarize_pairwise(rows, baseline="full_cache", comparators=["drop"])

    assert summary["paired_cases"] == 2
    assert summary["baseline_wrong_actions"] == 2
    assert summary["prevented_wrong_actions"] == 2
    assert summary["wrong_action_prevention_rate"] == 1.0


def test_ollama_parser_normalizes_common_json_typos() -> None:
    parsed = normalize_parsed_fields({"answer": "Zoom", "eviidence_ids": ["m001"], "reasoning_summaary": "brief"})

    assert parsed["evidence_ids"] == ["m001"]
    assert parsed["reasoning_summary"] == "brief"


def row(model_id: str, item_id: str, policy: str, *, correct: bool, stale: bool) -> dict[str, object]:
    return {
        "model_id": model_id,
        "item_id": item_id,
        "policy": policy,
        "correct": correct,
        "evidence_correct": False,
        "used_stale_fact": stale,
    }
