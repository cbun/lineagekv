from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_kv_tradeoffs import comparison_summary, complete_items, policy_summary


def test_tradeoff_summary_tracks_answer_preservation_and_stale_cleanup() -> None:
    rows = [
        row("a", "full_cache", correct=True, evidence=False, stale=True),
        row("a", "drop", correct=False, evidence=True, stale=False),
        row("a", "q4", correct=True, evidence=True, stale=False),
        row("b", "full_cache", correct=True, evidence=True, stale=False),
        row("b", "drop", correct=False, evidence=True, stale=False),
        row("b", "q4", correct=True, evidence=True, stale=False),
        row("c", "full_cache", correct=False, evidence=False, stale=True),
        row("c", "drop", correct=False, evidence=True, stale=False),
        row("c", "q4", correct=True, evidence=True, stale=False),
    ]

    paired = complete_items(rows, ["full_cache", "drop", "q4"])
    policies = policy_summary(paired, ["full_cache", "drop", "q4"])
    comparisons = comparison_summary(paired, "full_cache", ["drop", "q4"])

    q4_policy = next(policy for policy in policies if policy["policy"] == "q4")
    assert q4_policy["accuracy"] == 1.0
    assert q4_policy["clean_accuracy"] == 1.0

    drop = next(comparison for comparison in comparisons if comparison["comparator"] == "drop")
    q4 = next(comparison for comparison in comparisons if comparison["comparator"] == "q4")

    assert drop["answer_preservation_rate"] == 0.0
    assert q4["answer_preservation_rate"] == 1.0
    assert drop["stale_cleanup_rate"] == 1.0
    assert q4["stale_cleanup_rate"] == 1.0
    assert q4["answer_rescue_rate"] == 1.0


def row(item_id: str, policy: str, *, correct: bool, evidence: bool, stale: bool) -> dict[str, object]:
    return {
        "item_id": item_id,
        "policy": policy,
        "correct": correct,
        "evidence_correct": evidence,
        "used_stale_fact": stale,
    }
