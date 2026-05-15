from __future__ import annotations

import json

from context_rot.datasets.schema import BenchmarkItem, CompressionResult


class HeuristicModel:
    """Deterministic sanity model with a deliberate context-rot failure mode.

    This adapter exists to validate the benchmark pipeline and compressors when a
    local LLM is unavailable. It should not be presented as final model evidence.
    """

    model_id = "heuristic-context-rot-simulator"

    def generate(self, prompt: str, item: BenchmarkItem, compression: CompressionResult) -> str:
        selected = compression.bundle.context_blocks
        selected_ids = [block.id for block in selected]
        selected_id_set = set(selected_ids)
        gold_ids = set(item.gold_evidence_ids)
        stale_ids = set(item.stale_ids)
        target_value = str(item.metadata.get("target_value", item.gold_answer))
        stale_value = str(item.metadata.get("stale_value", ""))
        structured_gold_ids = _structured_gold_ids(selected)
        has_gold = bool(gold_ids & selected_id_set) or bool(gold_ids & structured_gold_ids)
        stale_selected = [block for block in selected if block.id in stale_ids or block.status == "superseded"]
        gold_positions = [idx for idx, block in enumerate(selected) if block.id in gold_ids]
        stale_positions = [idx for idx, block in enumerate(selected) if block.id in stale_ids or block.status == "superseded"]

        answer = "I cannot determine the current answer from the provided context."
        evidence_ids: list[str] = []
        used_stale = False
        confidence = 0.25

        if has_gold:
            answer = target_value
            evidence_ids = [
                block_id
                for block_id in item.gold_evidence_ids
                if block_id in selected_id_set or block_id in structured_gold_ids
            ]
            confidence = 0.85
            stale_after_gold = stale_positions and gold_positions and max(stale_positions) > min(gold_positions)
            stale_majority = len(stale_selected) >= 3 and compression.strategy in {"full", "recency", "retrieval", "random_control"}
            if stale_after_gold and stale_majority:
                answer = stale_value
                evidence_ids = [block.id for block in stale_selected[:2]]
                used_stale = True
                confidence = 0.62
        elif stale_selected:
            answer = stale_value
            evidence_ids = [stale_selected[0].id]
            used_stale = True
            confidence = 0.45

        return json.dumps(
            {
                "answer": answer,
                "evidence_ids": evidence_ids,
                "used_stale_fact": used_stale,
                "confidence": confidence,
                "abstain": not (has_gold or stale_selected),
                "reasoning_summary": "Selected the best available current evidence; stale duplicates can mislead this simulator.",
            },
            ensure_ascii=False,
        )


def _structured_gold_ids(selected) -> set[str]:
    evidence_ids: set[str] = set()
    for block in selected:
        if block.type != "structured_state":
            continue
        try:
            state = json.loads(block.text)
        except json.JSONDecodeError:
            continue
        for evidence_id in state.get("relevant_evidence_ids", []):
            evidence_ids.add(str(evidence_id))
    return evidence_ids
