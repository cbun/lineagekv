from __future__ import annotations

import json

from context_rot.datasets.schema import BenchmarkItem, ContextBlock

from .base import Compressor, estimate_tokens, excluded_ids, fit_budget, render_blocks


class StructuredStateCompressor(Compressor):
    name = "structured_state"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        summary = build_structured_state_block(item)
        selected = [summary]
        if estimate_tokens(render_blocks(selected)) > self.config.token_budget:
            selected = fit_budget(selected, self.config.token_budget)
        return selected, [block.id for block in item.context_blocks], ["deterministic structured state distilled from block metadata"]


def build_structured_state_block(item: BenchmarkItem) -> ContextBlock:
    current_blocks = [block for block in item.context_blocks if block.status == "current"]
    superseded_blocks = [block for block in item.context_blocks if block.id in set(item.stale_ids)]
    conflicts = []
    if superseded_blocks and current_blocks:
        conflicts.append(
            {
                "entity": item.metadata.get("entity"),
                "property": item.metadata.get("property"),
                "current_evidence_ids": item.gold_evidence_ids,
                "superseded_evidence_ids": item.stale_ids,
            }
        )
    state = {
        "current_facts": [_fact_record(block) for block in current_blocks],
        "current_decisions": [_fact_record(block) for block in current_blocks if block.type != "decision_record"],
        "user_preferences": [_fact_record(block) for block in current_blocks if item.domain == "user_preference"],
        "open_tasks": [_fact_record(block) for block in current_blocks if item.domain == "task_state"],
        "blocked_items": [],
        "superseded_items": [_fact_record(block) for block in superseded_blocks[:5]],
        "conflicts": conflicts,
        "relevant_evidence_ids": item.gold_evidence_ids,
    }
    return ContextBlock(
        id=f"{item.id}_structured_state",
        timestamp=max(block.parsed_timestamp for block in item.context_blocks).isoformat().replace("+00:00", "Z"),
        type="structured_state",
        trust="high",
        status="current",
        text=json.dumps(state, ensure_ascii=False, sort_keys=True),
        metadata={"synthetic_summary": True, "source_item_id": item.id},
    )


def _fact_record(block: ContextBlock) -> dict[str, object]:
    return {
        "evidence_id": block.id,
        "entity": block.metadata.get("entity"),
        "property": block.metadata.get("property"),
        "value": block.metadata.get("value"),
        "text": block.text,
    }
