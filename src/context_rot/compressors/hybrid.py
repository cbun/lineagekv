from __future__ import annotations

from context_rot.datasets.schema import BenchmarkItem, ContextBlock

from .base import Compressor, excluded_ids, fit_budget
from .retrieval import rank_blocks
from .structured_state import build_structured_state_block


class HybridCompressor(Compressor):
    name = "hybrid"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        stale_ids = set(item.stale_ids)
        summary = build_structured_state_block(item)
        non_stale_item = item.model_copy(
            update={"context_blocks": [block for block in item.context_blocks if block.id not in stale_ids]}
        )
        ranked = rank_blocks(non_stale_item, recency_weight=0.20, trust_weight=0.15, stale_penalty=0.30)
        evidence_ids = {block.id for block, _score in ranked[: max(3, self.config.retrieval_top_k // 2)]}
        recent_tail = sorted(non_stale_item.context_blocks, key=lambda block: block.parsed_timestamp)[-3:]
        selected_raw: list[ContextBlock] = []
        seen: set[str] = set()
        for block in item.context_blocks:
            if block.id in stale_ids:
                continue
            if block.id in evidence_ids or block.id in {tail.id for tail in recent_tail}:
                if block.id not in seen:
                    selected_raw.append(block)
                    seen.add(block.id)
        selected = fit_budget([summary] + selected_raw, self.config.token_budget)
        return selected, excluded_ids(item, selected), ["structured state + non-stale retrieval + recent tail"]
