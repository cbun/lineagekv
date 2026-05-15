from __future__ import annotations

from context_rot.datasets.schema import BenchmarkItem, ContextBlock

from .base import Compressor, excluded_ids, normalize_text
from .retrieval import _top_ranked_in_original_order, rank_blocks


class DedupRetrievalCompressor(Compressor):
    name = "dedup_retrieval"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        seen: set[str] = set()
        deduped: list[ContextBlock] = []
        removed: list[str] = []
        for block in item.context_blocks:
            key = normalize_text(block.text)
            if key in seen:
                removed.append(block.id)
            else:
                seen.add(key)
                deduped.append(block)
        deduped_item = item.model_copy(update={"context_blocks": deduped})
        ranked = rank_blocks(deduped_item, recency_weight=0.0, trust_weight=0.0, stale_penalty=0.0)
        selected = _top_ranked_in_original_order(deduped_item, ranked, self.config.retrieval_top_k, self.config.token_budget)
        excluded = excluded_ids(item, selected)
        return selected, excluded, [f"removed {len(removed)} normalized duplicates before retrieval"]
