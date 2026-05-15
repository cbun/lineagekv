from __future__ import annotations

from context_rot.datasets.schema import BenchmarkItem, ContextBlock

from .base import Compressor, excluded_ids, fit_budget


class RecencyCompressor(Compressor):
    name = "recency"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        newest_first = sorted(item.context_blocks, key=lambda block: block.parsed_timestamp, reverse=True)
        selected_newest_first = fit_budget(newest_first, self.config.token_budget)
        selected_ids = {block.id for block in selected_newest_first}
        selected = [block for block in item.context_blocks if block.id in selected_ids]
        return selected, excluded_ids(item, selected), [f"kept newest blocks within {self.config.token_budget} token estimate"]
