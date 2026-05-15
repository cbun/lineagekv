from __future__ import annotations

from context_rot.datasets.schema import BenchmarkItem, ContextBlock

from .base import Compressor, excluded_ids, fit_budget


class ContradictionOracleCompressor(Compressor):
    name = "contradiction_oracle"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        gold_ids = set(item.gold_evidence_ids)
        stale_ids = set(item.stale_ids)
        selected = [block for block in item.context_blocks if block.id in gold_ids]
        fillers = [
            block
            for block in item.context_blocks
            if block.id not in gold_ids and block.id not in stale_ids and block.status != "superseded"
        ]
        selected = fit_budget(selected + fillers, self.config.token_budget)
        return selected, excluded_ids(item, selected), ["oracle labels remove known stale/conflicting facts"]
