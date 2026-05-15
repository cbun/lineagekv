from __future__ import annotations

from context_rot.datasets.schema import BenchmarkItem, ContextBlock

from .base import Compressor, excluded_ids, fit_budget


class BudgetedFullCompressor(Compressor):
    name = "budgeted_full"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        selected = fit_budget(list(item.context_blocks), self.config.token_budget)
        return selected, excluded_ids(item, selected), [f"kept chronological context within {self.config.token_budget} token estimate"]


class LongMemEvalFocusedOracleCompressor(Compressor):
    name = "longmemeval_focused_oracle"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        focused = str(item.metadata.get("focused_input", "")).strip()
        if not focused:
            selected = fit_budget(list(item.context_blocks), self.config.token_budget)
            return selected, excluded_ids(item, selected), ["focused_input missing; fell back to budgeted full context"]
        block = ContextBlock(
            id=f"{item.id}_focused_oracle",
            timestamp="2026-01-01T00:00:00Z",
            type="focused_oracle_context",
            trust="high",
            status="current",
            text=focused,
            metadata={
                "source": item.metadata.get("source_dataset", "unknown"),
                "focused_input_tokens": item.metadata.get("focused_input_tokens"),
                "oracle_context": True,
            },
        )
        return [block], [context_block.id for context_block in item.context_blocks], ["used dataset-provided focused/oracle context"]


class EvidenceOracleCompressor(Compressor):
    name = "evidence_oracle"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        gold_ids = set(item.gold_evidence_ids)
        selected = [block for block in item.context_blocks if block.id in gold_ids]
        if not selected:
            selected = fit_budget(list(item.context_blocks), self.config.token_budget)
            return selected, excluded_ids(item, selected), ["gold evidence ids missing; fell back to budgeted full context"]
        selected = fit_budget(selected, self.config.token_budget)
        return selected, excluded_ids(item, selected), ["used gold evidence ids as an oracle upper bound"]
