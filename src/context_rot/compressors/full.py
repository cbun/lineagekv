from __future__ import annotations

from context_rot.datasets.schema import BenchmarkItem, ContextBlock

from .base import Compressor


class FullContextCompressor(Compressor):
    name = "full"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        return list(item.context_blocks), [], ["all context blocks included"]
