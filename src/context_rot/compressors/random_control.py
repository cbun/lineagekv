from __future__ import annotations

from random import Random

from context_rot.datasets.schema import BenchmarkItem, ContextBlock

from .base import Compressor, estimate_tokens, excluded_ids, render_blocks, stable_seed


class RandomControlCompressor(Compressor):
    name = "random_control"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        rng = Random(stable_seed(item.id, self.name, base_seed=self.config.random_seed))
        shuffled = list(item.context_blocks)
        rng.shuffle(shuffled)
        selected: list[ContextBlock] = []
        for block in shuffled:
            candidate = selected + [block]
            if estimate_tokens(render_blocks(candidate)) <= self.config.token_budget or not selected:
                selected.append(block)
        selected_ids = {block.id for block in selected}
        ordered = [block for block in item.context_blocks if block.id in selected_ids]
        return ordered, excluded_ids(item, ordered), ["random same-budget control"]
