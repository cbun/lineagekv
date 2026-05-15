from __future__ import annotations

import re
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from context_rot.datasets.schema import BenchmarkItem, CompressionResult, ContextBlock, ContextBundle


@dataclass(frozen=True)
class CompressionConfig:
    token_budget: int = 1200
    retrieval_top_k: int = 8
    random_seed: int = 20260515


class Compressor:
    name = "base"

    def __init__(self, config: CompressionConfig):
        self.config = config

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        raise NotImplementedError

    def compress(self, item: BenchmarkItem) -> CompressionResult:
        started = time.perf_counter()
        original_tokens = estimate_tokens(render_blocks(item.context_blocks))
        blocks, excluded_ids, notes = self.select_blocks(item)
        rendered = render_blocks(blocks)
        bundle = ContextBundle(
            item_id=item.id,
            strategy=self.name,
            context_blocks=blocks,
            rendered_context=rendered,
            input_tokens_estimate=estimate_tokens(rendered),
            metadata={
                "selected_block_ids": [block.id for block in blocks],
                "selected_block_count": len(blocks),
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000
        compression_ratio = bundle.input_tokens_estimate / max(original_tokens, 1)
        return CompressionResult(
            item_id=item.id,
            strategy=self.name,
            bundle=bundle,
            excluded_block_ids=excluded_ids,
            original_tokens_estimate=original_tokens,
            compression_ratio=compression_ratio,
            latency_ms=latency_ms,
            notes=notes,
        )


def render_blocks(blocks: Iterable[ContextBlock]) -> str:
    lines: list[str] = []
    for block in blocks:
        lines.append(
            f"[id={block.id} timestamp={block.timestamp} type={block.type} "
            f"trust={block.trust} status={block.status}]\n{block.text}"
        )
    return "\n\n".join(lines)


def estimate_tokens(text: str) -> int:
    # Cheap and stable approximation. The benchmark stores estimates, not billable token counts.
    return max(1, int(len(text) / 4))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"duplicate stale note \d+:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_seed(*parts: str, base_seed: int = 0) -> int:
    digest = sha256(("|".join(parts) + f"|{base_seed}").encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def fit_budget(blocks: list[ContextBlock], token_budget: int) -> list[ContextBlock]:
    selected: list[ContextBlock] = []
    for block in blocks:
        candidate = selected + [block]
        if estimate_tokens(render_blocks(candidate)) <= token_budget or not selected:
            selected.append(block)
        else:
            break
    return selected


def excluded_ids(item: BenchmarkItem, selected: list[ContextBlock]) -> list[str]:
    selected_ids = {block.id for block in selected}
    return [block.id for block in item.context_blocks if block.id not in selected_ids]
