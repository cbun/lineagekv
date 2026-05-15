from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from context_rot.datasets.schema import BenchmarkItem, ContextBlock

from .base import Compressor, excluded_ids, estimate_tokens, fit_budget, render_blocks


class RetrievalCompressor(Compressor):
    name = "retrieval"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        ranked = rank_blocks(item, recency_weight=0.0, trust_weight=0.0, stale_penalty=0.0)
        selected = _top_ranked_in_original_order(item, ranked, self.config.retrieval_top_k, self.config.token_budget)
        return selected, excluded_ids(item, selected), ["tf-idf top-k retrieval"]


class RecencyWeightedRetrievalCompressor(Compressor):
    name = "recency_weighted_retrieval"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        ranked = rank_blocks(item, recency_weight=0.25, trust_weight=0.15, stale_penalty=0.20)
        selected = _top_ranked_in_original_order(item, ranked, self.config.retrieval_top_k, self.config.token_budget)
        return selected, excluded_ids(item, selected), ["tf-idf retrieval with freshness/trust/stale weights"]


class WindowedRetrievalCompressor(Compressor):
    name = "windowed_retrieval"

    def select_blocks(self, item: BenchmarkItem) -> tuple[list[ContextBlock], list[str], list[str]]:
        ranked = rank_blocks_by_windows(item, window_radius=2, char_weight=0.5)
        selected = _top_ranked_score_first_in_chronological_order(
            item,
            ranked,
            self.config.retrieval_top_k,
            self.config.token_budget,
            expansion_radius=1,
        )
        return selected, excluded_ids(item, selected), ["word+char tf-idf over local windows with score-first budget fill"]


def rank_blocks(
    item: BenchmarkItem,
    recency_weight: float,
    trust_weight: float,
    stale_penalty: float,
) -> list[tuple[ContextBlock, float]]:
    corpus = [item.query] + [block.text for block in item.context_blocks]
    if not item.context_blocks:
        return []
    matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(corpus)
    similarities = (matrix[1:] @ matrix[0].T).toarray().ravel()
    timestamps = np.array([block.parsed_timestamp.timestamp() for block in item.context_blocks], dtype=float)
    if timestamps.max() > timestamps.min():
        freshness = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    else:
        freshness = np.zeros_like(timestamps)
    trust_scores = np.array([{"low": 0.0, "medium": 0.5, "high": 1.0}.get(block.trust, 0.0) for block in item.context_blocks])
    stale_scores = np.array([1.0 if block.status in {"superseded", "failed"} else 0.0 for block in item.context_blocks])
    scores = similarities + recency_weight * freshness + trust_weight * trust_scores - stale_penalty * stale_scores
    return sorted(zip(item.context_blocks, scores), key=lambda pair: pair[1], reverse=True)


def rank_blocks_by_windows(
    item: BenchmarkItem,
    window_radius: int = 2,
    char_weight: float = 0.5,
) -> list[tuple[ContextBlock, float]]:
    if not item.context_blocks:
        return []
    texts = _window_texts(item.context_blocks, window_radius)
    corpus = [item.query] + texts
    word_matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(corpus)
    word_similarities = (word_matrix[1:] @ word_matrix[0].T).toarray().ravel()
    char_matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True).fit_transform(corpus)
    char_similarities = (char_matrix[1:] @ char_matrix[0].T).toarray().ravel()
    scores = word_similarities + char_weight * char_similarities
    return sorted(zip(item.context_blocks, scores), key=lambda pair: pair[1], reverse=True)


def _window_texts(blocks: list[ContextBlock], radius: int) -> list[str]:
    texts: list[str] = []
    for idx, _block in enumerate(blocks):
        start = max(0, idx - radius)
        end = min(len(blocks), idx + radius + 1)
        texts.append("\n".join(block.text for block in blocks[start:end]))
    return texts


def _top_ranked_in_original_order(
    item: BenchmarkItem,
    ranked: list[tuple[ContextBlock, float]],
    top_k: int,
    token_budget: int,
) -> list[ContextBlock]:
    selected_ids: set[str] = set()
    selected: list[ContextBlock] = []
    for block, _score in ranked[:top_k]:
        selected_ids.add(block.id)
    for block in item.context_blocks:
        if block.id in selected_ids:
            candidate = selected + [block]
            if estimate_tokens(render_blocks(candidate)) <= token_budget or not selected:
                selected.append(block)
    if not selected and ranked:
        selected = [ranked[0][0]]
    return fit_budget(selected, token_budget)


def _top_ranked_score_first_in_chronological_order(
    item: BenchmarkItem,
    ranked: list[tuple[ContextBlock, float]],
    top_k: int,
    token_budget: int,
    expansion_radius: int = 0,
) -> list[ContextBlock]:
    positions = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    candidates: list[ContextBlock] = []
    seen: set[str] = set()
    for block, _score in ranked[:top_k]:
        idx = positions[block.id]
        start = max(0, idx - expansion_radius)
        end = min(len(item.context_blocks), idx + expansion_radius + 1)
        for candidate in item.context_blocks[start:end]:
            if candidate.id not in seen:
                seen.add(candidate.id)
                candidates.append(candidate)
    selected: list[ContextBlock] = []
    for block in candidates:
        candidate = selected + [block]
        if estimate_tokens(render_blocks(candidate)) <= token_budget or not selected:
            selected.append(block)
    if not selected and ranked:
        selected = [ranked[0][0]]
    selected_ids = {block.id for block in selected}
    return [block for block in item.context_blocks if block.id in selected_ids]
