from __future__ import annotations

from .base import Compressor, CompressionConfig
from .contradiction import ContradictionOracleCompressor
from .dedup import DedupRetrievalCompressor
from .external import BudgetedFullCompressor, EvidenceOracleCompressor, LongMemEvalFocusedOracleCompressor
from .full import FullContextCompressor
from .hybrid import HybridCompressor
from .random_control import RandomControlCompressor
from .recency import RecencyCompressor
from .retrieval import RecencyWeightedRetrievalCompressor, RetrievalCompressor, WindowedRetrievalCompressor
from .structured_state import StructuredStateCompressor


def build_compressor(name: str, config: CompressionConfig | None = None) -> Compressor:
    config = config or CompressionConfig()
    registry = {
        "full": FullContextCompressor,
        "budgeted_full": BudgetedFullCompressor,
        "evidence_oracle": EvidenceOracleCompressor,
        "longmemeval_focused_oracle": LongMemEvalFocusedOracleCompressor,
        "recency": RecencyCompressor,
        "retrieval": RetrievalCompressor,
        "windowed_retrieval": WindowedRetrievalCompressor,
        "dedup_retrieval": DedupRetrievalCompressor,
        "recency_weighted_retrieval": RecencyWeightedRetrievalCompressor,
        "contradiction_oracle": ContradictionOracleCompressor,
        "structured_state": StructuredStateCompressor,
        "hybrid": HybridCompressor,
        "random_control": RandomControlCompressor,
    }
    if name not in registry:
        raise KeyError(f"Unknown compressor strategy: {name}")
    return registry[name](config)


__all__ = [
    "Compressor",
    "CompressionConfig",
    "build_compressor",
]
