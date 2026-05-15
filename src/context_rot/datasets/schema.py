from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


Trust = Literal["low", "medium", "high"]
BlockStatus = Literal["current", "superseded", "failed", "irrelevant", "unknown"]


class ContextBlock(BaseModel):
    id: str
    timestamp: str
    type: str
    trust: Trust = "medium"
    status: BlockStatus = "unknown"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def parsed_timestamp(self) -> datetime:
        value = self.timestamp.replace("Z", "+00:00")
        return datetime.fromisoformat(value)


class BenchmarkItem(BaseModel):
    id: str
    domain: str
    question_type: str
    context_blocks: list[ContextBlock]
    query: str
    gold_answer: str
    gold_evidence_ids: list[str]
    stale_ids: list[str] = Field(default_factory=list)
    distractor_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBundle(BaseModel):
    item_id: str
    strategy: str
    context_blocks: list[ContextBlock]
    rendered_context: str
    input_tokens_estimate: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompressionResult(BaseModel):
    item_id: str
    strategy: str
    bundle: ContextBundle
    excluded_block_ids: list[str] = Field(default_factory=list)
    original_tokens_estimate: int
    compression_ratio: float
    latency_ms: float
    notes: list[str] = Field(default_factory=list)


class ModelRun(BaseModel):
    item_id: str
    strategy: str
    model_id: str
    prompt: str
    raw_output: str
    parsed_output: dict[str, Any]
    latency_ms: float


class GradingResult(BaseModel):
    item_id: str
    strategy: str
    model_id: str
    correct: bool
    evidence_correct: bool
    cited_stale_evidence: bool
    used_stale_fact: bool
    contradiction_failure: bool
    abstain_correct: bool | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


def to_plain(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
