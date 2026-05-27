from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from context_rot.datasets.schema import BenchmarkItem, ContextBlock


MemoryStatus = Literal["current", "superseded"]


class MemoryRecord(BaseModel):
    id: str
    memory_key: str
    value: str
    text: str
    timestamp: str
    status: MemoryStatus = "current"
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RenderedMemorySpan(BaseModel):
    source_id: str
    alias: str
    memory_key: str
    status: MemoryStatus
    start: int
    end: int
    header_start: int
    header_end: int
    body_start: int
    body_end: int


class RenderedMemoryPrompt(BaseModel):
    prefix_text: str
    suffix_text: str
    rendered_text: str
    token_count: int
    spans: list[RenderedMemorySpan]
    alias_to_source_id: dict[str, str]


class CacheEdit(BaseModel):
    stale_id: str
    current_id: str
    stale_start: int
    stale_end: int
    current_start: int
    current_end: int
    layer_start: int
    layer_end: int
    zero_values: bool = True
    preserve_keys: bool = True
    current_value_scale: float = 1.0


class CacheEditPlan(BaseModel):
    edits: list[CacheEdit]
    provenance_current_ids: list[str]


class SQLiteMemoryLedger:
    """Small SQLite memory ledger used by the public LineageKV demos.

    The ledger keeps superseded records in storage, records explicit
    old-record -> new-record lineage edges, and emits token spans for cache
    repair. It is intentionally minimal: production systems can back the same
    protocol with their own memory store.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.connection.close()

    def write_memory(
        self,
        memory_key: str,
        value: str,
        *,
        text: str,
        timestamp: str,
        record_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        record_id = record_id or self._default_record_id(memory_key)
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        current_records = self.records(memory_key=memory_key, status="current")

        with self.connection:
            for record in current_records:
                self.connection.execute(
                    "UPDATE memories SET status = 'superseded' WHERE id = ?",
                    (record.id,),
                )
            self.connection.execute(
                """
                INSERT INTO memories (id, memory_key, value, text, timestamp, status, metadata_json)
                VALUES (?, ?, ?, ?, ?, 'current', ?)
                """,
                (record_id, memory_key, value, text, timestamp, metadata_json),
            )
            for record in current_records:
                self.connection.execute(
                    "INSERT OR IGNORE INTO memory_edges (stale_id, current_id) VALUES (?, ?)",
                    (record.id, record_id),
                )

        return self.record(record_id)

    def record(self, record_id: str) -> MemoryRecord:
        rows = self.records()
        for record in rows:
            if record.id == record_id:
                return record
        raise KeyError(record_id)

    def records(self, memory_key: str | None = None, status: MemoryStatus | None = None) -> list[MemoryRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if memory_key is not None:
            clauses.append("memory_key = ?")
            params.append(memory_key)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"""
            SELECT id, memory_key, value, text, timestamp, status, metadata_json
            FROM memories
            {where}
            ORDER BY timestamp, id
            """,
            params,
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def current_records(self, memory_key: str | None = None) -> list[MemoryRecord]:
        return self.records(memory_key=memory_key, status="current")

    def render_prompt(
        self,
        *,
        query: str,
        queried_memory_key: str | None = None,
        encode: Callable[[str], list[int]],
        include_history: bool = True,
        neutral_aliases: bool = True,
    ) -> RenderedMemoryPrompt:
        records = self.records() if include_history else self.current_records()
        records = self._render_order(records, queried_memory_key)

        prefix_parts = ["Memory records:\n"]
        spans: list[RenderedMemorySpan] = []
        alias_to_source_id: dict[str, str] = {}

        for idx, record in enumerate(records):
            alias = f"m{idx:03d}" if neutral_aliases else record.id
            alias_to_source_id[alias] = record.id
            header = f"[{alias}] key={record.memory_key} status={record.status} timestamp={record.timestamp}\n"
            body = f"{record.text}\n"
            before = "".join(prefix_parts)
            header_text = before + header
            block_text = header_text + body
            spans.append(
                RenderedMemorySpan(
                    source_id=record.id,
                    alias=alias,
                    memory_key=record.memory_key,
                    status=record.status,
                    start=len(encode(before)),
                    end=len(encode(block_text)),
                    header_start=len(encode(before)),
                    header_end=len(encode(header_text)),
                    body_start=len(encode(header_text)),
                    body_end=len(encode(block_text)),
                )
            )
            prefix_parts.extend([header, body])

        prefix_text = "".join(prefix_parts)
        suffix_text = (
            "\nQuestion: "
            + query
            + "\nRespond with JSON containing answer and evidence_ids using the memory aliases above.\n"
        )
        rendered_text = prefix_text + suffix_text
        return RenderedMemoryPrompt(
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            rendered_text=rendered_text,
            token_count=len(encode(rendered_text)),
            spans=spans,
            alias_to_source_id=alias_to_source_id,
        )

    def cache_edit_plan(
        self,
        rendered: RenderedMemoryPrompt,
        *,
        layer_start: int = 8,
        layer_end: int = 23,
        current_value_scale: float = 1.0,
    ) -> CacheEditPlan:
        spans = {span.source_id: span for span in rendered.spans}
        edits: list[CacheEdit] = []
        provenance_current_ids: list[str] = []
        for stale_id, current_id in self._lineage_pairs():
            stale_span = spans.get(stale_id)
            current_span = spans.get(current_id)
            if stale_span is None or current_span is None:
                continue
            edits.append(
                CacheEdit(
                    stale_id=stale_id,
                    current_id=current_id,
                    stale_start=stale_span.start,
                    stale_end=stale_span.end,
                    current_start=current_span.start,
                    current_end=current_span.end,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    current_value_scale=current_value_scale,
                )
            )
            if current_id not in provenance_current_ids:
                provenance_current_ids.append(current_id)
        edits.sort(key=lambda edit: edit.stale_start)
        provenance_current_ids.sort(key=lambda record_id: spans[record_id].start if record_id in spans else 10**9)
        return CacheEditPlan(edits=edits, provenance_current_ids=provenance_current_ids)

    def to_benchmark_item(self, *, item_id: str, query: str, memory_key: str) -> BenchmarkItem:
        current = self.current_records(memory_key=memory_key)
        if not current:
            raise ValueError(f"no current memory for key {memory_key!r}")
        stale_ids = [record.id for record in self.records(memory_key=memory_key, status="superseded")]
        gold_ids = [record.id for record in current]
        distractor_ids = [record.id for record in self.records() if record.memory_key != memory_key]
        stale_values = [record.value for record in self.records(memory_key=memory_key, status="superseded")]
        context_blocks = [
            ContextBlock(
                id=record.id,
                timestamp=record.timestamp,
                type="memory",
                trust="high",
                status=record.status,
                text=record.text,
                metadata={
                    "memory_key": record.memory_key,
                    "value": record.value,
                    "supersedes": record.supersedes,
                    "superseded_by": record.superseded_by,
                    **record.metadata,
                },
            )
            for record in self.records()
        ]
        return BenchmarkItem(
            id=item_id,
            domain="assistant_memory",
            question_type="memory_lineage_update",
            context_blocks=context_blocks,
            query=query,
            gold_answer=current[-1].value,
            gold_evidence_ids=gold_ids,
            stale_ids=stale_ids,
            distractor_ids=distractor_ids,
            metadata={
                "memory_key": memory_key,
                "target_value": current[-1].value,
                "stale_value": stale_values[-1] if stale_values else "",
                "lineage_source": "sqlite_memory_ledger",
            },
        )

    def _init_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory_key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('current', 'superseded')),
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_edges (
                    stale_id TEXT NOT NULL,
                    current_id TEXT NOT NULL,
                    PRIMARY KEY (stale_id, current_id),
                    FOREIGN KEY (stale_id) REFERENCES memories(id),
                    FOREIGN KEY (current_id) REFERENCES memories(id)
                )
                """
            )

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        supersedes = [
            edge["stale_id"]
            for edge in self.connection.execute(
                "SELECT stale_id FROM memory_edges WHERE current_id = ? ORDER BY stale_id",
                (row["id"],),
            ).fetchall()
        ]
        superseded_by = [
            edge["current_id"]
            for edge in self.connection.execute(
                "SELECT current_id FROM memory_edges WHERE stale_id = ? ORDER BY current_id",
                (row["id"],),
            ).fetchall()
        ]
        return MemoryRecord(
            id=row["id"],
            memory_key=row["memory_key"],
            value=row["value"],
            text=row["text"],
            timestamp=row["timestamp"],
            status=row["status"],
            supersedes=supersedes,
            superseded_by=superseded_by,
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def _lineage_pairs(self) -> list[tuple[str, str]]:
        rows = self.connection.execute(
            "SELECT stale_id, current_id FROM memory_edges ORDER BY stale_id, current_id"
        ).fetchall()
        return [(row["stale_id"], row["current_id"]) for row in rows]

    @staticmethod
    def _default_record_id(memory_key: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", memory_key).strip("_").lower()
        return f"{slug}_{uuid4().hex[:12]}"

    @staticmethod
    def _render_order(records: list[MemoryRecord], queried_memory_key: str | None) -> list[MemoryRecord]:
        if queried_memory_key is None:
            return records
        return sorted(
            records,
            key=lambda record: (record.memory_key != queried_memory_key, record.timestamp, record.id),
        )
