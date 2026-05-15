from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd

from .schema import BenchmarkItem, ContextBlock


def adapt_longmemeval_s(
    csv_path: str | Path,
    limit: int = 50,
    chunk_chars: int = 3500,
    max_chunks: int = 40,
) -> list[BenchmarkItem]:
    df = pd.read_csv(csv_path, nrows=limit)
    items: list[BenchmarkItem] = []
    for row_idx, row in df.iterrows():
        custom_id = str(row["custom_id"])
        full_input = str(row["full_input"])
        blocks = [
            ContextBlock(
                id=f"lme_{custom_id}_c{idx + 1:03d}",
                timestamp="2026-01-01T00:00:00Z",
                type="chat_history_chunk",
                trust="medium",
                status="unknown",
                text=chunk,
                metadata={"chunk_index": idx, "source": "cleaned-longmemeval-s"},
            )
            for idx, chunk in enumerate(_chunk_text(full_input, chunk_chars)[:max_chunks])
        ]
        items.append(
            BenchmarkItem(
                id=f"longmemeval_s_{custom_id}",
                domain="external_longmemeval_s",
                question_type="long_chat_memory",
                context_blocks=blocks,
                query=str(row["question"]),
                gold_answer=str(row["answer"]),
                gold_evidence_ids=[f"longmemeval_s_{custom_id}_focused_oracle"],
                stale_ids=[],
                distractor_ids=[],
                metadata={
                    "source_dataset": "kellyhongg/cleaned-longmemeval-s",
                    "row_index": int(row_idx),
                    "full_input_tokens": int(row.get("full_input_tokens", 0)),
                    "focused_input_tokens": int(row.get("focused_input_tokens", 0)),
                    "focused_input": str(row.get("focused_input", ""))[:12000],
                    "distractor_ratio": None,
                    "external_adapter": "longmemeval_s_v0",
                },
            )
        )
    return items


def adapt_locomo(raw_json_path: str | Path, limit: int = 100) -> list[BenchmarkItem]:
    data = json.loads(Path(raw_json_path).read_text(encoding="utf-8"))
    items: list[BenchmarkItem] = []
    for conversation_idx, sample in enumerate(data):
        blocks = _locomo_blocks(sample)
        block_ids = {block.id for block in blocks}
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            evidence_ids = [str(eid) for eid in qa.get("evidence", []) if str(eid) in block_ids]
            items.append(
                BenchmarkItem(
                    id=f"locomo_{sample.get('sample_id', conversation_idx)}_qa{qa_idx + 1:03d}",
                    domain="external_locomo",
                    question_type=f"locomo_category_{qa.get('category', 'unknown')}",
                    context_blocks=blocks,
                    query=str(qa.get("question", "")),
                    gold_answer=str(qa.get("answer", "")),
                    gold_evidence_ids=evidence_ids,
                    stale_ids=[],
                    distractor_ids=[],
                    metadata={
                        "source_dataset": "Percena/locomo-mc10",
                        "sample_id": sample.get("sample_id"),
                        "qa_category": qa.get("category"),
                        "distractor_ratio": None,
                        "external_adapter": "locomo_v0",
                    },
                )
            )
            if len(items) >= limit:
                return items
    return items


def adapt_nolima(
    root: str | Path,
    limit: int = 50,
    seed: int = 20260515,
    haystack_chars: int = 2500,
) -> list[BenchmarkItem]:
    root = Path(root)
    rng = Random(seed)
    needles = json.loads((root / "needlesets" / "needle_set.json").read_text(encoding="utf-8"))
    haystack_files = sorted((root / "haystack" / "rand_shuffle").glob("*.txt"))
    items: list[BenchmarkItem] = []
    for needle_idx, needle in enumerate(needles):
        character_set = needle.get("character_set") or ["Alex"]
        tests = list((needle.get("tests") or {}).items())
        for test_name, test in tests:
            character = character_set[(len(items) + needle_idx) % len(character_set)]
            args = list(test.get("input_args", []))
            if not args:
                continue
            filled_needle = _format_template(str(needle.get("needle", "")), character, args)
            question_template = (needle.get("questions") or {}).get("onehop") or next(iter((needle.get("questions") or {}).values()))
            question = _format_template(str(question_template), character, args)
            haystack_text = haystack_files[len(items) % len(haystack_files)].read_text(encoding="utf-8", errors="ignore")
            haystack_blocks = [
                ContextBlock(
                    id=f"nolima_{needle.get('id', needle_idx)}_{test_name}_h{idx + 1:02d}",
                    timestamp="2026-01-01T00:00:00Z",
                    type="haystack_chunk",
                    trust="low",
                    status="irrelevant",
                    text=chunk,
                    metadata={"source": "NoLiMa", "chunk_index": idx},
                )
                for idx, chunk in enumerate(_chunk_text(haystack_text[: haystack_chars * 3], haystack_chars))
            ]
            insert_at = rng.randrange(len(haystack_blocks) + 1)
            needle_block = ContextBlock(
                id=f"nolima_{needle.get('id', needle_idx)}_{test_name}_needle",
                timestamp="2026-01-02T00:00:00Z",
                type="needle",
                trust="high",
                status="current",
                text=filled_needle,
                metadata={"source": "NoLiMa", "reasoning_type": needle.get("reasoning_type"), "answer": character},
            )
            blocks = haystack_blocks[:insert_at] + [needle_block] + haystack_blocks[insert_at:]
            items.append(
                BenchmarkItem(
                    id=f"nolima_{needle.get('id', needle_idx)}_{test_name}",
                    domain="external_nolima",
                    question_type=str(needle.get("reasoning_type", "semantic_retrieval")),
                    context_blocks=blocks,
                    query=question,
                    gold_answer=character,
                    gold_evidence_ids=[needle_block.id],
                    stale_ids=[],
                    distractor_ids=[block.id for block in haystack_blocks],
                    metadata={
                        "source_dataset": "amodaresi/NoLiMa",
                        "needle_id": needle.get("id"),
                        "test_name": test_name,
                        "distractor_ratio": len(haystack_blocks) / max(len(blocks), 1),
                        "external_adapter": "nolima_v0",
                    },
                )
            )
            if len(items) >= limit:
                return items
    return items


def _chunk_text(text: str, chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    for start in range(0, len(text), chunk_chars):
        chunk = text[start : start + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _locomo_blocks(sample: dict[str, Any]) -> list[ContextBlock]:
    conv = sample.get("conversation", {})
    blocks: list[ContextBlock] = []
    for key, value in conv.items():
        if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(value, list):
            continue
        session_num = key.split("_")[1]
        timestamp = _parse_locomo_time(str(conv.get(f"session_{session_num}_date_time", "")))
        for turn in value:
            dia_id = str(turn.get("dia_id", f"{key}:{len(blocks) + 1}"))
            blocks.append(
                ContextBlock(
                    id=dia_id,
                    timestamp=timestamp,
                    type="dialogue_turn",
                    trust="medium",
                    status="unknown",
                    text=f"{turn.get('speaker', 'speaker')}: {turn.get('text', '')}",
                    metadata={"source": "LoCoMo", "session": key, "speaker": turn.get("speaker")},
                )
            )
    return blocks


def _parse_locomo_time(value: str) -> str:
    # The raw dates are informal. Preserve coarse ordering with a stable fallback.
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            parsed = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return parsed.isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    return "2026-01-01T00:00:00Z"


def _format_template(template: str, character: str, args: list[str]) -> str:
    rendered = template.replace("{CHAR}", character)
    for idx, arg in enumerate(args, start=1):
        rendered = rendered.replace("{" + str(idx) + "}", str(arg))
    return rendered
