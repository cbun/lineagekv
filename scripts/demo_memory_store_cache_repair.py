#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import write_jsonl
from context_rot.memory import SQLiteMemoryLedger


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a memory-store lineage and cache-edit-plan demo.")
    parser.add_argument("--sqlite", default="results/kv/memory_store_lineage_demo.sqlite")
    parser.add_argument("--output", default="results/kv/kv_memory_store_lineage_demo.json")
    parser.add_argument("--dataset-output", default="data/generated/memory_store_lineage_demo.jsonl")
    args = parser.parse_args()

    sqlite_path = ROOT / args.sqlite
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()

    ledger = SQLiteMemoryLedger(sqlite_path)
    try:
        seed_demo_memories(ledger)
        query_key = "calendar.weekly_review.location"
        query = "Where should I join the weekly review now?"
        rendered = ledger.render_prompt(
            query=query,
            queried_memory_key=query_key,
            encode=byte_encode,
            include_history=True,
            neutral_aliases=True,
        )
        plan = ledger.cache_edit_plan(rendered, current_value_scale=1.0)
        item = ledger.to_benchmark_item(
            item_id="memory_store_lineage_demo_weekly_review",
            query=query,
            memory_key=query_key,
        )
        write_jsonl([item], ROOT / args.dataset_output)

        output = {
            "records": [record.model_dump() for record in ledger.records()],
            "current_records": [record.model_dump() for record in ledger.current_records()],
            "rendered": rendered.model_dump(),
            "cache_edit_plan": plan.model_dump(),
            "benchmark_item": item.model_dump(),
            "assertions": {
                "old_records_preserved": any(record.status == "superseded" for record in ledger.records(query_key)),
                "lineage_edges_created": bool(plan.edits),
                "token_spans_created": bool(rendered.spans),
                "neutral_aliases_used": all(alias.startswith("m") for alias in rendered.alias_to_source_id),
                "provenance_current_ids": plan.provenance_current_ids,
            },
        }
    finally:
        ledger.close()

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {ROOT / args.dataset_output}")


def seed_demo_memories(ledger: SQLiteMemoryLedger) -> None:
    ledger.write_memory(
        "calendar.weekly_review.location",
        "Room 3B",
        text="Memory key: calendar.weekly_review.location. Weekly review location is Room 3B.",
        timestamp="2024-01-05T09:00:00+00:00",
        record_id="weekly_review_location_stale",
    )
    ledger.write_memory(
        "calendar.weekly_review.location",
        "Zoom",
        text="Memory key: calendar.weekly_review.location. Weekly review location is Zoom.",
        timestamp="2024-03-20T09:00:00+00:00",
        record_id="weekly_review_location_current",
    )
    ledger.write_memory(
        "repo.install.package_manager",
        "pnpm",
        text="Memory key: repo.install.package_manager. JavaScript installs should use pnpm.",
        timestamp="2024-02-01T09:00:00+00:00",
        record_id="package_manager_current",
    )
    ledger.write_memory(
        "ops.deploy.window",
        "avoid Friday afternoon",
        text="Memory key: ops.deploy.window. Routine deploys should avoid Friday afternoon.",
        timestamp="2024-02-03T09:00:00+00:00",
        record_id="deploy_window_current",
    )


def byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


if __name__ == "__main__":
    main()
