from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from context_rot.datasets.io import read_jsonl
from context_rot.memory import SQLiteMemoryLedger


ROOT = Path(__file__).resolve().parents[1]


def byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def test_sqlite_memory_ledger_preserves_history_and_emits_cache_edit_plan(tmp_path: Path) -> None:
    ledger = SQLiteMemoryLedger(tmp_path / "memories.sqlite")
    try:
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

        records = {record.id: record for record in ledger.records()}
        assert records["weekly_review_location_stale"].status == "superseded"
        assert records["weekly_review_location_stale"].superseded_by == ["weekly_review_location_current"]
        assert records["weekly_review_location_current"].status == "current"
        assert records["weekly_review_location_current"].supersedes == ["weekly_review_location_stale"]

        rendered = ledger.render_prompt(
            query="Where should I join the weekly review now?",
            queried_memory_key="calendar.weekly_review.location",
            encode=byte_encode,
            include_history=True,
            neutral_aliases=True,
        )
        assert rendered.token_count == len(byte_encode(rendered.rendered_text))
        assert set(rendered.alias_to_source_id.values()) == set(records)
        assert all(alias.startswith("m") for alias in rendered.alias_to_source_id)
        assert "weekly_review_location_current" not in rendered.prefix_text

        spans = {span.source_id: span for span in rendered.spans}
        assert spans["weekly_review_location_stale"].body_start < spans["weekly_review_location_stale"].body_end
        assert spans["weekly_review_location_current"].body_start < spans["weekly_review_location_current"].body_end

        plan = ledger.cache_edit_plan(rendered, layer_start=8, layer_end=23, current_value_scale=1.0)
        assert plan.provenance_current_ids == ["weekly_review_location_current"]
        assert len(plan.edits) == 1
        edit = plan.edits[0]
        assert edit.stale_id == "weekly_review_location_stale"
        assert edit.current_id == "weekly_review_location_current"
        assert edit.stale_start == spans["weekly_review_location_stale"].start
        assert edit.stale_end == spans["weekly_review_location_stale"].end
        assert edit.current_start == spans["weekly_review_location_current"].start
        assert edit.current_end == spans["weekly_review_location_current"].end
        assert edit.zero_values is True
        assert edit.preserve_keys is True

        item = ledger.to_benchmark_item(
            item_id="memory_store_lineage_demo_weekly_review",
            query="Where should I join the weekly review now?",
            memory_key="calendar.weekly_review.location",
        )
        assert item.gold_answer == "Zoom"
        assert item.gold_evidence_ids == ["weekly_review_location_current"]
        assert item.stale_ids == ["weekly_review_location_stale"]
        assert item.distractor_ids == ["package_manager_current"]
        assert any("Room 3B" in block.text for block in item.context_blocks)
    finally:
        ledger.close()


def test_memory_store_cache_repair_demo_writes_lineage_artifacts(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "memory_store_lineage_demo.sqlite"
    output_path = tmp_path / "kv_memory_store_lineage_demo.json"
    dataset_path = tmp_path / "memory_store_lineage_demo.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/demo_memory_store_cache_repair.py"),
            "--sqlite",
            str(sqlite_path),
            "--output",
            str(output_path),
            "--dataset-output",
            str(dataset_path),
        ],
        check=True,
        cwd=ROOT,
    )

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["assertions"] == {
        "old_records_preserved": True,
        "lineage_edges_created": True,
        "token_spans_created": True,
        "neutral_aliases_used": True,
        "provenance_current_ids": ["weekly_review_location_current"],
    }

    plan = artifact["cache_edit_plan"]
    assert plan["edits"][0]["stale_id"] == "weekly_review_location_stale"
    assert plan["edits"][0]["current_id"] == "weekly_review_location_current"
    assert plan["edits"][0]["preserve_keys"] is True

    items = read_jsonl(dataset_path)
    assert len(items) == 1
    assert items[0].gold_answer == "Zoom"
    assert items[0].stale_ids == ["weekly_review_location_stale"]
