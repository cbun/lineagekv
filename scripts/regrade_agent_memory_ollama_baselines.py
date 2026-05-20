#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.io import read_jsonl
from context_rot.eval.graders import grade_output, parse_model_json
from run_agent_memory_ollama_baselines import normalize_parsed_fields, remap_evidence_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Regrade Ollama agent-memory baseline rows with parser normalization.")
    parser.add_argument("--dataset", default="data/generated/agent_memory_action_test32.jsonl")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    items = {item.id: item for item in read_jsonl(ROOT / args.dataset)}
    output_rows: list[dict[str, Any]] = []
    for row in read_jsonl_dicts(ROOT / args.input):
        item = items[str(row["item_id"])]
        aliases = {block.id: f"m{idx:03d}" for idx, block in enumerate(item.context_blocks)}
        parsed = normalize_parsed_fields(parse_model_json(str(row.get("raw_output", ""))))
        parsed = remap_evidence_ids(parsed, aliases)
        policy = str(row.get("policy", row.get("strategy", "")))
        model_id = str(row.get("model_id", ""))
        grade = grade_output(item, policy, model_id, parsed)
        updated = dict(row)
        updated["parsed_output"] = parsed
        updated.update(grade.model_dump())
        output_rows.append(updated)
    write_jsonl_dicts(ROOT / args.output, output_rows)
    write_summary(ROOT / args.summary_output, summarize(output_rows))
    print(f"wrote {ROOT / args.output}")
    print(f"wrote {ROOT / args.summary_output}")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["policy"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model_id, policy), policy_rows in sorted(grouped.items()):
        output.append(
            {
                "model_id": model_id,
                "policy": policy,
                "cases": len(policy_rows),
                "accuracy": mean_bool(policy_rows, "correct"),
                "evidence_accuracy": mean_bool(policy_rows, "evidence_correct"),
                "stale_use_rate": mean_bool(policy_rows, "used_stale_fact"),
                "clean_action_rate": mean(row.get("correct") and not row.get("used_stale_fact") for row in policy_rows),
                "avg_latency_ms": sum(float(row["latency_ms"]) for row in policy_rows) / max(1, len(policy_rows)),
            }
        )
    return output


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return mean(bool(row.get(key)) for row in rows)


def mean(values: Any) -> float:
    values = list(values)
    return sum(1.0 if value else 0.0 for value in values) / max(1, len(values))


def read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
