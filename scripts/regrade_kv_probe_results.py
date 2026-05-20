#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.io import read_jsonl
from context_rot.datasets.schema import to_plain
from context_rot.eval.graders import grade_output
from kv_value_ablation_probe import parse_probe_output, remap_parsed_evidence_ids  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Reparse raw KV probe outputs and refresh grading fields.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parse-mode", default="json_or_answer", choices=["json", "json_or_answer"])
    args = parser.parse_args()

    items = {item.id: item for item in read_jsonl(ROOT / args.dataset)}
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (ROOT / args.input).open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            item = items[row["item_id"]]
            old_parsed = row.get("parsed_output") if isinstance(row.get("parsed_output"), dict) else {}
            parsed = parse_probe_output(row.get("raw_output", ""), item, args.parse_mode)
            alias_to_block_id = row.get("alias_to_block_id") or {}
            if alias_to_block_id:
                parsed = remap_parsed_evidence_ids(parsed, alias_to_block_id)
            # Existing probe rows store remapped evidence IDs but not the alias map.
            # Preserve those remapped IDs while allowing parser fixes to recover answers.
            if old_parsed.get("evidence_ids") and (not alias_to_block_id or not parsed.get("evidence_ids")):
                parsed["evidence_ids"] = old_parsed["evidence_ids"]
            if "used_stale_fact" not in parsed and "used_stale_fact" in old_parsed:
                parsed["used_stale_fact"] = old_parsed["used_stale_fact"]
            grade = grade_output(item, row["policy"], row["model_id"], parsed)
            row["parsed_output"] = parsed
            for key, value in to_plain(grade).items():
                if key not in {"item_id", "strategy", "model_id"}:
                    row[key] = value
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"regraded {count} rows to {output_path}")
if __name__ == "__main__":
    main()
