#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl
from context_rot.datasets.schema import to_plain
from context_rot.eval.graders import grade_output, parse_model_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Reparse raw model outputs and refresh grading fields.")
    parser.add_argument("--dataset", default="data/generated/context_rot_v1.jsonl")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
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
            parsed = parse_model_json(row.get("raw_output", ""))
            grade = grade_output(item, row["strategy"], row["model_id"], parsed)
            row["parsed_output"] = parsed
            for key, value in to_plain(grade).items():
                if key not in {"item_id", "strategy", "model_id"}:
                    row[key] = value
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"regraded {count} rows to {output_path}")


if __name__ == "__main__":
    main()
