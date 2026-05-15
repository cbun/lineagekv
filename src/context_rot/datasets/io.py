from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import BenchmarkItem, to_plain


def write_jsonl(items: Iterable[BenchmarkItem], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(to_plain(item), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[BenchmarkItem]:
    rows: list[BenchmarkItem] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(BenchmarkItem.model_validate_json(line))
    return rows
