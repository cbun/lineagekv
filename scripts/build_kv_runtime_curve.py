#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_INPUTS = [
    (
        "6.7k d48/s4 32-case",
        6790,
        "results/kv/kv_mlx_qwen25_7b_4bit_agent_memory_long_context_test32_refined_policies_regraded_persistent_runtime.csv",
    ),
    (
        "31k d230/s4 4-case",
        31300,
        "results/kv/kv_mlx_qwen25_7b_4bit_agent_memory_long_context_test4_d230_s4_persistent_runtime.csv",
    ),
    (
        "32k d236/s4 2-case",
        32115,
        "results/kv/kv_mlx_qwen25_7b_4bit_agent_memory_long_context_test2_d236_s4_persistent_runtime.csv",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact runtime curve CSV for KV cache repair.")
    parser.add_argument("--output", default="results/kv/kv_public_runtime_curve.csv")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for label, approx_tokens, path in DEFAULT_INPUTS:
        input_path = ROOT / path
        with input_path.open("r", newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        if not source_rows:
            raise SystemExit(f"No rows in {input_path}")
        row = source_rows[0]
        rows.append(
            {
                "label": label,
                "approx_total_tokens": approx_tokens,
                "cases": row["cases"],
                "prompt_deletion_update_ms": row["avg_prompt_deletion_update_ms"],
                "cache_repair_update_ms": row["avg_cache_repair_update_ms"],
                "speedup_ratio": row["speedup_ratio"],
                "prefill_saved_ms": row["avg_prefill_saved_ms"],
                "source_csv": path,
            }
        )

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row["label"], "speedup", f"{float(row['speedup_ratio']):.2f}x")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
