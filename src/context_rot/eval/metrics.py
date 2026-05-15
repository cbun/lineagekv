from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_results(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = df.groupby(["model_id", "strategy", "distractor_ratio"], dropna=False)
    summary = grouped.agg(
        cases=("item_id", "count"),
        accuracy=("correct", "mean"),
        evidence_accuracy=("evidence_correct", "mean"),
        stale_usage_rate=("used_stale_fact", "mean"),
        contradiction_failure_rate=("contradiction_failure", "mean"),
        avg_input_tokens=("input_tokens_estimate", "mean"),
        avg_compression_ratio=("compression_ratio", "mean"),
        avg_total_latency_ms=("total_latency_ms", "mean"),
    ).reset_index()
    full = summary[summary["strategy"] == "full"][
        ["model_id", "distractor_ratio", "accuracy"]
    ].rename(columns={"accuracy": "full_accuracy"})
    summary = summary.merge(full, on=["model_id", "distractor_ratio"], how="left")
    summary["context_rot_delta"] = summary["accuracy"] - summary["full_accuracy"]
    summary["quality_per_1k_tokens"] = summary["accuracy"] / (summary["avg_input_tokens"] / 1000.0)
    return summary
