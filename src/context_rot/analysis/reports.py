from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def write_research_memo(summary: pd.DataFrame, results: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    high_rot = summary[summary["distractor_ratio"] >= 0.75].copy()
    best_rows = high_rot.sort_values("accuracy", ascending=False).head(8)
    full_rows = high_rot[high_rot["strategy"] == "full"]
    hybrid_rows = high_rot[high_rot["strategy"] == "hybrid"]
    if not full_rows.empty and not hybrid_rows.empty:
        merged = hybrid_rows.merge(
            full_rows[["model_id", "distractor_ratio", "accuracy", "avg_input_tokens", "stale_usage_rate"]],
            on=["model_id", "distractor_ratio"],
            suffixes=("_hybrid", "_full"),
        )
        if not merged.empty:
            avg_delta = merged["accuracy_hybrid"].mean() - merged["accuracy_full"].mean()
            token_reduction = 1 - (merged["avg_input_tokens_hybrid"].mean() / merged["avg_input_tokens_full"].mean())
            stale_delta = merged["stale_usage_rate_full"].mean() - merged["stale_usage_rate_hybrid"].mean()
        else:
            avg_delta = token_reduction = stale_delta = float("nan")
    else:
        avg_delta = token_reduction = stale_delta = float("nan")

    lines = [
        "# Context Rot Research Memo v0",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Status",
        "",
        "This memo summarizes the reproducible local benchmark harness and deterministic sanity run. "
        "The heuristic adapter is intentionally labeled as a pipeline validator, not final LLM evidence. "
        "Use `--model mlx --mlx-model <id>` for actual local model validation.",
        "",
        "## Headline Sanity Result",
        "",
        f"- High-rot hybrid minus full accuracy: {avg_delta:.3f}",
        f"- High-rot hybrid token reduction vs full: {token_reduction:.3f}",
        f"- High-rot stale-usage reduction vs full: {stale_delta:.3f}",
        "",
        "## Best High-Rot Rows",
        "",
        best_rows[[
            "model_id",
            "strategy",
            "distractor_ratio",
            "cases",
            "accuracy",
            "stale_usage_rate",
            "avg_input_tokens",
            "context_rot_delta",
        ]].to_markdown(index=False) if not best_rows.empty else "No high-rot rows found.",
        "",
        "## Failure Taxonomy",
        "",
        "- Full context fails when duplicate stale blocks appear after the current decision.",
        "- Recency can recover newer facts, but it loses older still-relevant support evidence in split-relevant cases.",
        "- Plain retrieval is vulnerable to semantically similar stale facts.",
        "- Oracle and hybrid variants separate the upper bound from production-feasible heuristics.",
        "",
        "## Recommendation",
        "",
        "Proceed to local MLX model validation on a 100- to 250-case subset, then repeat on a second cached local model. "
        "Only use the deterministic run as evidence that the harness, metrics, plots, and compression arms work end to end.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
