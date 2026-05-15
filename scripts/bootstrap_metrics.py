#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for context-rot result metrics.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--high-rot-threshold", type=float, default=0.75)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (ROOT / args.results).read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No result rows found.")
    output_rows = []
    for scope, scoped in _scopes(df, args.high_rot_threshold):
        for strategy, strategy_df in scoped.groupby("strategy"):
            output_rows.extend(_bootstrap_strategy(scope, strategy, strategy_df, args.iterations, args.seed))
    output = pd.DataFrame(output_rows)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"wrote bootstrap metrics to {output_path}")


def _scopes(df: pd.DataFrame, high_rot_threshold: float):
    yield "all", df
    if "distractor_ratio" in df:
        ratios = pd.to_numeric(df["distractor_ratio"], errors="coerce")
        high = df[ratios >= high_rot_threshold]
        if not high.empty:
            yield f"high_rot_ge_{high_rot_threshold:g}", high


def _bootstrap_strategy(scope: str, strategy: str, df: pd.DataFrame, iterations: int, seed: int) -> list[dict[str, object]]:
    digest = hashlib.sha256(f"{scope}|{strategy}".encode("utf-8")).hexdigest()
    strategy_seed = seed + int(digest[:8], 16)
    rng = np.random.default_rng(strategy_seed)
    values = {
        "accuracy": df["correct"].astype(float).to_numpy(),
        "evidence_accuracy": df["evidence_correct"].astype(float).to_numpy(),
        "stale_usage_rate": df["used_stale_fact"].astype(float).to_numpy(),
        "input_tokens": df["input_tokens_estimate"].astype(float).to_numpy(),
    }
    rows: list[dict[str, object]] = []
    for metric, array in values.items():
        samples = _bootstrap_mean(array, iterations, rng)
        rows.append(
            {
                "scope": scope,
                "strategy": strategy,
                "metric": metric,
                "cases": int(len(array)),
                "mean": float(array.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
            }
        )
    return rows


def _bootstrap_mean(values: np.ndarray, iterations: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) == 0:
        return np.array([np.nan])
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    return values[indices].mean(axis=1)


if __name__ == "__main__":
    main()
