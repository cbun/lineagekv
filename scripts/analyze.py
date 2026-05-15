#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.analysis.plots import write_plots
from context_rot.analysis.reports import write_research_memo
from context_rot.eval.metrics import load_results, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize results and create plots.")
    parser.add_argument("--results", default="results/runs/context_rot_v1_heuristic.jsonl")
    parser.add_argument("--summary", default="results/context_rot_v1_summary.csv")
    parser.add_argument("--plots", default="results/plots")
    parser.add_argument("--memo", default="docs/context_rot_research_memo_v0.md")
    args = parser.parse_args()

    results = load_results(ROOT / args.results)
    summary = summarize(results)
    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    plot_paths = write_plots(results, summary, ROOT / args.plots)
    memo_path = write_research_memo(summary, results, ROOT / args.memo)
    print(f"wrote summary to {summary_path}")
    print(f"wrote memo to {memo_path}")
    for path in plot_paths:
        print(f"wrote plot {path}")


if __name__ == "__main__":
    main()
