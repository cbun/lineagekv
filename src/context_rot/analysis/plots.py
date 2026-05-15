from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


def write_plots(results: pd.DataFrame, summary: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.append(_line_plot(summary, "accuracy", "Accuracy vs Distractor Ratio", out_dir / "accuracy_vs_distractor.png"))
    paths.append(_line_plot(summary, "stale_usage_rate", "Stale Fact Usage vs Distractor Ratio", out_dir / "stale_usage_vs_distractor.png"))
    paths.append(_scatter_tokens_accuracy(summary, out_dir / "tokens_vs_accuracy.png"))
    paths.append(_bar_plot(summary, "avg_input_tokens", "Average Input Tokens by Strategy", out_dir / "tokens_by_strategy.png"))
    paths.append(_bar_plot(summary, "avg_total_latency_ms", "Average Latency by Strategy", out_dir / "latency_by_strategy.png"))
    paths.append(_domain_breakdown(results, out_dir / "per_domain_accuracy.png"))
    return paths


def _line_plot(summary: pd.DataFrame, metric: str, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy, frame in summary.groupby("strategy"):
        frame = frame.sort_values("distractor_ratio")
        ax.plot(frame["distractor_ratio"], frame[metric], marker="o", label=strategy)
    ax.set_title(title)
    ax.set_xlabel("Distractor ratio")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _scatter_tokens_accuracy(summary: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, frame in summary.groupby("strategy"):
        ax.scatter(frame["avg_input_tokens"], frame["accuracy"], label=strategy)
    ax.set_title("Tokens vs Accuracy")
    ax.set_xlabel("Average input tokens")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _bar_plot(summary: pd.DataFrame, metric: str, title: str, path: Path) -> Path:
    frame = summary.groupby("strategy", as_index=False)[metric].mean().sort_values(metric)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(frame["strategy"], frame[metric])
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " "))
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _domain_breakdown(results: pd.DataFrame, path: Path) -> Path:
    pivot = results.groupby(["domain", "strategy"])["correct"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Per-Domain Accuracy")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
