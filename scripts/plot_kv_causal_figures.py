#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "results/.matplotlib").resolve()))

import matplotlib.pyplot as plt


METRICS = [
    ("accuracy", "Accuracy"),
    ("evidence_accuracy", "Evidence"),
    ("stale_usage_rate", "Stale use"),
    ("clean_accuracy", "Clean"),
]

METRIC_COLORS = {
    "accuracy": "#2f6f9f",
    "evidence_accuracy": "#3f8f5f",
    "stale_usage_rate": "#b65a46",
    "clean_accuracy": "#6f5aa8",
}

EDIT_LABELS = {
    "baseline": "Full cache",
    "values": "V only",
    "keys": "K only",
    "keys_values": "K+V",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper figures for KV address/content causal maps.")
    parser.add_argument("--output-dir", default="results/kv/figures")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    main_policy = pd.read_csv(ROOT / "results/kv/kv_address_content_causal_map_policy.csv")
    qwen3_policy = pd.read_csv(ROOT / "results/kv/kv_address_content_in_family_transfer_policy.csv")
    boundary_policy = pd.read_csv(ROOT / "results/kv/kv_address_content_boundary_policy.csv")

    paths = [
        _plot_qwen25_full_block(main_policy, output_dir / "qwen25_full_block_metrics.png"),
        _plot_qwen25_evidence_deltas(output_dir / "qwen25_evidence_deltas_vs_value.png"),
        _plot_boundary_full_block(qwen3_policy, boundary_policy, output_dir / "boundary_full_block_metrics.png"),
    ]
    for path in paths:
        print(path.relative_to(ROOT))


def _full_block_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["span"].isin(["full_context", "full_block"]))].copy()


def _ordered_rows(df: pd.DataFrame, edits: list[str]) -> pd.DataFrame:
    rows = []
    for edit in edits:
        match = df[df["edit"] == edit]
        if not match.empty:
            rows.append(match.iloc[0])
    return pd.DataFrame(rows)


def _plot_qwen25_full_block(df: pd.DataFrame, path: Path) -> Path:
    models = ["Qwen2.5-0.5B", "Qwen2.5-1.5B"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, models, strict=True):
        frame = _ordered_rows(
            _full_block_rows(df[df["model"] == model]),
            ["baseline", "values", "keys", "keys_values"],
        )
        _metric_grouped_bars(ax, frame, title=model)
    axes[0].set_ylabel("Rate")
    fig.suptitle("Qwen2.5 full-block stale-span interventions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _metric_grouped_bars(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    edits = [EDIT_LABELS.get(str(edit), str(edit)) for edit in frame["edit"]]
    x_positions = list(range(len(edits)))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, (metric, label) in zip(offsets, METRICS, strict=True):
        ax.bar(
            [x + offset for x in x_positions],
            frame[metric],
            width=width,
            label=label,
            color=METRIC_COLORS[metric],
        )
    ax.set_title(title)
    ax.set_xticks(x_positions, edits)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper right")


def _plot_qwen25_evidence_deltas(path: Path) -> Path:
    rows = []
    specs = [
        (
            "Qwen2.5-0.5B",
            ROOT / "results/kv/kv_neutral_ids_key_value_keyonly_contrast_qwen25_0_5b_layers_8_23_probe80_vs_value_fullspan_pairwise.csv",
            "zero_lineage_conflict_keys_layers_8_23",
            "zero_lineage_conflict_keys_values_layers_8_23",
        ),
        (
            "Qwen2.5-1.5B",
            ROOT / "results/kv/kv_neutral_ids_key_value_keyonly_contrast_qwen25_1_5b_layers_12_27_probe40_vs_value_fullspan_pairwise.csv",
            "zero_lineage_conflict_keys_layers_12_27",
            "zero_lineage_conflict_keys_values_layers_12_27",
        ),
    ]
    for model, csv_path, key_policy, kv_policy in specs:
        frame = pd.read_csv(csv_path)
        for label, policy in [("K only - V only", key_policy), ("K+V - V only", kv_policy)]:
            row = frame[
                (frame["scope"] == "all")
                & (frame["metric"] == "evidence_correct")
                & (frame["comparator"] == policy)
            ].iloc[0]
            rows.append(
                {
                    "label": f"{model}\n{label}",
                    "delta": row["delta"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "kind": label,
                }
            )
    plot_df = pd.DataFrame(rows)
    y_positions = list(range(len(plot_df)))
    colors = ["#8b4a9c" if kind.startswith("K only") else "#b65a46" for kind in plot_df["kind"]]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axvline(0, color="#333333", linewidth=1)
    for y, (_, row), color in zip(y_positions, plot_df.iterrows(), colors, strict=True):
        ax.errorbar(
            row["delta"],
            y,
            xerr=[[row["delta"] - row["ci_low"]], [row["ci_high"] - row["delta"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
        )
    ax.set_yticks(y_positions, plot_df["label"])
    ax.set_xlabel("Evidence delta versus value-only full-block edit")
    ax.set_xlim(-1.05, 0.1)
    ax.set_title("Deleting stale keys collapses Qwen2.5 evidence routing", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_boundary_full_block(qwen3: pd.DataFrame, boundary: pd.DataFrame, path: Path) -> Path:
    qwen3_rows = _ordered_rows(
        _full_block_rows(qwen3[qwen3["model"] == "Qwen3-0.6B"]),
        ["baseline", "values", "keys", "keys_values"],
    )
    smol_rows = _ordered_rows(
        _full_block_rows(boundary[boundary["model"] == "SmolLM2-360M"]),
        ["baseline", "values", "keys_values"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    _metric_grouped_bars(axes[0], qwen3_rows, "Qwen3-0.6B boundary")
    _metric_grouped_bars(axes[1], smol_rows, "SmolLM2-360M boundary")
    axes[0].set_ylabel("Rate")
    fig.suptitle("Boundary models: cleanup without universal Qwen2.5 K/V split", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
