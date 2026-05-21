#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

METRICS = ("correct", "evidence_correct", "used_stale_fact", "clean_correct")


@dataclass(frozen=True)
class MatrixConfig:
    model_label: str
    band: str
    result_path: str
    policies: dict[str, tuple[str, str]]


MATRICES = [
    MatrixConfig(
        model_label="Qwen2.5-0.5B",
        band="8-23",
        result_path="results/kv/kv_neutral_ids_key_value_keyonly_contrast_qwen25_0_5b_layers_8_23_probe80.jsonl",
        policies={
            "full_cache": ("baseline", "full_context"),
            "zero_lineage_conflict_values_layers_8_23": ("values", "full_block"),
            "zero_lineage_conflict_keys_layers_8_23": ("keys", "full_block"),
            "zero_lineage_conflict_keys_values_layers_8_23": ("keys_values", "full_block"),
            "zero_lineage_conflict_values_layers_8_23_span_header": ("values", "header"),
            "zero_lineage_conflict_keys_layers_8_23_span_header": ("keys", "header"),
            "zero_lineage_conflict_keys_values_layers_8_23_span_header": ("keys_values", "header"),
            "zero_lineage_conflict_values_layers_8_23_span_body": ("values", "body"),
            "zero_lineage_conflict_keys_layers_8_23_span_body": ("keys", "body"),
            "zero_lineage_conflict_keys_values_layers_8_23_span_body": ("keys_values", "body"),
        },
    ),
    MatrixConfig(
        model_label="Qwen2.5-1.5B",
        band="12-27",
        result_path="results/kv/kv_neutral_ids_key_value_keyonly_contrast_qwen25_1_5b_layers_12_27_probe40.jsonl",
        policies={
            "full_cache": ("baseline", "full_context"),
            "zero_lineage_conflict_values_layers_12_27": ("values", "full_block"),
            "zero_lineage_conflict_keys_layers_12_27": ("keys", "full_block"),
            "zero_lineage_conflict_keys_values_layers_12_27": ("keys_values", "full_block"),
            "zero_lineage_conflict_values_layers_12_27_span_header": ("values", "header"),
            "zero_lineage_conflict_keys_layers_12_27_span_header": ("keys", "header"),
            "zero_lineage_conflict_keys_values_layers_12_27_span_header": ("keys_values", "header"),
            "zero_lineage_conflict_values_layers_12_27_span_body": ("values", "body"),
            "zero_lineage_conflict_keys_layers_12_27_span_body": ("keys", "body"),
            "zero_lineage_conflict_keys_values_layers_12_27_span_body": ("keys_values", "body"),
        },
    ),
]

IN_FAMILY_TRANSFER_MATRICES = [
    MatrixConfig(
        model_label="Qwen3-0.6B",
        band="9-27",
        result_path="results/kv/kv_qwen3_0_6b_neutral_ids_key_value_keyonly_contrast_layers_9_27_probe20.jsonl",
        policies={
            "full_cache": ("baseline", "full_context"),
            "zero_lineage_conflict_values_layers_9_27": ("values", "full_block"),
            "zero_lineage_conflict_keys_layers_9_27": ("keys", "full_block"),
            "zero_lineage_conflict_keys_values_layers_9_27": ("keys_values", "full_block"),
            "zero_lineage_conflict_values_layers_9_27_span_header": ("values", "header"),
            "zero_lineage_conflict_keys_layers_9_27_span_header": ("keys", "header"),
            "zero_lineage_conflict_keys_values_layers_9_27_span_header": ("keys_values", "header"),
            "zero_lineage_conflict_values_layers_9_27_span_body": ("values", "body"),
            "zero_lineage_conflict_keys_layers_9_27_span_body": ("keys", "body"),
            "zero_lineage_conflict_keys_values_layers_9_27_span_body": ("keys_values", "body"),
        },
    ),
]


CROSS_FAMILY_BOUNDARY_MATRICES = [
    MatrixConfig(
        model_label="SmolLM2-360M",
        band="0-31",
        result_path="results/kv/kv_neutral_ids_key_value_contrast_smolm2_360m_layers_0_31_probe80.jsonl",
        policies={
            "full_cache": ("baseline", "full_context"),
            "zero_lineage_conflict_values_layers_0_31": ("values", "full_block"),
            "zero_lineage_conflict_keys_values_layers_0_31": ("keys_values", "full_block"),
            "zero_lineage_conflict_values_layers_0_31_span_header": ("values", "header"),
            "zero_lineage_conflict_keys_values_layers_0_31_span_header": ("keys_values", "header"),
            "zero_lineage_conflict_values_layers_0_31_span_body": ("values", "body"),
            "zero_lineage_conflict_keys_values_layers_0_31_span_body": ("keys_values", "body"),
        },
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build K/V causal-map tables from neutral-ID result JSONL.")
    parser.add_argument(
        "--output-policy",
        default="results/kv/kv_address_content_causal_map_policy.csv",
        help="Policy-level causal-map CSV.",
    )
    parser.add_argument(
        "--output-contrast",
        default="results/kv/kv_address_content_causal_map_kv_vs_value.csv",
        help="K+V minus value-only same-span contrast CSV.",
    )
    parser.add_argument(
        "--output-report",
        default="docs/kv_address_content_causal_map.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--output-boundary-policy",
        default="results/kv/kv_address_content_boundary_policy.csv",
        help="Boundary-model policy-level CSV.",
    )
    parser.add_argument(
        "--output-boundary-contrast",
        default="results/kv/kv_address_content_boundary_kv_vs_value.csv",
        help="Boundary-model K+V minus value-only same-span contrast CSV.",
    )
    parser.add_argument(
        "--output-transfer-policy",
        default="results/kv/kv_address_content_in_family_transfer_policy.csv",
        help="In-family transfer/boundary policy-level CSV.",
    )
    parser.add_argument(
        "--output-transfer-contrast",
        default="results/kv/kv_address_content_in_family_transfer_kv_vs_value.csv",
        help="In-family transfer/boundary K+V minus value-only same-span contrast CSV.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    policy_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    report_sections: list[dict[str, Any]] = []
    for matrix in MATRICES:
        rows = read_rows(ROOT / matrix.result_path)
        by_policy = grouped(rows, "policy")
        by_item = group_by_item(rows)
        ensure_policies(matrix, by_policy)

        policy_rows.extend(build_policy_rows(matrix, by_policy, by_item, args.bootstrap_samples, args.seed))
        matrix_contrasts = build_contrast_rows(matrix, by_item, args.bootstrap_samples, args.seed)
        contrast_rows.extend(matrix_contrasts)
        report_sections.append(build_report_section(matrix, by_policy, matrix_contrasts))

    transfer_policy_rows: list[dict[str, Any]] = []
    transfer_contrast_rows: list[dict[str, Any]] = []
    transfer_sections: list[dict[str, Any]] = []
    for matrix in IN_FAMILY_TRANSFER_MATRICES:
        rows = read_rows(ROOT / matrix.result_path)
        by_policy = grouped(rows, "policy")
        by_item = group_by_item(rows)
        ensure_policies(matrix, by_policy)

        transfer_policy_rows.extend(build_policy_rows(matrix, by_policy, by_item, args.bootstrap_samples, args.seed))
        matrix_contrasts = build_contrast_rows(matrix, by_item, args.bootstrap_samples, args.seed)
        transfer_contrast_rows.extend(matrix_contrasts)
        transfer_sections.append(build_report_section(matrix, by_policy, matrix_contrasts))

    cross_family_policy_rows: list[dict[str, Any]] = []
    cross_family_contrast_rows: list[dict[str, Any]] = []
    cross_family_sections: list[dict[str, Any]] = []
    for matrix in CROSS_FAMILY_BOUNDARY_MATRICES:
        rows = read_rows(ROOT / matrix.result_path)
        by_policy = grouped(rows, "policy")
        by_item = group_by_item(rows)
        ensure_policies(matrix, by_policy)

        cross_family_policy_rows.extend(build_policy_rows(matrix, by_policy, by_item, args.bootstrap_samples, args.seed))
        matrix_contrasts = build_contrast_rows(matrix, by_item, args.bootstrap_samples, args.seed)
        cross_family_contrast_rows.extend(matrix_contrasts)
        cross_family_sections.append(build_report_section(matrix, by_policy, matrix_contrasts))

    write_csv(ROOT / args.output_policy, policy_rows)
    write_csv(ROOT / args.output_contrast, contrast_rows)
    write_csv(ROOT / args.output_transfer_policy, transfer_policy_rows)
    write_csv(ROOT / args.output_transfer_contrast, transfer_contrast_rows)
    write_csv(ROOT / args.output_boundary_policy, cross_family_policy_rows)
    write_csv(ROOT / args.output_boundary_contrast, cross_family_contrast_rows)
    write_report(
        ROOT / args.output_report,
        report_sections,
        transfer_sections,
        cross_family_sections,
        args.output_policy,
        args.output_contrast,
        args.output_transfer_policy,
        args.output_transfer_contrast,
        args.output_boundary_policy,
        args.output_boundary_contrast,
    )
    print(f"wrote {ROOT / args.output_policy}")
    print(f"wrote {ROOT / args.output_contrast}")
    print(f"wrote {ROOT / args.output_transfer_policy}")
    print(f"wrote {ROOT / args.output_transfer_contrast}")
    print(f"wrote {ROOT / args.output_boundary_policy}")
    print(f"wrote {ROOT / args.output_boundary_contrast}")
    print(f"wrote {ROOT / args.output_report}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["clean_correct"] = bool(row.get("correct")) and not bool(row.get("used_stale_fact"))
                rows.append(row)
    return rows


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return dict(groups)


def group_by_item(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_item[str(row["item_id"])][str(row["policy"])] = row
    return dict(by_item)


def ensure_policies(matrix: MatrixConfig, by_policy: dict[str, list[dict[str, Any]]]) -> None:
    missing = sorted(set(matrix.policies) - set(by_policy))
    if missing:
        raise SystemExit(f"{matrix.model_label} is missing policies: {', '.join(missing)}")


def build_policy_rows(
    matrix: MatrixConfig,
    by_policy: dict[str, list[dict[str, Any]]],
    by_item: dict[str, dict[str, dict[str, Any]]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for policy, (edit, span) in matrix.policies.items():
        rows = by_policy[policy]
        deltas = {}
        if policy != "full_cache":
            deltas = paired_delta(by_item, "full_cache", policy, bootstrap_samples, seed)
        output.append(
            {
                "model": matrix.model_label,
                "band": matrix.band,
                "edit": edit,
                "span": span,
                "policy": policy,
                "cases": len(rows),
                "accuracy": mean_metric(rows, "correct"),
                "evidence_accuracy": mean_metric(rows, "evidence_correct"),
                "stale_usage_rate": mean_metric(rows, "used_stale_fact"),
                "clean_accuracy": mean_metric(rows, "clean_correct"),
                **flatten_deltas("vs_full", deltas),
            }
        )
    return output


def build_contrast_rows(
    matrix: MatrixConfig,
    by_item: dict[str, dict[str, dict[str, Any]]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for span in ("full_block", "header", "body"):
        value_policy = policy_for(matrix, "values", span)
        kv_policy = policy_for(matrix, "keys_values", span)
        deltas = paired_delta(by_item, value_policy, kv_policy, bootstrap_samples, seed)
        output.append(
            {
                "model": matrix.model_label,
                "band": matrix.band,
                "span": span,
                "baseline_edit": "values",
                "comparator_edit": "keys_values",
                "baseline_policy": value_policy,
                "comparator_policy": kv_policy,
                **flatten_deltas("kv_minus_value", deltas),
            }
        )
    return output


def policy_for(matrix: MatrixConfig, edit: str, span: str) -> str:
    for policy, labels in matrix.policies.items():
        if labels == (edit, span):
            return policy
    raise KeyError((matrix.model_label, edit, span))


def mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    return sum(1.0 if row.get(metric) else 0.0 for row in rows) / len(rows)


def paired_delta(
    by_item: dict[str, dict[str, dict[str, Any]]],
    baseline: str,
    comparator: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    complete = [rows for rows in by_item.values() if baseline in rows and comparator in rows]
    if not complete:
        raise SystemExit(f"No complete paired items for {baseline} versus {comparator}")
    output: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        baseline_values = [as_float(rows[baseline].get(metric)) for rows in complete]
        comparator_values = [as_float(rows[comparator].get(metric)) for rows in complete]
        diffs = [right - left for left, right in zip(baseline_values, comparator_values)]
        ci_low, ci_high = bootstrap_ci(diffs, bootstrap_samples, seed)
        output[metric] = {
            "cases": float(len(diffs)),
            "baseline_mean": mean(baseline_values),
            "comparator_mean": mean(comparator_values),
            "delta": mean(diffs),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return output


def as_float(value: Any) -> float:
    return 1.0 if value else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(diffs: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    low_idx = int(0.025 * (samples - 1))
    high_idx = int(0.975 * (samples - 1))
    return estimates[low_idx], estimates[high_idx]


def flatten_deltas(prefix: str, deltas: dict[str, dict[str, float]]) -> dict[str, float | str]:
    flat: dict[str, float | str] = {}
    for metric in METRICS:
        values = deltas.get(metric)
        metric_name = metric.replace("correct", "accuracy")
        if not values:
            flat[f"{prefix}_{metric_name}_delta"] = ""
            flat[f"{prefix}_{metric_name}_ci_low"] = ""
            flat[f"{prefix}_{metric_name}_ci_high"] = ""
            continue
        flat[f"{prefix}_{metric_name}_delta"] = values["delta"]
        flat[f"{prefix}_{metric_name}_ci_low"] = values["ci_low"]
        flat[f"{prefix}_{metric_name}_ci_high"] = values["ci_high"]
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report_section(
    matrix: MatrixConfig,
    by_policy: dict[str, list[dict[str, Any]]],
    contrasts: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = []
    for policy, (edit, span) in matrix.policies.items():
        rows = by_policy[policy]
        summary.append(
            {
                "edit": edit,
                "span": span,
                "accuracy": mean_metric(rows, "correct"),
                "evidence": mean_metric(rows, "evidence_correct"),
                "stale": mean_metric(rows, "used_stale_fact"),
                "clean": mean_metric(rows, "clean_correct"),
            }
        )
    return {"matrix": matrix, "summary": summary, "contrasts": contrasts}


def write_report(
    path: Path,
    sections: list[dict[str, Any]],
    transfer_sections: list[dict[str, Any]],
    cross_family_sections: list[dict[str, Any]],
    policy_csv: str,
    contrast_csv: str,
    transfer_policy_csv: str,
    transfer_contrast_csv: str,
    cross_family_policy_csv: str,
    cross_family_contrast_csv: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KV Address/Content Causal Map",
        "",
        "Neutral-ID K/V matrices test whether stale-cache repair comes from removing value content while preserving cache addresses, or from destructive key+value deletion.",
        "",
        f"Generated Qwen tables: `{policy_csv}` and `{contrast_csv}`.",
        f"Generated in-family transfer tables: `{transfer_policy_csv}` and `{transfer_contrast_csv}`.",
        f"Generated cross-family boundary tables: `{cross_family_policy_csv}` and `{cross_family_contrast_csv}`.",
        "",
        "## Main Result",
        "",
        "Across Qwen2.5-0.5B and Qwen2.5-1.5B, K+V edits suppress stale use more aggressively, but value-only edits preserve evidence/currentness substantially better. This supports an address/content split in Qwen2.5: stale values carry harmful content, while keys preserve useful retrieval/citation structure.",
        "",
    ]
    for section in sections:
        append_matrix_section(lines, section)
    if transfer_sections:
        lines.extend(
            [
                "## In-Family Transfer Boundary",
                "",
                "Qwen3-0.6B reproduces the stale-cleanup and header/body separation pattern, but not the Qwen2.5 K/V address/content evidence split. In this 20-case transfer, full-block value-only, key-only, and K+V edits are behaviorally identical, and key-only deletion does not collapse evidence.",
                "",
            ]
        )
        for section in transfer_sections:
            append_matrix_section(lines, section, heading="###")
    if cross_family_sections:
        lines.extend(
            [
                "## Cross-Family Boundary",
                "",
                "SmolLM2-360M keeps the broad stale-cleanup pattern but does not reproduce the Qwen address/content evidence split. Evidence is weak across policies, and K+V full-block suppression increases evidence relative to value-only while reducing stale use more aggressively.",
                "",
            ]
        )
        for section in cross_family_sections:
            append_matrix_section(lines, section, heading="###")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Full-block value-only suppression is the best mechanism-shaped repair in both Qwen sizes: it restores evidence/currentness while keeping answer behavior usable.",
            "- Full-block K+V suppression reduces stale use more, but it consistently loses evidence accuracy relative to value-only suppression.",
            "- Qwen2.5 key-only suppression is even more destructive than K+V for the full-block/header spans, directly supporting the interpretation that stale keys carry useful address structure.",
            "- Header-only value suppression carries most evidence/currentness repair in Qwen2.5-1.5B, while body-only edits affect raw answer behavior without repairing stale evidence.",
            "- Qwen3-0.6B is an in-family boundary: the stale-block/header repair transfers, but full-block value-only, key-only, and K+V edits collapse to the same behavior on this sample.",
            "- SmolLM2-360M is a boundary case: value-only stale suppression still improves answer/clean tradeoffs, but the K/V evidence dissociation is not universal at this scale/family point.",
            "- The current address/content causal claim is strongest for Qwen2.5 neutral-ID contexts; broader-family and newer-Qwen claims should be scoped to value-only stale cleanup unless larger models replicate the K/V split.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def append_matrix_section(lines: list[str], section: dict[str, Any], heading: str = "##") -> None:
    matrix: MatrixConfig = section["matrix"]
    lines.extend([f"{heading} {matrix.model_label} Layers {matrix.band}", ""])
    lines.extend(
        [
            "| Edit | Span | Accuracy | Evidence | Stale Use | Clean |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in section["summary"]:
        lines.append(
            "| {edit} | {span} | {accuracy:.3f} | {evidence:.3f} | {stale:.3f} | {clean:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "| Span | K+V - Value Evidence Delta | 95% CI | K+V - Value Stale Delta | 95% CI |",
            "| --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in section["contrasts"]:
        evidence_delta = float(row["kv_minus_value_evidence_accuracy_delta"])
        evidence_low = float(row["kv_minus_value_evidence_accuracy_ci_low"])
        evidence_high = float(row["kv_minus_value_evidence_accuracy_ci_high"])
        stale_delta = float(row["kv_minus_value_used_stale_fact_delta"])
        stale_low = float(row["kv_minus_value_used_stale_fact_ci_low"])
        stale_high = float(row["kv_minus_value_used_stale_fact_ci_high"])
        lines.append(
            f"| {row['span']} | {evidence_delta:.3f} | [{evidence_low:.3f}, {evidence_high:.3f}] | {stale_delta:.3f} | [{stale_low:.3f}, {stale_high:.3f}] |"
        )
    lines.append("")


if __name__ == "__main__":
    main()
