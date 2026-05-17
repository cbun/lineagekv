#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.io import read_jsonl
from kv_value_ablation_probe import (
    detected_conflict_stale_ids,
    detected_lineage_stale_ids,
    detected_query_ref_stale_ids,
    detected_text_timestamp_stale_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit non-oracle KV stale-span detector against labels.")
    parser.add_argument("--dataset", default="data/generated/context_rot_v1.jsonl")
    parser.add_argument("--output", default="results/kv/kv_conflict_detector_audit.csv")
    parser.add_argument("--distractor-ratio", type=float, default=0.9)
    parser.add_argument(
        "--detector-mode",
        choices=["cue_aware", "timestamp_only", "duplicate_aware", "lineage", "query_ref", "text_timestamp"],
        default="cue_aware",
        help=(
            "cue_aware uses explicit current/supersedes text cues; timestamp_only ignores them; "
            "duplicate_aware ignores current-text cues and prefers singleton values over repeated conflict bursts; "
            "lineage reads explicit supersedes/superseded_by metadata; "
            "query_ref resolves the target revision from the query and block repo/path/ref metadata; "
            "text_timestamp uses visible option text, block text, and timestamps without lineage/ref metadata."
        ),
    )
    parser.add_argument(
        "--variants",
        default="wrong_duplicate_10x,current_late,current_early_stale_late,stale_repeated_after",
    )
    args = parser.parse_args()

    variants = {value for value in args.variants.split(",") if value}
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "items": 0, "exact": 0}
    for item in read_jsonl(ROOT / args.dataset):
        if (
            float(item.metadata.get("distractor_ratio", -1.0)) != args.distractor_ratio
            or str(item.metadata.get("variant", "")) not in variants
        ):
            continue
        if args.detector_mode == "lineage":
            predicted = set(detected_lineage_stale_ids(item))
        elif args.detector_mode == "query_ref":
            predicted = set(detected_query_ref_stale_ids(item))
        elif args.detector_mode == "text_timestamp":
            predicted = set(detected_text_timestamp_stale_ids(item))
        else:
            predicted = set(
                detected_conflict_stale_ids(
                    item,
                    use_current_text_cues=args.detector_mode == "cue_aware",
                    prefer_singleton_conflicts=args.detector_mode == "duplicate_aware",
                )
            )
        gold = set(item.stale_ids)
        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        exact = predicted == gold
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["items"] += 1
        totals["exact"] += int(exact)
        rows.append(
            {
                "item_id": item.id,
                "domain": item.domain,
                "variant": item.metadata.get("variant"),
                "detector_mode": args.detector_mode,
                "gold_count": len(gold),
                "predicted_count": len(predicted),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "exact_match": exact,
                "predicted_ids": " ".join(sorted(predicted)),
                "missed_ids": " ".join(sorted(gold - predicted)),
                "extra_ids": " ".join(sorted(predicted - gold)),
            }
        )

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "item_id",
            "domain",
            "variant",
            "detector_mode",
            "gold_count",
            "predicted_count",
            "tp",
            "fp",
            "fn",
            "exact_match",
            "predicted_ids",
            "missed_ids",
            "extra_ids",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    exact_rate = totals["exact"] / max(1, totals["items"])
    print(
        "items",
        totals["items"],
        "precision",
        f"{precision:.3f}",
        "recall",
        f"{recall:.3f}",
        "exact",
        f"{exact_rate:.3f}",
        "tp/fp/fn",
        f"{totals['tp']}/{totals['fp']}/{totals['fn']}",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
