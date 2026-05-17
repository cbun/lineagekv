#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.io import read_jsonl
from kv_value_ablation_probe import detected_lineage_stale_ids, detected_text_timestamp_stale_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the public release benchmark stale-memory targeters.")
    parser.add_argument("--dataset", default="data/public/stale_memory_repair_benchmark_v1.jsonl")
    parser.add_argument("--output", default="results/kv/kv_public_release_targeting_audit.csv")
    args = parser.parse_args()

    rows = []
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for item in read_jsonl(ROOT / args.dataset):
        gold = set(item.stale_ids)
        for detector, predicted_ids in [
            ("lineage", set(detected_lineage_stale_ids(item))),
            ("text_timestamp", set(detected_text_timestamp_stale_ids(item))),
        ]:
            tp = len(predicted_ids & gold)
            fp = len(predicted_ids - gold)
            fn = len(gold - predicted_ids)
            exact = predicted_ids == gold
            source = str(item.metadata.get("release_source", item.domain))
            key = f"{detector}:{source}"
            totals[key].update({"items": 1, "tp": tp, "fp": fp, "fn": fn, "exact": int(exact)})
            totals[detector].update({"items": 1, "tp": tp, "fp": fp, "fn": fn, "exact": int(exact)})
            rows.append(
                {
                    "item_id": item.id,
                    "release_source": source,
                    "domain": item.domain,
                    "detector": detector,
                    "gold_count": len(gold),
                    "predicted_count": len(predicted_ids),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "exact_match": exact,
                    "predicted_ids": " ".join(sorted(predicted_ids)),
                    "missed_ids": " ".join(sorted(gold - predicted_ids)),
                    "extra_ids": " ".join(sorted(predicted_ids - gold)),
                }
            )

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for key in sorted(totals):
        total = totals[key]
        precision = total["tp"] / max(1, total["tp"] + total["fp"])
        recall = total["tp"] / max(1, total["tp"] + total["fn"])
        exact = total["exact"] / max(1, total["items"])
        print(
            key,
            "items",
            total["items"],
            "precision",
            f"{precision:.3f}",
            "recall",
            f"{recall:.3f}",
            "exact",
            f"{exact:.3f}",
            "tp/fp/fn",
            f"{total['tp']}/{total['fp']}/{total['fn']}",
        )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
