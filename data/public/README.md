# Stale Memory Repair Benchmark v1

This folder contains the public-facing release benchmark for stale-memory KV-cache repair.

## Files

- `stale_memory_repair_benchmark_v1.jsonl` — 184 assistant-memory cases.
- `stale_memory_repair_benchmark_v1_metadata.json` — source counts and benchmark description.

## Composition

| Source | Cases | Description |
| --- | ---: | --- |
| `update_nonce_option` | 80 | realistic stale/current memory updates with nonce-coded answer options |
| `action` | 64 | direct/action assistant-memory queries with distractor memories |
| `messy_multi_update` | 40 | older/middle/current records with varied wording and noisy prose |

Total: 184 cases.

## Targeting Contract

Each row includes current evidence IDs, stale IDs, distractor IDs, timestamps, and memory metadata. The benchmark is designed to test both:

- explicit memory-lineage targeting, and
- text+timestamp currentness targeting without hidden stale labels.

Release audit:

- `results/kv/kv_public_release_targeting_audit.csv`
- lineage: 184/184 exact, 224 TP / 0 FP / 0 FN
- text+timestamp: 184/184 exact, 224 TP / 0 FP / 0 FN

## Reproduce

From the repo root:

```bash
bash scripts/run_kv_public_repro.sh
```

This rebuilds the public benchmark, targeter audit, runtime curve, and paired bootstrap statistics from existing checked-in/generated artifacts.
