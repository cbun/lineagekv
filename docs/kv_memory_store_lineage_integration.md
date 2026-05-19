# Memory-Store Lineage Integration

Updated: 2026-05-20

Purpose: record the first real memory-store integration for stale-memory KV cache repair.

## Public Artifacts

- `src/context_rot/memory/ledger.py`
- `scripts/demo_memory_store_cache_repair.py`
- `scripts/run_memory_store_cache_repair_mlx_demo.py`
- `tests/test_memory_ledger.py`

The demo scripts can emit local JSON/SQLite artifacts when run, but those
one-case outputs are not required for the public no-model reproduction path.

## What It Demonstrates

The prototype uses a SQLite memory ledger rather than hidden benchmark labels.

On write/update:

1. A memory is stored with `memory_key`, value, text, timestamp, status, and metadata.
2. A newer write to the same `memory_key` automatically supersedes the previous current record.
3. The ledger creates explicit `memory_edges(stale_id, current_id)`.
4. Superseded records stay in the store instead of being deleted.

On render:

1. The ledger renders a cached memory prefix with neutral aliases such as `m000`.
2. It records token spans for every rendered memory block.
3. It keeps an alias-to-source-ID map for source provenance.

On cache repair:

1. The ledger emits a `CacheEditPlan` from memory lineage and render spans.
2. The plan identifies stale source spans, current source spans, layer range, and value-only edit intent.
3. It preserves keys by default and targets stale values.
4. It emits `provenance_current_ids` for deterministic source repair.

## Demo Scenario

Memory key: `calendar.weekly_review.location`

Historical record:

- `weekly_review_location_stale`: "Weekly review location is Room 3B."

Current record:

- `weekly_review_location_current`: "Weekly review location is Zoom."

User query:

- "Where should I join the weekly review now?"

The generated cache edit plan targets only the superseded Room 3B record and points provenance to the Zoom record. The stale record remains present in the ledger and benchmark item, so historical context is preserved. Runtime repair is a read-time cache policy, not destructive memory replacement.

## MLX Execution Demo

`scripts/run_memory_store_cache_repair_mlx_demo.py` consumes the ledger-generated plan with the real MLX tokenizer and applies it to `mlx-community/Qwen2.5-7B-Instruct-4bit`.

| Policy | Prefix Tokens | Correct | Evidence Correct | Stale Fact | Cache Edit |
| --- | ---: | --- | --- | --- | ---: |
| `full_cache` | 249 | true | true | false | 0.0 ms |
| `drop_stale_prompt` | 191 | true | true | false | 0.0 ms |
| `ledger_cache_repair` | 249 | true | true | false | 1.9 ms |

The important point is not the one-case accuracy, since this example is intentionally simple. The important point is the path: SQLite memory update lineage -> tokenizer render spans -> cache edit plan -> MLX KV cache edit -> graded model output.

## Scope

The prototype demonstrates the memory-store lineage path:

- Old values are not deleted.
- Update lineage is created programmatically.
- Render-time token spans are captured.
- Cache repair plans are generated from ledger state.
- Provenance repair is tied to current memory IDs.
- The MLX runner can consume the ledger-generated plan directly.

It is not a full deployed agent integration. A larger end-to-end agent workflow
would be needed to measure user-visible stale-action prevention in production.
