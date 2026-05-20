# KV Result Artifacts

This directory contains compact artifacts for the public research release. Most
exploratory JSONL/CSV outputs are ignored by `.gitignore`.

Committed artifacts should satisfy at least one of these criteria:

- Required by `scripts/run_kv_public_repro.sh`.
- Referenced directly by `paper/stale_memory_kv_cache_repair.tex`.
- Needed to substantiate a headline table in the public technical report.

Full model-run dumps, failed probes, one-off timing files, and scratch outputs
are excluded unless they support a specific paper table or public reproduction
path.
