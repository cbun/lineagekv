#!/usr/bin/env bash
set -euo pipefail

python3 scripts/build_public_release_dataset.py
python3 scripts/build_universality_trace_benchmark.py
python3 scripts/audit_public_release_dataset.py
python3 scripts/build_kv_runtime_curve.py

python3 scripts/paired_kv_stats.py \
  --results results/kv/kv_mlx_qwen25_7b_4bit_agent_memory_long_context_test32_refined_policies_regraded.jsonl \
  --baseline full_cache \
  --comparators drop_stale_prompt,zero_text_timestamp_conflict_values_layers_8_23_boost_current_factor_1_0 \
  --output results/kv/kv_public_7b32_paired_stats.csv \
  --bootstrap-samples 2000

python3 scripts/paired_kv_stats.py \
  --results results/kv/kv_qwen3_0_6b_agent_memory_action_test32_evidence_full.jsonl \
  --baseline full_cache \
  --comparators drop_stale_prompt,zero_text_timestamp_conflict_values_layers_8_23_boost_current_factor_1_5 \
  --output results/kv/kv_public_qwen3_action32_paired_stats.csv \
  --bootstrap-samples 2000

python3 scripts/paired_kv_stats.py \
  --results results/kv/kv_qwen25_0_5b_agent_memory_text_timestamp_decile1_full80.jsonl \
  --baseline full_cache \
  --comparators drop_stale_prompt,zero_text_timestamp_conflict_values_layers_8_23_span_decile_1 \
  --output results/kv/kv_public_qwen25_agent_memory80_paired_stats.csv \
  --bootstrap-samples 2000

echo "Public release artifacts regenerated."
