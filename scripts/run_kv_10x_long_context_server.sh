#!/usr/bin/env bash
set -euo pipefail

# Reproducible long-context persistent-cache timing run for a GPU/server host.
# Expected use:
#   DEVICE=cuda DTYPE=bfloat16 MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/run_kv_10x_long_context_server.sh
#
# The script intentionally keeps defaults small enough for a first server smoke,
# while preserving the same benchmark/policy structure used by local CPU runs.

DATASET="${DATASET:-data/generated/agent_memory_long_context_test8_d48_s4.jsonl}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "$MODEL" | tr '/:.' '___')}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LIMIT="${LIMIT:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-40}"
OUT_PREFIX="${OUT_PREFIX:-results/kv/kv_${MODEL_TAG}_agent_memory_long_context_d48_s4_${DEVICE}}"

POLICIES="full_cache,drop_stale_prompt,zero_text_timestamp_conflict_values_layers_8_23_boost_current_factor_1_5"

python3 scripts/kv_value_ablation_probe.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --output "${OUT_PREFIX}.jsonl" \
  --limit "$LIMIT" \
  --distractor-ratio 0.95 \
  --variants long_context \
  --policies "$POLICIES" \
  --render-mode temporal_blind \
  --prompt-style plain \
  --parse-mode json_or_answer \
  --block-id-mode neutral \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --low-cpu-mem-usage

python3 scripts/summarize_kv_results.py \
  --results "${OUT_PREFIX}.jsonl" \
  --output "${OUT_PREFIX}_summary.csv" \
  --paired-policies "$POLICIES"

python3 scripts/summarize_kv_runtime_utility.py \
  --results "${OUT_PREFIX}.jsonl" \
  --output "${OUT_PREFIX}_runtime_utility.csv"

python3 scripts/summarize_kv_persistent_runtime.py \
  --results "${OUT_PREFIX}.jsonl" \
  --baseline drop_stale_prompt \
  --comparators zero_text_timestamp_conflict_values_layers_8_23_boost_current_factor_1_5 \
  --output "${OUT_PREFIX}_persistent_runtime.csv"

python3 scripts/summarize_kv_provenance_repair.py \
  --dataset "$DATASET" \
  --results "${OUT_PREFIX}.jsonl" \
  --output "${OUT_PREFIX}_provenance_repair.csv" \
  --summary-output "${OUT_PREFIX}_provenance_repair_summary.csv" \
  --detector text_timestamp
