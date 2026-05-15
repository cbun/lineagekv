# Dataset Notes

The public repo includes only compact generated artifacts needed for
reproducibility. Raw external datasets, processed external conversions, and
cloned repositories are intentionally excluded.

## Public Files

| Path | Rows | Description |
| --- | ---: | --- |
| `data/public/stale_memory_repair_benchmark_v1.jsonl` | 184 | Generated assistant-memory stale/current cases |
| `data/public/stale_memory_universality_traces_v1.jsonl` | 264 | Mixed assistant-memory and public Git-history traces |
| `data/generated/agent_memory_update_nonce_option_probe80.jsonl` | 80 | Source rows for nonce-coded update cases |
| `data/generated/agent_memory_action_probe64.jsonl` | 64 | Source rows for action-memory cases |
| `data/generated/agent_memory_messy_probe40.jsonl` | 40 | Source rows for messy multi-update cases |
| `data/generated/git_doc_nonce_option_audited_history_four_repo_probe80.jsonl` | 80 | Public Git-history nonce-option probe |
| `data/generated/agent_memory_long_context_test32_d48_s4.jsonl` | 32 | Long-context runtime benchmark source |
| `data/generated/agent_memory_long_context_test2_d236_s4.jsonl` | 2 | About-32k token runtime benchmark source |

## External Datasets Used During Development

The private development workspace also used LongMemEval-S, LoCoMo, NoLiMa, and
public Git histories for exploratory comparisons. Those raw sources are not
committed here. The release focuses on generated stale-memory cases and compact
public traces that can be rebuilt by:

```bash
bash scripts/run_kv_public_repro.sh
```
