# KV Lineage Targeting Validity

Date: 2026-05-19

## Question

The main Qwen2.5 causal-map policies use names like `zero_lineage_conflict_values_layers_8_23`.

This audit checks what those policies actually depend on:

- Are they using benchmark `stale_ids` labels?
- Are they using block `status` labels?
- Or are they using explicit update-lineage metadata that could exist in a memory ledger?

## Mechanism

The relevant policy path is:

1. `apply_policy(...)`
2. `detected_lineage_stale_ids(item)`
3. `metadata["supersedes"]` and `metadata["superseded_by"]`

`detected_lineage_stale_ids` does not read `item.stale_ids` and does not read `block.status`. It marks a block stale if another present block says it supersedes that block, or if the block says it was superseded by another present block.

## Audit Command

```bash
python3 scripts/audit_lineage_targeting_ablations.py \
  --dataset data/generated/git_doc_nonce_option_audited_history_four_repo_probe80.jsonl \
  --output results/kv/kv_final_nonce_option_lineage_targeting_ablations.csv \
  --distractor-ratio 0.0 \
  --variants current_first,stale_first
```

Standard detector audit:

```bash
python3 scripts/audit_kv_detector.py \
  --dataset data/generated/git_doc_nonce_option_audited_history_four_repo_probe80.jsonl \
  --output results/kv/kv_conflict_detector_git_doc_nonce_option_audited_history_four_repo_probe80_lineage_audit.csv \
  --distractor-ratio 0.0 \
  --variants current_first,stale_first \
  --detector-mode lineage
```

## Result

Primary lineage detector audit on the 80-row four-repo nonce-option probe:

| Items | Precision | Recall | Exact | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | 1.000 | 1.000 | 1.000 | 80 | 0 | 0 |

Targeting ablation audit:

| Ablation | Exact / Empty Rate |
| --- | ---: |
| `stale_ids` blanked | 80/80 exact |
| block `status` set to `unknown` | 80/80 exact |
| both `stale_ids` blanked and `status` set to `unknown` | 80/80 exact |
| lineage metadata removed | 80/80 empty predictions |
| lineage metadata removed | 0/80 exact |

## Interpretation

The final neutral-ID Qwen2.5 causal-map policies are not raw `stale_ids` oracle policies and are not status-label policies. They are ledger-lineage policies: they require explicit `supersedes` / `superseded_by` edges.

This narrows the remaining deployment limitation:

- **Resolved:** the main causal-map intervention does not require hidden benchmark stale labels when a memory ledger supplies update lineage.
- **Not resolved:** raw text alone cannot reliably infer currentness in all adversarial cases. Prior structural-detector audits show that count/timestamp heuristics hit identifiability limits without source semantics, edit lineage, ledger state, or a model-based currentness resolver.

For the paper, the honest phrasing is:

> The intervention is oracle-free with respect to benchmark stale labels under a memory-ledger contract, but it is not yet a raw-text stale discovery method.
