# KV Address/Content Causal Map

Neutral-ID K/V matrices test whether stale-cache repair comes from removing value content while preserving cache addresses, or from destructive key+value deletion.

Generated Qwen tables: `results/kv/kv_address_content_causal_map_policy.csv` and `results/kv/kv_address_content_causal_map_kv_vs_value.csv`.
Generated in-family transfer tables: `results/kv/kv_address_content_in_family_transfer_policy.csv` and `results/kv/kv_address_content_in_family_transfer_kv_vs_value.csv`.
Generated cross-family boundary tables: `results/kv/kv_address_content_boundary_policy.csv` and `results/kv/kv_address_content_boundary_kv_vs_value.csv`.

## Main Result

Across Qwen2.5-0.5B and Qwen2.5-1.5B, K+V edits suppress stale use more aggressively, but value-only edits preserve evidence/currentness substantially better. This supports an address/content split in Qwen2.5: stale values carry harmful content, while keys preserve useful retrieval/citation structure.

## Qwen2.5-0.5B Layers 8-23

| Edit | Span | Accuracy | Evidence | Stale Use | Clean |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | full_context | 0.362 | 0.450 | 0.650 | 0.188 |
| values | full_block | 0.412 | 0.650 | 0.375 | 0.400 |
| keys | full_block | 0.087 | 0.000 | 0.000 | 0.087 |
| keys_values | full_block | 0.287 | 0.225 | 0.263 | 0.275 |
| values | header | 0.463 | 0.562 | 0.350 | 0.450 |
| keys | header | 0.013 | 0.000 | 0.000 | 0.013 |
| keys_values | header | 0.312 | 0.250 | 0.275 | 0.300 |
| values | body | 0.325 | 0.450 | 0.675 | 0.163 |
| keys | body | 0.188 | 0.037 | 0.250 | 0.125 |
| keys_values | body | 0.362 | 0.412 | 0.725 | 0.138 |

| Span | K+V - Value Evidence Delta | 95% CI | K+V - Value Stale Delta | 95% CI |
| --- | ---: | --- | ---: | --- |
| full_block | -0.425 | [-0.550, -0.300] | -0.113 | [-0.250, 0.025] |
| header | -0.312 | [-0.450, -0.175] | -0.075 | [-0.200, 0.062] |
| body | -0.037 | [-0.163, 0.075] | 0.050 | [-0.050, 0.150] |

## Qwen2.5-1.5B Layers 12-27

| Edit | Span | Accuracy | Evidence | Stale Use | Clean |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | full_context | 0.375 | 0.000 | 1.000 | 0.000 |
| values | full_block | 0.425 | 0.950 | 0.425 | 0.400 |
| keys | full_block | 0.150 | 0.025 | 0.175 | 0.125 |
| keys_values | full_block | 0.300 | 0.500 | 0.175 | 0.300 |
| values | header | 0.325 | 0.925 | 0.575 | 0.300 |
| keys | header | 0.250 | 0.025 | 0.350 | 0.225 |
| keys_values | header | 0.325 | 0.600 | 0.325 | 0.300 |
| values | body | 0.450 | 0.000 | 1.000 | 0.000 |
| keys | body | 0.275 | 0.400 | 0.625 | 0.100 |
| keys_values | body | 0.425 | 0.000 | 1.000 | 0.000 |

| Span | K+V - Value Evidence Delta | 95% CI | K+V - Value Stale Delta | 95% CI |
| --- | ---: | --- | ---: | --- |
| full_block | -0.450 | [-0.600, -0.300] | -0.250 | [-0.425, -0.100] |
| header | -0.325 | [-0.475, -0.175] | -0.250 | [-0.400, -0.100] |
| body | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |

## In-Family Transfer Boundary

Qwen3-0.6B reproduces the stale-cleanup and header/body separation pattern, but not the Qwen2.5 K/V address/content evidence split. In this 20-case transfer, full-block value-only, key-only, and K+V edits are behaviorally identical, and key-only deletion does not collapse evidence.

### Qwen3-0.6B Layers 9-27

| Edit | Span | Accuracy | Evidence | Stale Use | Clean |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | full_context | 0.550 | 0.350 | 0.800 | 0.200 |
| values | full_block | 0.500 | 1.000 | 0.550 | 0.450 |
| keys | full_block | 0.500 | 1.000 | 0.550 | 0.450 |
| keys_values | full_block | 0.500 | 1.000 | 0.550 | 0.450 |
| values | header | 0.500 | 1.000 | 0.500 | 0.500 |
| keys | header | 0.450 | 1.000 | 0.500 | 0.450 |
| keys_values | header | 0.500 | 1.000 | 0.500 | 0.500 |
| values | body | 0.450 | 0.450 | 0.800 | 0.200 |
| keys | body | 0.450 | 0.500 | 0.800 | 0.200 |
| keys_values | body | 0.450 | 0.450 | 0.800 | 0.200 |

| Span | K+V - Value Evidence Delta | 95% CI | K+V - Value Stale Delta | 95% CI |
| --- | ---: | --- | ---: | --- |
| full_block | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| header | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| body | 0.000 | [-0.150, 0.150] | 0.000 | [0.000, 0.000] |

## Cross-Family Boundary

SmolLM2-360M keeps the broad stale-cleanup pattern but does not reproduce the Qwen address/content evidence split. Evidence is weak across policies, and K+V full-block suppression increases evidence relative to value-only while reducing stale use more aggressively.

### SmolLM2-360M Layers 0-31

| Edit | Span | Accuracy | Evidence | Stale Use | Clean |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | full_context | 0.275 | 0.050 | 0.450 | 0.188 |
| values | full_block | 0.350 | 0.000 | 0.325 | 0.338 |
| keys_values | full_block | 0.250 | 0.125 | 0.212 | 0.200 |
| values | header | 0.287 | 0.000 | 0.325 | 0.250 |
| keys_values | header | 0.250 | 0.087 | 0.300 | 0.188 |
| values | body | 0.312 | 0.013 | 0.425 | 0.250 |
| keys_values | body | 0.338 | 0.025 | 0.412 | 0.287 |

| Span | K+V - Value Evidence Delta | 95% CI | K+V - Value Stale Delta | 95% CI |
| --- | ---: | --- | ---: | --- |
| full_block | 0.125 | [0.062, 0.200] | -0.113 | [-0.225, 0.000] |
| header | 0.087 | [0.037, 0.150] | -0.025 | [-0.138, 0.087] |
| body | 0.013 | [-0.025, 0.050] | -0.013 | [-0.125, 0.087] |

## Interpretation

- Full-block value-only suppression is the best mechanism-shaped repair in both Qwen sizes: it restores evidence/currentness while keeping answer behavior usable.
- Full-block K+V suppression reduces stale use more, but it consistently loses evidence accuracy relative to value-only suppression.
- Qwen2.5 key-only suppression is even more destructive than K+V for the full-block/header spans, directly supporting the interpretation that stale keys carry useful address structure.
- Header-only value suppression carries most evidence/currentness repair in Qwen2.5-1.5B, while body-only edits affect raw answer behavior without repairing stale evidence.
- Qwen3-0.6B is an in-family boundary: the stale-block/header repair transfers, but full-block value-only, key-only, and K+V edits collapse to the same behavior on this sample.
- SmolLM2-360M is a boundary case: value-only stale suppression still improves answer/clean tradeoffs, but the K/V evidence dissociation is not universal at this scale/family point.
- The current address/content causal claim is strongest for Qwen2.5 neutral-ID contexts; broader-family and newer-Qwen claims should be scoped to value-only stale cleanup unless larger models replicate the K/V split.
