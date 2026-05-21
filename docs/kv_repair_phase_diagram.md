# KV Repair Phase Diagram

Updated: 2026-05-21

## Thesis

Stale-memory cache repair is not a single universal edit. The stronger scientific claim is a phase diagram: given a model family, prompt/interface regime, memory domain, and provenance requirements, different repair policies dominate.

The deployable method is therefore two-layered:

1. Cache-state repair suppresses stale influence after prefill.
2. Provenance repair replaces model-generated evidence IDs with current IDs from programmatic memory lineage.

## Current Phase Diagram

| Setting | Full-Cache Failure | Best Prompt Baseline | Best Cache Policy | Provenance Layer | Interpretation |
| --- | ---: | --- | --- | --- | --- |
| Qwen2.5 action memory, 32 cases | 9/32 wrong actions | prompt deletion prevents 5/9 | current-mean/value repair prevents 5/9 | restores evidence/joint to prompt-deletion level | strongest value/content repair case |
| Qwen3 action memory, 32 cases | 7/32 wrong actions | prompt deletion prevents 7/7 | zero+current boost prevents 7/7 but introduces 1 | useful but needs policy gating | positive in-family repair with regression risk |
| MLX Qwen2.5-7B long context, 32 cases | stale evidence on 25% full-cache rows | prompt deletion is clean but slow | refined value repair: 1.000 accuracy / 0.9688 evidence / 0.0313 stale | restores 1.000 joint | strongest systems result: 15.35x update speedup |
| Gemma 4 E4B easy action memory, 8 cases | none under proper Gemma turn format | all policies clean | no repair needed | not needed | prompt/interface boundary |
| Gemma 4 E4B adversarial user-preference, 8 cases | 3/8 wrong actions | prompt deletion: 1.000 / 1.000 / 0 stale | value-only, key-only, and K+V all 1.000 / 1.000 / 0 stale | already clean | broad raw-Gemma repair surface |
| Gemma 4 E4B adversarial policy-approval, 4 cases | 0 answer errors but 2/4 stale-evidence rows | prompt deletion: 1.000 / 1.000 / 0 stale | value-only: 1.000 accuracy / 0.750 evidence / 0.250 stale; key-only and K+V: 1.000 / 0.500 / 0.500 | restores all cache policies to 1.000 repaired joint | domain/query boundary; evidence repair matters |
| Gemma 4 E4B adversarial CRM follow-up, 4 cases | 2/4 wrong actions | prompt deletion: 1.000 / 1.000 / 0 stale | value-only, key-only, and K+V all 1.000 / 1.000 / 0 stale | already clean | operational wrong-action prevention |
| Gemma 4 E4B adversarial deployment-state, 4 cases | 0 answer errors but 2/4 exact-evidence misses | prompt deletion: 1.000 / 1.000 / 0 stale | key-only and K+V: 1.000 / 1.000 / 0; value-only: 1.000 / 0.750 / 0 | restores all cache policies to 1.000 repaired joint | citation-precision boundary |
| Gemma 4 E4B adversarial tool-trace, 4 cases | 0 answer errors but 1/4 stale/contradictory trace row | prompt deletion: 1.000 / 1.000 / 0 stale | value-only, key-only, and K+V all 1.000 / 1.000 / 0 stale | already clean | tool execution provenance boundary |
| Gemma 4 E4B adversarial task-state, 4 cases | 1/4 wrong/stale work-state rows | prompt deletion: 1.000 / 1.000 / 0 stale | value-only, key-only, and K+V all 1.000 / 1.000 / 0 stale | already clean | project/task memory boundary |
| Gemma 4 E4B adversarial knowledge-base, 4 cases | none under proper Gemma turn format | all policies clean | no repair needed | not needed | prompt/interface boundary on trusted-answer facts |
| Gemma 4 E4B adversarial cross-domain, 32 cases | 9/32 wrong/stale actions | prompt deletion: 1.000 / 1.000 / 0 stale | value-only, key-only, and K+V all reach 1.000 accuracy; evidence/stale differ by domain | restores all cache policies to 1.000 repaired joint | raw-Gemma transfer, domain-specific repair surface |

Metrics in slash form are answer accuracy / model-generated evidence accuracy / stale-use rate.

## Emerging Rules

- If proper prompting already resolves the conflict, cache repair has no visible upside and can only introduce risk.
- If stale and current content are both plausible but answer extraction is easy, cache repair may fix answers while still leaking stale evidence IDs.
- Value-only repair is currently the safest answer-level cache edit for raw Gemma 4 E4B, but it is not perfect without provenance repair.
- Key-only and K+V can work in user-preference and CRM conflicts, leak stale provenance in policy-approval conflicts, and improve exact evidence in deployment-state conflicts.
- Qwen2.5 remains the clearest address/content dissociation: stale values can be suppressed while useful addressing is preserved.
- The production contract should choose policies per model/domain and always keep memory lineage for evidence repair.

## Interpretation

The result should not be framed as a universal transformer law. The supported
claim is:

> Reused long-context caches have a stale-memory failure mode; memory lineage makes those stale spans targetable; cache repair plus provenance repair can fix behavior without re-prefilling; and the best intervention follows a model/domain phase diagram.

The next decisive experiments are:

1. Add another raw 7B+ family or a standard full-precision CUDA replication.
2. Test less-generated or production-like memory logs where lineage comes from real writes rather than synthetic generated traces.
3. Convert the phase diagram into a policy selector: use prompt deletion, value-only repair, K+V repair, or cache repair plus provenance repair depending on measured model/domain behavior.
