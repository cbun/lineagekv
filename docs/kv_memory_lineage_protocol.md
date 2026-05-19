# Memory Lineage Protocol for KV Cache Repair

This protocol defines the system contract needed to use KV cache repair without hidden benchmark stale labels.

## Memory Write Contract

Every persisted memory record must have:

- `id`: stable source ID used for citation and cache span lookup.
- `timestamp`: source-write time in ISO 8601 format.
- `text`: the exact memory text shown to the model.
- `metadata.memory_key`: the logical memory slot, such as `profile.editor.default`.
- Optional lineage:
  - `metadata.supersedes`: IDs this record replaces.
  - `metadata.superseded_by`: IDs that replace this record.
- Optional value metadata, such as `metadata.value`, for audits and deterministic product behavior.

The model prompt may use neutral rendered aliases such as `m000`, but the system keeps an alias map back to source IDs.

## Update Contract

When a memory is updated:

1. Write a new memory record with the same `memory_key`.
2. Assign a later `timestamp`.
3. Add `supersedes` from the new record to the replaced record IDs when lineage is available.
4. Add `superseded_by` on replaced records when the store supports reverse links.
5. Do not delete old records merely to make the prompt easy; old records remain available as historical context.

This creates a recoverable history: values may be obsolete, but the source IDs and write order remain inspectable.

## Cache Span Contract

When rendering the prompt:

1. Render each context block separately.
2. Record token ranges for each source block ID.
3. Record header/body subranges when available.
4. Preserve an alias map when neutral block IDs are rendered.

The KV repair algorithm edits cache vectors by source block token span, not by searching generated text after the fact.

## Currentness Resolution

The system may choose one of three resolvers:

- **Lineage resolver:** a block is stale if another present block supersedes it or if it lists a present `superseded_by` record.
- **Text+timestamp resolver:** parse the queried `memory_key`, find blocks whose visible text names that key, and treat the latest timestamp as current.
- **Model resolver:** ask a model to identify current/stale records, with abstention. Current local evidence shows this is not reliable enough to replace structured provenance.

The first two are programmatic. They are not LLM reasoning and do not read hidden `stale_ids`.

## Repair Contract

For each stale/current pair under the chosen resolver:

1. Prefill the full prompt and keep the KV cache.
2. Map stale source IDs to token spans.
3. Zero stale value vectors in the selected layer band.
4. Optionally boost current value vectors in the same layer band.
5. Preserve stale keys unless an experiment is explicitly testing key deletion.
6. Decode from the edited cache.

The current strongest assistant-memory policy is:

```text
zero_text_timestamp_conflict_values_layers_8_23_boost_current_factor_1_5
```

Qwen2.5-1.5B uses the proportional larger-model band:

```text
zero_text_timestamp_conflict_values_layers_12_27_boost_current_factor_1_5
```

## Provenance Repair Contract

Decoder-generated `evidence_ids` are not the only valid citation path. A deployed memory system already knows which source record is current under the resolver. For source-grounded answers, the system can replace generated `evidence_ids` with the resolved current source IDs while leaving the model answer unchanged.

This is a deterministic provenance layer. It should be reported separately from model-generated evidence accuracy.

## Negative Controls

A valid evaluation must include:

- Targeting audit: true positives / false positives / false negatives for stale-span detection.
- Label/status scrub: prove the resolver does not need hidden `stale_ids` or visible status labels.
- Lineage removal, when claiming lineage dependence: predictions should fail closed without lineage.
- Text/timestamp removal, when claiming text+timestamp dependence: predictions should fail closed without block text or timestamp ordering.
- Key deletion control: test stale key-only and K+V deletion to verify whether address structure is being damaged.

## Current Evidence

- Action-memory text+timestamp targeter: 64 TP / 0 FP / 0 FN.
- Messy multi-update targeter: 80 TP / 0 FP / 0 FN.
- Long-context d48-s4 targeter: 4 TP / 0 FP / 0 FN.
- Qwen2.5 action-memory key-only deletion collapses answer accuracy to 0.0000, supporting key/address preservation.
- Qwen2.5 and Qwen3 action-memory runs show stale-value repair can suppress stale use while preserving or nearly preserving answer behavior.
