from __future__ import annotations

from context_rot.datasets.schema import BenchmarkItem, CompressionResult


INSTRUCTION = """Answer using only the provided context bundle. Prefer current, high-trust, non-superseded facts. If facts conflict and the conflict cannot be resolved, say so. The answer field must contain the requested value or decision, not a context block ID. Put context block IDs only in evidence_ids. Return JSON only with keys: answer, evidence_ids, used_stale_fact, confidence, abstain, reasoning_summary."""


def render_prompt(item: BenchmarkItem, compression: CompressionResult) -> str:
    extra_instruction = ""
    if item.domain == "external_locomo":
        extra_instruction = (
            "\nFor LoCoMo dialogue questions, answer as a standalone value. "
            "Resolve relative dates such as yesterday, last week, or last year using the context timestamps when possible. "
            "Evidence IDs must be quoted JSON strings exactly matching dialogue IDs such as \"D1:3\". "
            "Use the smallest sufficient set of evidence IDs, usually 1-3 IDs, and cite the dialogue turn(s) that directly support the answer rather than every related turn."
        )
    if item.domain == "external_longmemeval_s":
        extra_instruction = (
            "\nFor LongMemEval-S questions, cite the focused context block ID when the answer comes from focused memory. "
            "Evidence IDs must be quoted JSON strings exactly matching the allowed evidence IDs."
        )
    allowed_evidence_ids = [block.id for block in compression.bundle.context_blocks]
    return (
        f"{INSTRUCTION}{extra_instruction}\n\n"
        f"Context bundle strategy: {compression.strategy}\n\n"
        f"{compression.bundle.rendered_context}\n\n"
        f"Allowed evidence_ids: {allowed_evidence_ids}\n"
        "If you answer using context, include at least one matching ID from Allowed evidence_ids in evidence_ids.\n\n"
        f"Question: {item.query}\n\n"
        "JSON:"
    )
