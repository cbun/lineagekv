#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl
from context_rot.datasets.schema import BenchmarkItem, ContextBlock
from context_rot.eval.graders import grade_output, parse_model_json


OUTPUT_INSTRUCTION = (
    'Return JSON only with keys: "answer", "evidence_ids", "used_stale_fact", '
    '"confidence", "abstain", "reasoning_summary". Put context block IDs only in evidence_ids. '
    "Keep reasoning_summary under 8 words."
)


@dataclass
class EncodedCase:
    prefix_ids: torch.Tensor
    suffix_ids: torch.Tensor
    block_ranges: dict[str, tuple[int, int]]
    suffix_ranges: dict[str, tuple[int, int]] | None = None
    block_header_ranges: dict[str, tuple[int, int]] | None = None
    block_body_ranges: dict[str, tuple[int, int]] | None = None
    block_aliases: dict[str, str] | None = None
    alias_to_block_id: dict[str, str] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe KV value ablation on stale context-token ranges.")
    parser.add_argument("--dataset", default="data/generated/context_rot_v1.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="results/kv/kv_value_ablation_probe.jsonl")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--offsets",
        default="",
        help="Optional comma-separated offsets; when set, take --limit matching items from each offset.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--distractor-ratio", type=float, default=0.9)
    parser.add_argument(
        "--render-mode",
        choices=["ledger", "temporal_blind"],
        default="ledger",
        help=(
            "ledger preserves status/trust headers and stale instructions; temporal_blind hides status/trust "
            "and neutralizes current/superseded wording for value-bearing blocks."
        ),
    )
    parser.add_argument(
        "--variants",
        default="wrong_duplicate_10x,current_late,current_early_stale_late,stale_repeated_after",
        help="Comma-separated variant names to include.",
    )
    parser.add_argument(
        "--policies",
        default="full_cache,zero_stale_values,zero_stale_keys_values,zero_non_gold_values",
        help="Comma-separated KV edit policies.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["plain", "chat", "gemma4"],
        default="plain",
        help=(
            "plain uses the historical raw prompt; chat wraps the same content in the tokenizer chat template; "
            "gemma4 uses the text-only Gemma 4 turn format."
        ),
    )
    parser.add_argument(
        "--parse-mode",
        choices=["json", "json_or_answer"],
        default="json",
        help="json requires structured output; json_or_answer falls back to a unique nonce/code answer mention.",
    )
    parser.add_argument(
        "--block-id-mode",
        choices=["original", "neutral"],
        default="original",
        help="original renders dataset block IDs; neutral renders m000-style aliases and maps evidence back.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="",
        help="Optional transformers attention implementation override, e.g. eager for attention-gated policies.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model parameter dtype. Defaults to historical float32.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for model and inputs: cpu, mps, cuda, or auto.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loading.",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="Pass low_cpu_mem_usage=True to model loading.",
    )
    parser.add_argument(
        "--phi-compat-shims",
        action="store_true",
        help="Install local compatibility shims needed by cached microsoft/Phi-4-mini-instruct remote code.",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.phi_compat_shims:
        install_phi_compat_shims()
    device = resolve_device(args.device)
    variants = {value for value in args.variants.split(",") if value}
    matching_items = [
        item
        for item in read_jsonl(ROOT / args.dataset)
        if float(item.metadata.get("distractor_ratio", -1.0)) == args.distractor_ratio
        and str(item.metadata.get("variant", "")) in variants
    ]
    if args.offsets:
        items = []
        seen_ids: set[str] = set()
        for offset_text in args.offsets.split(","):
            if not offset_text:
                continue
            offset = int(offset_text)
            for item in matching_items[offset : offset + args.limit]:
                if item.id not in seen_ids:
                    items.append(item)
                    seen_ids.add(item.id)
    else:
        items = matching_items[args.offset : args.offset + args.limit]
    if not items:
        raise SystemExit("No matching items found.")

    tokenizer_kwargs: dict[str, Any] = {"local_files_only": True}
    model_kwargs: dict[str, Any] = {"local_files_only": True, "dtype": parse_torch_dtype(args.dtype)}
    if args.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
        model_kwargs["trust_remote_code"] = True
    if args.low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.phi_compat_shims:
        patched_buffers = patch_meta_rope_buffers(model)
        if patched_buffers:
            print(f"patched {patched_buffers} meta rotary buffers", file=sys.stderr)
    if device.type != "cpu":
        model.to(device)
    model.eval()

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    policies = [value for value in args.policies.split(",") if value]
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            encoded = encode_case(
                tokenizer,
                item,
                render_mode=args.render_mode,
                prompt_style=args.prompt_style,
                block_id_mode=args.block_id_mode,
            )
            encoded = move_encoded_case(encoded, device)
            encodings = {"full_prefix": encoded}
            prefills: dict[str, Any] = {}
            for policy in policies:
                excluded = prompt_policy_excluded_block_ids(policy, item)
                if excluded is None:
                    continue
                policy_item = prompt_policy_item(policy, item)
                encodings[policy] = encode_case(
                    tokenizer,
                    policy_item,
                    exclude_block_ids=excluded,
                    render_mode=args.render_mode,
                    prompt_style=args.prompt_style,
                    block_id_mode=args.block_id_mode,
                )
                encodings[policy] = move_encoded_case(encodings[policy], device)
            with torch.no_grad():
                prefill_ms: dict[str, float] = {}
                for key, encoded_case in encodings.items():
                    sync_torch_device(device)
                    started = time.perf_counter()
                    prefills[key] = model(input_ids=encoded_case.prefix_ids, use_cache=True)
                    sync_torch_device(device)
                    prefill_ms[key] = (time.perf_counter() - started) * 1000
            attention_token_scores: dict[str, torch.Tensor] = {}
            for policy in policies:
                encoding_key = policy if policy in encodings and policy != "full_cache" else "full_prefix"
                policy_encoded = encodings[encoding_key]
                sync_torch_device(device)
                started = time.perf_counter()
                cache = clone_cache(prefills[encoding_key].past_key_values)
                sync_torch_device(device)
                cache_clone_ms = (time.perf_counter() - started) * 1000
                edit_ms = 0.0
                if prompt_policy_excluded_block_ids(policy, item) is None:
                    layer_slice = attention_policy_layer_slice(policy)
                    policy_attention_scores = None
                    if layer_slice:
                        if layer_slice not in attention_token_scores:
                            attention_token_scores[layer_slice] = query_attention_token_scores(
                                model=model,
                                cache=clone_cache(prefills[encoding_key].past_key_values),
                                prefix_len=policy_encoded.prefix_ids.shape[1],
                                suffix_ids=policy_encoded.suffix_ids,
                                suffix_ranges=policy_encoded.suffix_ranges or {},
                                layer_slice=layer_slice,
                            )
                        policy_attention_scores = attention_token_scores[layer_slice]
                    sync_torch_device(device)
                    started = time.perf_counter()
                    apply_policy(cache, policy_encoded, item, policy, attention_token_scores=policy_attention_scores)
                    sync_torch_device(device)
                    edit_ms = (time.perf_counter() - started) * 1000
                sync_torch_device(device)
                started = time.perf_counter()
                raw_output = generate_from_cache(
                    model=model,
                    tokenizer=tokenizer,
                    cache=cache,
                    prefix_len=policy_encoded.prefix_ids.shape[1],
                    suffix_ids=policy_encoded.suffix_ids,
                    max_new_tokens=args.max_new_tokens,
                )
                sync_torch_device(device)
                latency_ms = (time.perf_counter() - started) * 1000
                parsed = parse_probe_output(raw_output, item, parse_mode=args.parse_mode)
                parsed = remap_parsed_evidence_ids(parsed, policy_encoded.alias_to_block_id or {})
                grade = grade_output(item, policy, f"kv:{args.model}", parsed)
                row = {
                    "item_id": item.id,
                    "domain": item.domain,
                    "question_type": item.question_type,
                    "variant": item.metadata.get("variant"),
                    "distractor_ratio": item.metadata.get("distractor_ratio"),
                    "render_mode": args.render_mode,
                    "prompt_style": args.prompt_style,
                    "parse_mode": args.parse_mode,
                    "block_id_mode": args.block_id_mode,
                    "block_aliases": policy_encoded.block_aliases or {},
                    "alias_to_block_id": policy_encoded.alias_to_block_id or {},
                    "policy": policy,
                    "model_id": f"kv:{args.model}",
                    "prefix_tokens": int(policy_encoded.prefix_ids.shape[1]),
                    "full_prefix_tokens": int(encoded.prefix_ids.shape[1]),
                    "suffix_tokens": int(policy_encoded.suffix_ids.shape[1]),
                    "stale_token_count": token_count_for_ids(encoded, item.stale_ids),
                    "gold_token_count": token_count_for_ids(encoded, item.gold_evidence_ids),
                    "latency_ms": latency_ms,
                    "prefill_ms": prefill_ms[encoding_key],
                    "full_prefill_ms": prefill_ms["full_prefix"],
                    "cache_clone_ms": cache_clone_ms,
                    "cache_edit_ms": edit_ms,
                    "decode_ms": latency_ms,
                    "raw_output": raw_output,
                    "parsed_output": parsed,
                    **grade.model_dump(),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    print(f"wrote {output_path}")


def encode_case(
    tokenizer: Any,
    item: BenchmarkItem,
    exclude_block_ids: set[str] | None = None,
    render_mode: str = "ledger",
    prompt_style: str = "plain",
    block_id_mode: str = "original",
) -> EncodedCase:
    exclude_block_ids = exclude_block_ids or set()
    if render_mode == "ledger":
        prefix_start = (
            "You answer questions from a cached agent memory prefix.\n"
            "Prefer current, high-trust, non-superseded facts. Ignore stale or superseded facts.\n"
            "Context blocks:\n"
        )
    elif render_mode == "temporal_blind":
        prefix_start = (
            "You answer questions from a cached agent memory prefix.\n"
            "Resolve conflicts using the memory contents and timestamps.\n"
            "Context blocks:\n"
        )
    else:
        raise KeyError(render_mode)
    prefix_ids = prompt_lead_ids(tokenizer, prompt_style)
    prefix_ids.extend(tokenizer(prefix_start, add_special_tokens=False).input_ids)
    block_ranges: dict[str, tuple[int, int]] = {}
    block_header_ranges: dict[str, tuple[int, int]] = {}
    block_body_ranges: dict[str, tuple[int, int]] = {}
    block_aliases = block_id_aliases(item, block_id_mode)
    visible_block_aliases: dict[str, str] = {}
    for block in item.context_blocks:
        if block.id in exclude_block_ids:
            continue
        visible_block_aliases[block.id] = block_aliases[block.id]
        block_header, block_body, block_suffix = render_block_parts(block, render_mode, rendered_block_id=block_aliases[block.id])
        header_ids = tokenizer(block_header, add_special_tokens=False).input_ids
        body_ids = tokenizer(block_body, add_special_tokens=False).input_ids
        block_suffix_ids = tokenizer(block_suffix, add_special_tokens=False).input_ids
        start = len(prefix_ids)
        prefix_ids.extend(header_ids)
        header_end = len(prefix_ids)
        prefix_ids.extend(body_ids)
        body_end = len(prefix_ids)
        prefix_ids.extend(block_suffix_ids)
        block_ranges[block.id] = (start, len(prefix_ids))
        block_header_ranges[block.id] = (start, header_end)
        block_body_ranges[block.id] = (header_end, body_end)
    suffix_ids: list[int] = []
    suffix_ranges: dict[str, tuple[int, int]] = {}
    question_lead_ids = tokenizer("\nQuestion: ", add_special_tokens=False).input_ids
    suffix_ids.extend(question_lead_ids)
    query_ids = tokenizer(item.query, add_special_tokens=False).input_ids
    query_start = len(suffix_ids)
    suffix_ids.extend(query_ids)
    suffix_ranges["query"] = (query_start, len(suffix_ids))
    suffix_ids.extend(tokenizer(f"\n{OUTPUT_INSTRUCTION}\nJSON:", add_special_tokens=False).input_ids)
    suffix_ids.extend(prompt_tail_ids(tokenizer, prompt_style))
    return EncodedCase(
        prefix_ids=torch.tensor([prefix_ids], dtype=torch.long),
        suffix_ids=torch.tensor([suffix_ids], dtype=torch.long),
        block_ranges=block_ranges,
        suffix_ranges=suffix_ranges,
        block_header_ranges=block_header_ranges,
        block_body_ranges=block_body_ranges,
        block_aliases=visible_block_aliases,
        alias_to_block_id={alias: block_id for block_id, alias in visible_block_aliases.items()},
    )


def block_id_aliases(item: BenchmarkItem, block_id_mode: str) -> dict[str, str]:
    if block_id_mode == "original":
        return {block.id: block.id for block in item.context_blocks}
    if block_id_mode == "neutral":
        return {block.id: f"m{idx:03d}" for idx, block in enumerate(item.context_blocks)}
    raise KeyError(block_id_mode)


def prompt_lead_ids(tokenizer: Any, prompt_style: str) -> list[int]:
    if prompt_style == "plain":
        return []
    if prompt_style == "gemma4":
        return tokenizer(gemma4_template_lead(), add_special_tokens=False).input_ids
    lead, _ = chat_template_split(tokenizer)
    return tokenizer(lead, add_special_tokens=False).input_ids


def prompt_tail_ids(tokenizer: Any, prompt_style: str) -> list[int]:
    if prompt_style == "plain":
        return []
    if prompt_style == "gemma4":
        return tokenizer(gemma4_template_tail(), add_special_tokens=False).input_ids
    _, tail = chat_template_split(tokenizer)
    return tokenizer(tail, add_special_tokens=False).input_ids


def gemma4_template_lead() -> str:
    return (
        "<bos><|turn>system\n"
        "You are a precise assistant. Return only the requested JSON object. "
        "Do not include a thought channel or extra prose."
        "<turn|>\n"
        "<|turn>user\n"
    )


def gemma4_template_tail() -> str:
    return "<turn|>\n<|turn>model\n"


def chat_template_split(tokenizer: Any) -> tuple[str, str]:
    marker = "__KV_CACHE_SUFFIX_SPLIT_MARKER__"
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("prompt_style=chat requires a tokenizer chat_template")
    chat_kwargs = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    if "enable_thinking" in str(tokenizer.chat_template):
        chat_kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        **chat_kwargs,
    )
    if marker not in rendered:
        raise ValueError("tokenizer chat_template did not preserve split marker")
    lead, tail = rendered.split(marker, 1)
    return lead, tail


def render_block(block: Any, render_mode: str) -> str:
    header, body, suffix = render_block_parts(block, render_mode, rendered_block_id=block.id)
    return f"{header}{body}{suffix}"


def render_block_parts(block: Any, render_mode: str, rendered_block_id: str) -> tuple[str, str, str]:
    if render_mode == "ledger":
        return (
            f"\n[block_id={rendered_block_id}; timestamp={block.timestamp}; type={block.type}; "
            f"trust={block.trust}; status={block.status}]\n",
            block.text,
            "\n",
        )
    if render_mode == "temporal_blind":
        text = block.text
        entity = block.metadata.get("entity")
        prop = block.metadata.get("property")
        value = block.metadata.get("value")
        if entity and prop and value:
            text = f"Note for {entity}: {prop} was '{value}'."
        return f"\n[block_id={rendered_block_id}; timestamp={block.timestamp}; type={block.type}]\n", text, "\n"
    raise KeyError(render_mode)


def parse_probe_output(raw_output: str, item: BenchmarkItem, parse_mode: str = "json") -> dict[str, Any]:
    parsed = parse_model_json(raw_output)
    if parse_mode == "json":
        return parsed
    if parse_mode == "json_or_answer":
        answer = extract_unique_answer_mention(raw_output, item)
        if answer and isinstance(parsed, dict) and not usable_answer(parsed.get("answer")):
            repaired = dict(parsed)
            repaired["answer"] = answer
            repaired.setdefault("evidence_ids", [])
            repaired.setdefault("used_stale_fact", False)
            return repaired
        salvaged_whole_text = (
            isinstance(parsed, dict)
            and str(parsed.get("answer", "")).strip() == raw_output.strip()
            and not parsed.get("evidence_ids")
        )
        if answer and (not parsed or salvaged_whole_text):
            return {"answer": answer, "evidence_ids": [], "used_stale_fact": False}
        if salvaged_whole_text:
            return {}
        return parsed
    raise KeyError(parse_mode)


def usable_answer(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith("{") or text.startswith("["):
        return False
    return True


def remap_parsed_evidence_ids(parsed: dict[str, Any], alias_to_block_id: dict[str, str]) -> dict[str, Any]:
    if not alias_to_block_id or not isinstance(parsed, dict) or "evidence_ids" not in parsed:
        return parsed
    remapped = dict(parsed)
    evidence_ids = parsed.get("evidence_ids")
    if isinstance(evidence_ids, list):
        remapped["evidence_ids"] = [
            alias_to_block_id.get(normalize_evidence_alias(evidence_id, alias_to_block_id), str(evidence_id))
            for evidence_id in evidence_ids
        ]
    return remapped


def normalize_evidence_alias(evidence_id: Any, alias_to_block_id: dict[str, str]) -> str:
    raw = str(evidence_id).strip()
    if raw in alias_to_block_id:
        return raw
    if raw.isdigit():
        candidate = f"m{int(raw):03d}"
        if candidate in alias_to_block_id:
            return candidate
    match = re.fullmatch(r"m(\d+)", raw)
    if match:
        candidate = f"m{int(match.group(1)):03d}"
        if candidate in alias_to_block_id:
            return candidate
    return raw


def extract_unique_answer_mention(raw_output: str, item: BenchmarkItem) -> str:
    candidates = answer_candidates(item)
    if not candidates:
        return ""
    text = truncate_answer_region(raw_output)
    explicit = re.search(r"(?:answer|option\s+code)\s*[:=-]\s*[\"']?([A-Za-z0-9_.%-]+)", text, flags=re.IGNORECASE)
    if explicit:
        value = explicit.group(1)
        for candidate in candidates:
            if value.lower() == candidate.lower():
                return candidate
    matches = {
        candidate
        for candidate in candidates
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE)
    }
    if len(matches) == 1:
        return next(iter(matches))
    return ""


def answer_candidates(item: BenchmarkItem) -> list[str]:
    values: list[str] = []
    for value in [
        item.gold_answer,
        item.metadata.get("target_value"),
        item.metadata.get("stale_value"),
    ]:
        if value:
            values.append(str(value))
    option_labels = item.metadata.get("option_labels")
    if isinstance(option_labels, dict):
        values.extend(str(key) for key in option_labels)
    seen: set[str] = set()
    candidates: list[str] = []
    for value in sorted(values, key=len, reverse=True):
        lowered = value.lower()
        if lowered not in seen:
            seen.add(lowered)
            candidates.append(value)
    return candidates


def truncate_answer_region(raw_output: str) -> str:
    text = raw_output.strip()
    cut_markers = ["JSON only", "Context block", "[block_id=", "Question:", "Return JSON"]
    cut_positions = [idx for marker in cut_markers if (idx := text.find(marker)) >= 0]
    if cut_positions:
        text = text[: min(cut_positions)]
    return text[:500]


def clone_cache(cache: Any) -> Any:
    cloned = copy.deepcopy(cache)
    for layer in cloned.layers:
        clone_tensor_attr(layer, "keys")
        clone_tensor_attr(layer, "values")
    return cloned


def clone_tensor_attr(obj: Any, attr: str) -> None:
    value = getattr(obj, attr, None)
    if isinstance(value, torch.Tensor):
        setattr(obj, attr, value.clone())


def parse_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise KeyError(name)


def sync_torch_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def move_encoded_case(encoded: EncodedCase, device: torch.device) -> EncodedCase:
    if device.type == "cpu":
        return encoded
    return EncodedCase(
        prefix_ids=encoded.prefix_ids.to(device),
        suffix_ids=encoded.suffix_ids.to(device),
        block_ranges=encoded.block_ranges,
        suffix_ranges=encoded.suffix_ranges,
        block_header_ranges=encoded.block_header_ranges,
        block_body_ranges=encoded.block_body_ranges,
        block_aliases=encoded.block_aliases,
        alias_to_block_id=encoded.alias_to_block_id,
    )


def install_phi_compat_shims() -> None:
    from typing import TypedDict

    import transformers.cache_utils as cache_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.utils as transformers_utils

    if not hasattr(cache_utils, "SlidingWindowCache"):
        cache_utils.SlidingWindowCache = cache_utils.StaticCache
    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            pass

        transformers_utils.LossKwargs = LossKwargs

    original = modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys
    if getattr(modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys, "_kv_probe_patched", False):
        return

    def patched_get_expanded_tied_weights_keys(self: Any, all_submodels: bool = False) -> dict[str, str]:
        tied = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied, list):
            if tied == ["lm_head.weight"] and hasattr(self, "model") and hasattr(self.model, "embed_tokens"):
                return {"lm_head.weight": "model.embed_tokens.weight"}
            return {key: key for key in tied}
        return original(self, all_submodels=all_submodels)

    patched_get_expanded_tied_weights_keys._kv_probe_patched = True  # type: ignore[attr-defined]
    modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = patched_get_expanded_tied_weights_keys


def patch_meta_rope_buffers(model: Any) -> int:
    patched = 0
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        original_inv_freq = getattr(module, "original_inv_freq", None)
        if (
            isinstance(inv_freq, torch.Tensor)
            and isinstance(original_inv_freq, torch.Tensor)
            and original_inv_freq.is_meta
        ):
            module.original_inv_freq = inv_freq.detach().clone()
            patched += 1
    return patched


def apply_policy(
    cache: Any,
    encoded: EncodedCase,
    item: BenchmarkItem,
    policy: str,
    attention_token_scores: torch.Tensor | None = None,
) -> None:
    if policy == "full_cache":
        return
    if policy == "zero_detected_conflict_values_last_quarter":
        zero_ranges(
            cache,
            encoded,
            detected_conflict_stale_ids(item, use_current_text_cues=True),
            zero_keys=False,
            zero_values=True,
            layer_slice="last_quarter",
        )
        return
    if policy == "zero_timestamp_conflict_values_last_quarter":
        zero_ranges(
            cache,
            encoded,
            detected_conflict_stale_ids(item, use_current_text_cues=False),
            zero_keys=False,
            zero_values=True,
            layer_slice="last_quarter",
        )
        return
    if policy == "zero_duplicate_aware_conflict_values_last_quarter":
        zero_ranges(
            cache,
            encoded,
            detected_conflict_stale_ids(item, use_current_text_cues=False, prefer_singleton_conflicts=True),
            zero_keys=False,
            zero_values=True,
            layer_slice="last_quarter",
        )
        return
    if policy == "zero_lineage_conflict_values_last_quarter":
        zero_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            zero_keys=False,
            zero_values=True,
            layer_slice="last_quarter",
        )
        return
    if policy == "zero_lineage_conflict_values":
        zero_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            zero_keys=False,
            zero_values=True,
        )
        return
    if policy == "zero_lineage_conflict_keys_values":
        zero_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            zero_keys=True,
            zero_values=True,
        )
        return
    text_timestamp_layer_range_span_match = re.fullmatch(
        r"zero_text_timestamp_conflict_(values|keys|keys_values)_layers_(\d+)_(\d+)_span_(header|body|first_half|second_half|last_quarter|last_token|decile_[0-9])",
        policy,
    )
    if text_timestamp_layer_range_span_match:
        target = text_timestamp_layer_range_span_match.group(1)
        start_idx = int(text_timestamp_layer_range_span_match.group(2))
        end_idx = int(text_timestamp_layer_range_span_match.group(3))
        span_slice = text_timestamp_layer_range_span_match.group(4)
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_partial_ranges(
            cache,
            encoded,
            detected_text_timestamp_stale_ids(item),
            zero_keys=zero_keys,
            zero_values=zero_values,
            layer_slice=f"layers_{start_idx}_{end_idx}",
            span_slice=span_slice,
        )
        return
    text_timestamp_layer_range_match = re.fullmatch(
        r"zero_text_timestamp_conflict_(values|keys|keys_values)_layers_(\d+)_(\d+)",
        policy,
    )
    if text_timestamp_layer_range_match:
        target = text_timestamp_layer_range_match.group(1)
        start_idx = int(text_timestamp_layer_range_match.group(2))
        end_idx = int(text_timestamp_layer_range_match.group(3))
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_ranges(
            cache,
            encoded,
            detected_text_timestamp_stale_ids(item),
            zero_keys=zero_keys,
            zero_values=zero_values,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    text_timestamp_graft_range_match = re.fullmatch(
        r"graft_text_timestamp_current_mean_values_layers_(\d+)_(\d+)",
        policy,
    )
    if text_timestamp_graft_range_match:
        start_idx = int(text_timestamp_graft_range_match.group(1))
        end_idx = int(text_timestamp_graft_range_match.group(2))
        graft_current_mean_values(
            cache,
            encoded,
            detected_text_timestamp_pairs(item),
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    text_timestamp_graft_resample_range_match = re.fullmatch(
        r"graft_text_timestamp_current_resample_values_layers_(\d+)_(\d+)",
        policy,
    )
    if text_timestamp_graft_resample_range_match:
        start_idx = int(text_timestamp_graft_resample_range_match.group(1))
        end_idx = int(text_timestamp_graft_resample_range_match.group(2))
        graft_current_resample_values(
            cache,
            encoded,
            detected_text_timestamp_pairs(item),
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    text_timestamp_graft_normmatch_range_match = re.fullmatch(
        r"graft_text_timestamp_current_mean_normmatch_values_layers_(\d+)_(\d+)",
        policy,
    )
    if text_timestamp_graft_normmatch_range_match:
        start_idx = int(text_timestamp_graft_normmatch_range_match.group(1))
        end_idx = int(text_timestamp_graft_normmatch_range_match.group(2))
        graft_current_mean_values(
            cache,
            encoded,
            detected_text_timestamp_pairs(item),
            layer_slice=f"layers_{start_idx}_{end_idx}",
            norm_match=True,
        )
        return
    text_timestamp_scale_range_match = re.fullmatch(
        r"scale_text_timestamp_conflict_values_layers_(\d+)_(\d+)_factor_([0-9]+(?:_[0-9]+)?)",
        policy,
    )
    if text_timestamp_scale_range_match:
        start_idx = int(text_timestamp_scale_range_match.group(1))
        end_idx = int(text_timestamp_scale_range_match.group(2))
        factor = parse_policy_factor(text_timestamp_scale_range_match.group(3))
        scale_ranges(
            cache,
            encoded,
            detected_text_timestamp_stale_ids(item),
            value_scale=factor,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    text_timestamp_scale_current_range_match = re.fullmatch(
        r"scale_text_timestamp_current_values_layers_(\d+)_(\d+)_factor_([0-9]+(?:_[0-9]+)?)",
        policy,
    )
    if text_timestamp_scale_current_range_match:
        start_idx = int(text_timestamp_scale_current_range_match.group(1))
        end_idx = int(text_timestamp_scale_current_range_match.group(2))
        factor = parse_policy_factor(text_timestamp_scale_current_range_match.group(3))
        scale_ranges(
            cache,
            encoded,
            detected_text_timestamp_current_ids(item),
            value_scale=factor,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    text_timestamp_zero_boost_range_match = re.fullmatch(
        r"zero_text_timestamp_conflict_values_layers_(\d+)_(\d+)_boost_current_factor_([0-9]+(?:_[0-9]+)?)",
        policy,
    )
    if text_timestamp_zero_boost_range_match:
        start_idx = int(text_timestamp_zero_boost_range_match.group(1))
        end_idx = int(text_timestamp_zero_boost_range_match.group(2))
        factor = parse_policy_factor(text_timestamp_zero_boost_range_match.group(3))
        zero_stale_and_scale_current_values(
            cache,
            encoded,
            detected_text_timestamp_pairs(item),
            value_scale=factor,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    text_timestamp_zero_target_boost_range_match = re.fullmatch(
        r"zero_text_timestamp_conflict_(keys|keys_values)_layers_(\d+)_(\d+)_boost_current_values_factor_([0-9]+(?:_[0-9]+)?)",
        policy,
    )
    if text_timestamp_zero_target_boost_range_match:
        target = text_timestamp_zero_target_boost_range_match.group(1)
        start_idx = int(text_timestamp_zero_target_boost_range_match.group(2))
        end_idx = int(text_timestamp_zero_target_boost_range_match.group(3))
        factor = parse_policy_factor(text_timestamp_zero_target_boost_range_match.group(4))
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_stale_and_scale_current(
            cache,
            encoded,
            detected_text_timestamp_pairs(item),
            zero_keys=zero_keys,
            zero_values=zero_values,
            current_value_scale=factor,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    query_ref_layer_range_span_match = re.fullmatch(
        r"zero_query_ref_conflict_(values|keys|keys_values)_layers_(\d+)_(\d+)_span_(header|body|first_half|second_half|last_quarter|last_token|decile_[0-9])",
        policy,
    )
    if query_ref_layer_range_span_match:
        target = query_ref_layer_range_span_match.group(1)
        start_idx = int(query_ref_layer_range_span_match.group(2))
        end_idx = int(query_ref_layer_range_span_match.group(3))
        span_slice = query_ref_layer_range_span_match.group(4)
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_partial_ranges(
            cache,
            encoded,
            detected_query_ref_stale_ids(item),
            zero_keys=zero_keys,
            zero_values=zero_values,
            layer_slice=f"layers_{start_idx}_{end_idx}",
            span_slice=span_slice,
        )
        return
    query_ref_layer_range_match = re.fullmatch(
        r"zero_query_ref_conflict_(values|keys|keys_values)_layers_(\d+)_(\d+)",
        policy,
    )
    if query_ref_layer_range_match:
        target = query_ref_layer_range_match.group(1)
        start_idx = int(query_ref_layer_range_match.group(2))
        end_idx = int(query_ref_layer_range_match.group(3))
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_ranges(
            cache,
            encoded,
            detected_query_ref_stale_ids(item),
            zero_keys=zero_keys,
            zero_values=zero_values,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    if policy == "graft_lineage_current_mean_values":
        graft_current_mean_values(cache, encoded, detected_lineage_pairs(item))
        return
    if policy == "graft_lineage_current_resample_values":
        graft_current_resample_values(cache, encoded, detected_lineage_pairs(item))
        return
    if policy == "graft_lineage_current_mean_normmatch_values":
        graft_current_mean_values(cache, encoded, detected_lineage_pairs(item), norm_match=True)
        return
    if policy == "graft_lineage_current_mean_values_last_quarter":
        graft_current_mean_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            layer_slice="last_quarter",
        )
        return
    if policy == "graft_lineage_current_resample_values_last_quarter":
        graft_current_resample_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            layer_slice="last_quarter",
        )
        return
    if policy == "graft_lineage_current_mean_normmatch_values_last_quarter":
        graft_current_mean_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            layer_slice="last_quarter",
            norm_match=True,
        )
        return
    lineage_graft_range_head_match = re.fullmatch(
        r"graft_lineage_current_mean_values_layers_(\d+)_(\d+)_head_(\d+)",
        policy,
    )
    if lineage_graft_range_head_match:
        start_idx = int(lineage_graft_range_head_match.group(1))
        end_idx = int(lineage_graft_range_head_match.group(2))
        head_idx = int(lineage_graft_range_head_match.group(3))
        graft_current_mean_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            layer_slice=f"layers_{start_idx}_{end_idx}",
            head_idx=head_idx,
        )
        return
    lineage_graft_resample_range_head_match = re.fullmatch(
        r"graft_lineage_current_resample_values_layers_(\d+)_(\d+)_head_(\d+)",
        policy,
    )
    if lineage_graft_resample_range_head_match:
        start_idx = int(lineage_graft_resample_range_head_match.group(1))
        end_idx = int(lineage_graft_resample_range_head_match.group(2))
        head_idx = int(lineage_graft_resample_range_head_match.group(3))
        graft_current_resample_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            layer_slice=f"layers_{start_idx}_{end_idx}",
            head_idx=head_idx,
        )
        return
    lineage_graft_normmatch_range_head_match = re.fullmatch(
        r"graft_lineage_current_mean_normmatch_values_layers_(\d+)_(\d+)_head_(\d+)",
        policy,
    )
    if lineage_graft_normmatch_range_head_match:
        start_idx = int(lineage_graft_normmatch_range_head_match.group(1))
        end_idx = int(lineage_graft_normmatch_range_head_match.group(2))
        head_idx = int(lineage_graft_normmatch_range_head_match.group(3))
        graft_current_mean_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            layer_slice=f"layers_{start_idx}_{end_idx}",
            head_idx=head_idx,
            norm_match=True,
        )
        return
    lineage_graft_range_match = re.fullmatch(r"graft_lineage_current_mean_values_layers_(\d+)_(\d+)", policy)
    if lineage_graft_range_match:
        start_idx = int(lineage_graft_range_match.group(1))
        end_idx = int(lineage_graft_range_match.group(2))
        graft_current_mean_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    lineage_graft_resample_range_match = re.fullmatch(
        r"graft_lineage_current_resample_values_layers_(\d+)_(\d+)",
        policy,
    )
    if lineage_graft_resample_range_match:
        start_idx = int(lineage_graft_resample_range_match.group(1))
        end_idx = int(lineage_graft_resample_range_match.group(2))
        graft_current_resample_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    lineage_graft_normmatch_range_match = re.fullmatch(
        r"graft_lineage_current_mean_normmatch_values_layers_(\d+)_(\d+)",
        policy,
    )
    if lineage_graft_normmatch_range_match:
        start_idx = int(lineage_graft_normmatch_range_match.group(1))
        end_idx = int(lineage_graft_normmatch_range_match.group(2))
        graft_current_mean_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            layer_slice=f"layers_{start_idx}_{end_idx}",
            norm_match=True,
        )
        return
    lineage_scale_range_head_match = re.fullmatch(
        r"scale_lineage_conflict_values_layers_(\d+)_(\d+)_factor_([0-9]+(?:_[0-9]+)?)_head_(\d+)",
        policy,
    )
    if lineage_scale_range_head_match:
        start_idx = int(lineage_scale_range_head_match.group(1))
        end_idx = int(lineage_scale_range_head_match.group(2))
        factor = parse_policy_factor(lineage_scale_range_head_match.group(3))
        head_idx = int(lineage_scale_range_head_match.group(4))
        scale_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            value_scale=factor,
            layer_slice=f"layers_{start_idx}_{end_idx}",
            head_idx=head_idx,
        )
        return
    lineage_scale_range_match = re.fullmatch(
        r"scale_lineage_conflict_values_layers_(\d+)_(\d+)_factor_([0-9]+(?:_[0-9]+)?)",
        policy,
    )
    if lineage_scale_range_match:
        start_idx = int(lineage_scale_range_match.group(1))
        end_idx = int(lineage_scale_range_match.group(2))
        factor = parse_policy_factor(lineage_scale_range_match.group(3))
        scale_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            value_scale=factor,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    lineage_scale_current_range_head_match = re.fullmatch(
        r"scale_lineage_current_values_layers_(\d+)_(\d+)_factor_([0-9]+(?:_[0-9]+)?)_head_(\d+)",
        policy,
    )
    if lineage_scale_current_range_head_match:
        start_idx = int(lineage_scale_current_range_head_match.group(1))
        end_idx = int(lineage_scale_current_range_head_match.group(2))
        factor = parse_policy_factor(lineage_scale_current_range_head_match.group(3))
        head_idx = int(lineage_scale_current_range_head_match.group(4))
        scale_ranges(
            cache,
            encoded,
            detected_lineage_current_ids(item),
            value_scale=factor,
            layer_slice=f"layers_{start_idx}_{end_idx}",
            head_idx=head_idx,
        )
        return
    lineage_scale_current_range_match = re.fullmatch(
        r"scale_lineage_current_values_layers_(\d+)_(\d+)_factor_([0-9]+(?:_[0-9]+)?)",
        policy,
    )
    if lineage_scale_current_range_match:
        start_idx = int(lineage_scale_current_range_match.group(1))
        end_idx = int(lineage_scale_current_range_match.group(2))
        factor = parse_policy_factor(lineage_scale_current_range_match.group(3))
        scale_ranges(
            cache,
            encoded,
            detected_lineage_current_ids(item),
            value_scale=factor,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    lineage_zero_boost_range_head_match = re.fullmatch(
        r"zero_lineage_conflict_values_layers_(\d+)_(\d+)_boost_current_factor_([0-9]+(?:_[0-9]+)?)_head_(\d+)",
        policy,
    )
    if lineage_zero_boost_range_head_match:
        start_idx = int(lineage_zero_boost_range_head_match.group(1))
        end_idx = int(lineage_zero_boost_range_head_match.group(2))
        factor = parse_policy_factor(lineage_zero_boost_range_head_match.group(3))
        head_idx = int(lineage_zero_boost_range_head_match.group(4))
        zero_stale_and_scale_current_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            value_scale=factor,
            layer_slice=f"layers_{start_idx}_{end_idx}",
            head_idx=head_idx,
        )
        return
    lineage_zero_boost_range_match = re.fullmatch(
        r"zero_lineage_conflict_values_layers_(\d+)_(\d+)_boost_current_factor_([0-9]+(?:_[0-9]+)?)",
        policy,
    )
    if lineage_zero_boost_range_match:
        start_idx = int(lineage_zero_boost_range_match.group(1))
        end_idx = int(lineage_zero_boost_range_match.group(2))
        factor = parse_policy_factor(lineage_zero_boost_range_match.group(3))
        zero_stale_and_scale_current_values(
            cache,
            encoded,
            detected_lineage_pairs(item),
            value_scale=factor,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    lineage_layer_head_match = re.fullmatch(
        r"zero_lineage_conflict_(values|keys|keys_values)_layer_(\d+)_head_(\d+)",
        policy,
    )
    if lineage_layer_head_match:
        target = lineage_layer_head_match.group(1)
        layer_idx = int(lineage_layer_head_match.group(2))
        head_idx = int(lineage_layer_head_match.group(3))
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            zero_keys=zero_keys,
            zero_values=zero_values,
            layer_slice=f"layer_{layer_idx}",
            head_idx=head_idx,
        )
        return
    lineage_layer_range_head_match = re.fullmatch(
        r"zero_lineage_conflict_(values|keys|keys_values)_layers_(\d+)_(\d+)_head_(\d+)",
        policy,
    )
    if lineage_layer_range_head_match:
        target = lineage_layer_range_head_match.group(1)
        start_idx = int(lineage_layer_range_head_match.group(2))
        end_idx = int(lineage_layer_range_head_match.group(3))
        head_idx = int(lineage_layer_range_head_match.group(4))
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            zero_keys=zero_keys,
            zero_values=zero_values,
            layer_slice=f"layers_{start_idx}_{end_idx}",
            head_idx=head_idx,
        )
        return
    lineage_layer_range_span_match = re.fullmatch(
        r"zero_lineage_conflict_(values|keys|keys_values)_layers_(\d+)_(\d+)_span_(header|body|first_half|second_half|last_quarter|last_token|decile_[0-9])",
        policy,
    )
    if lineage_layer_range_span_match:
        target = lineage_layer_range_span_match.group(1)
        start_idx = int(lineage_layer_range_span_match.group(2))
        end_idx = int(lineage_layer_range_span_match.group(3))
        span_slice = lineage_layer_range_span_match.group(4)
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_partial_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            zero_keys=zero_keys,
            zero_values=zero_values,
            layer_slice=f"layers_{start_idx}_{end_idx}",
            span_slice=span_slice,
        )
        return
    lineage_layer_range_attn_match = re.fullmatch(
        r"zero_lineage_conflict_(values|keys|keys_values)_layers_(\d+)_(\d+)_attn_topfrac_([0-9]+(?:_[0-9]+)?)",
        policy,
    )
    if lineage_layer_range_attn_match:
        target = lineage_layer_range_attn_match.group(1)
        start_idx = int(lineage_layer_range_attn_match.group(2))
        end_idx = int(lineage_layer_range_attn_match.group(3))
        fraction = parse_policy_factor(lineage_layer_range_attn_match.group(4))
        if attention_token_scores is None:
            raise ValueError(f"{policy} requires query attention token scores")
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_attention_top_fraction_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            attention_token_scores=attention_token_scores,
            fraction=fraction,
            zero_keys=zero_keys,
            zero_values=zero_values,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    lineage_layer_range_match = re.fullmatch(r"zero_lineage_conflict_(values|keys|keys_values)_layers_(\d+)_(\d+)", policy)
    if lineage_layer_range_match:
        target = lineage_layer_range_match.group(1)
        start_idx = int(lineage_layer_range_match.group(2))
        end_idx = int(lineage_layer_range_match.group(3))
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            zero_keys=zero_keys,
            zero_values=zero_values,
            layer_slice=f"layers_{start_idx}_{end_idx}",
        )
        return
    lineage_single_layer_match = re.fullmatch(r"zero_lineage_conflict_(values|keys|keys_values)_layer_(\d+)", policy)
    if lineage_single_layer_match:
        target = lineage_single_layer_match.group(1)
        layer_idx = int(lineage_single_layer_match.group(2))
        zero_keys, zero_values = zero_modes_for_target(target)
        zero_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            zero_keys=zero_keys,
            zero_values=zero_values,
            layer_slice=f"layer_{layer_idx}",
        )
        return
    if policy == "zero_detected_conflict_layer_20_head_1":
        zero_ranges(
            cache,
            encoded,
            detected_conflict_stale_ids(item, use_current_text_cues=True),
            zero_keys=False,
            zero_values=True,
            layer_slice="layer_20",
            head_idx=1,
        )
        return
    if policy == "zero_lineage_conflict_layer_20_head_1":
        zero_ranges(
            cache,
            encoded,
            detected_lineage_stale_ids(item),
            zero_keys=False,
            zero_values=True,
            layer_slice="layer_20",
            head_idx=1,
        )
        return
    if policy == "zero_timestamp_conflict_layer_20_head_1":
        zero_ranges(
            cache,
            encoded,
            detected_conflict_stale_ids(item, use_current_text_cues=False),
            zero_keys=False,
            zero_values=True,
            layer_slice="layer_20",
            head_idx=1,
        )
        return
    if policy == "zero_duplicate_aware_conflict_layer_20_head_1":
        zero_ranges(
            cache,
            encoded,
            detected_conflict_stale_ids(item, use_current_text_cues=False, prefer_singleton_conflicts=True),
            zero_keys=False,
            zero_values=True,
            layer_slice="layer_20",
            head_idx=1,
        )
        return
    layer_head_match = re.fullmatch(r"zero_stale_values_layer_(\d+)_head_(\d+)", policy)
    if layer_head_match:
        layer_idx = int(layer_head_match.group(1))
        head_idx = int(layer_head_match.group(2))
        zero_ranges(
            cache,
            encoded,
            item.stale_ids,
            zero_keys=False,
            zero_values=True,
            layer_slice=f"layer_{layer_idx}",
            head_idx=head_idx,
        )
        return
    single_layer_match = re.fullmatch(r"zero_stale_values_layer_(\d+)", policy)
    if single_layer_match:
        layer_idx = int(single_layer_match.group(1))
        zero_ranges(cache, encoded, item.stale_ids, zero_keys=False, zero_values=True, layer_slice=f"layer_{layer_idx}")
        return
    if policy == "zero_stale_values":
        zero_ranges(cache, encoded, item.stale_ids, zero_keys=False, zero_values=True)
        return
    if policy == "zero_stale_keys_values":
        zero_ranges(cache, encoded, item.stale_ids, zero_keys=True, zero_values=True)
        return
    if policy == "zero_stale_values_lower_half":
        zero_ranges(cache, encoded, item.stale_ids, zero_keys=False, zero_values=True, layer_slice="lower_half")
        return
    if policy == "zero_stale_values_upper_half":
        zero_ranges(cache, encoded, item.stale_ids, zero_keys=False, zero_values=True, layer_slice="upper_half")
        return
    if policy == "zero_stale_values_first_quarter":
        zero_ranges(cache, encoded, item.stale_ids, zero_keys=False, zero_values=True, layer_slice="first_quarter")
        return
    if policy == "zero_stale_values_second_quarter":
        zero_ranges(cache, encoded, item.stale_ids, zero_keys=False, zero_values=True, layer_slice="second_quarter")
        return
    if policy == "zero_stale_values_third_quarter":
        zero_ranges(cache, encoded, item.stale_ids, zero_keys=False, zero_values=True, layer_slice="third_quarter")
        return
    if policy == "zero_stale_values_fourth_quarter":
        zero_ranges(cache, encoded, item.stale_ids, zero_keys=False, zero_values=True, layer_slice="fourth_quarter")
        return
    if policy == "zero_stale_values_last_quarter":
        zero_ranges(cache, encoded, item.stale_ids, zero_keys=False, zero_values=True, layer_slice="last_quarter")
        return
    if policy == "zero_superseded_values_last_quarter":
        block_ids = [block.id for block in item.context_blocks if block.status == "superseded"]
        zero_ranges(cache, encoded, block_ids, zero_keys=False, zero_values=True, layer_slice="last_quarter")
        return
    if policy == "zero_random_non_stale_values_last_quarter":
        block_ids = random_non_stale_blocks(encoded, item)
        zero_ranges(cache, encoded, block_ids, zero_keys=False, zero_values=True, layer_slice="last_quarter")
        return
    if policy == "zero_non_gold_values":
        gold = set(item.gold_evidence_ids)
        target_ids = [block_id for block_id in encoded.block_ranges if block_id not in gold]
        zero_ranges(cache, encoded, target_ids, zero_keys=False, zero_values=True)
        return
    raise KeyError(policy)


def prompt_policy_excluded_block_ids(policy: str, item: BenchmarkItem) -> set[str] | None:
    if policy == "drop_stale_prompt":
        return set(item.stale_ids)
    if policy == "keep_gold_prompt":
        return {block.id for block in item.context_blocks if block.id not in set(item.gold_evidence_ids)}
    if policy == "keep_query_key_prompt":
        memory_key = query_memory_key(item.query)
        if not memory_key:
            return set()
        return {
            block.id
            for block in item.context_blocks
            if memory_key not in block_memory_keys(block.text)
        }
    if policy == "keep_latest_per_memory_key_prompt":
        keep_ids = latest_block_ids_per_memory_key(item)
        if not keep_ids:
            return set()
        return {block.id for block in item.context_blocks if block.id not in keep_ids}
    if policy == "keep_text_timestamp_current_prompt":
        current_ids = set(detected_text_timestamp_current_ids(item))
        if not current_ids:
            return {block.id for block in item.context_blocks}
        return {block.id for block in item.context_blocks if block.id not in current_ids}
    if policy == "summarize_text_timestamp_current_prompt":
        return set()
    return None


def prompt_policy_item(policy: str, item: BenchmarkItem) -> BenchmarkItem:
    if policy != "summarize_text_timestamp_current_prompt":
        return item
    current_ids = set(detected_text_timestamp_current_ids(item))
    current_blocks = [block for block in item.context_blocks if block.id in current_ids]
    summary_blocks = [current_memory_summary_block(block) for block in current_blocks]
    metadata = dict(item.metadata)
    metadata["prompt_policy_source"] = policy
    return BenchmarkItem(
        id=item.id,
        domain=item.domain,
        question_type=item.question_type,
        context_blocks=summary_blocks,
        query=item.query,
        gold_answer=item.gold_answer,
        gold_evidence_ids=item.gold_evidence_ids,
        stale_ids=[],
        distractor_ids=[],
        metadata=metadata,
    )


def current_memory_summary_block(block: Any) -> ContextBlock:
    keys = sorted(block_memory_keys(block.text))
    memory_key = keys[0] if keys else normalize_memory_key(block.metadata.get("memory_key", ""))
    value = block.metadata.get("value")
    if value:
        summary = f"Memory key: {memory_key}. Current value: {value}."
    else:
        summary = f"Current memory summary. {block.text}"
    metadata = dict(block.metadata)
    metadata["synthetic_summary"] = True
    metadata["source_block_id"] = block.id
    return ContextBlock(
        id=block.id,
        timestamp=block.timestamp,
        type=block.type,
        trust=block.trust,
        status=block.status,
        text=summary,
        metadata=metadata,
    )


def zero_modes_for_target(target: str) -> tuple[bool, bool]:
    if target == "values":
        return False, True
    if target == "keys":
        return True, False
    if target == "keys_values":
        return True, True
    raise KeyError(target)


def latest_block_ids_per_memory_key(item: BenchmarkItem) -> set[str]:
    latest_by_key: dict[str, Any] = {}
    for block in item.context_blocks:
        keys = block_memory_keys(block.text)
        if not keys:
            continue
        for key in keys:
            current = latest_by_key.get(key)
            if current is None or block.parsed_timestamp > current.parsed_timestamp:
                latest_by_key[key] = block
    return {block.id for block in latest_by_key.values()}


def zero_ranges(
    cache: Any,
    encoded: EncodedCase,
    block_ids: list[str],
    zero_keys: bool,
    zero_values: bool,
    layer_slice: str = "all",
    head_idx: int | None = None,
) -> None:
    layers = selected_layers(cache.layers, layer_slice)
    for block_id in block_ids:
        if block_id not in encoded.block_ranges:
            continue
        start, end = encoded.block_ranges[block_id]
        for layer in layers:
            head_slice = slice(None)
            if head_idx is not None:
                head_count = layer.values.shape[1]
                if head_idx < 0 or head_idx >= head_count:
                    raise IndexError(f"Head index {head_idx} outside 0..{head_count - 1}")
                head_slice = slice(head_idx, head_idx + 1)
            if zero_keys:
                layer.keys[:, head_slice, start:end, :] = 0
            if zero_values:
                layer.values[:, head_slice, start:end, :] = 0


def zero_partial_ranges(
    cache: Any,
    encoded: EncodedCase,
    block_ids: list[str],
    zero_keys: bool,
    zero_values: bool,
    layer_slice: str,
    span_slice: str,
    head_idx: int | None = None,
) -> None:
    layers = selected_layers(cache.layers, layer_slice)
    for block_id in block_ids:
        if block_id not in encoded.block_ranges:
            continue
        start, end = block_subrange(encoded, block_id, span_slice)
        if start >= end:
            continue
        for layer in layers:
            head_slice = slice(None)
            if head_idx is not None:
                head_count = layer.values.shape[1]
                if head_idx < 0 or head_idx >= head_count:
                    raise IndexError(f"Head index {head_idx} outside 0..{head_count - 1}")
                head_slice = slice(head_idx, head_idx + 1)
            if zero_keys:
                layer.keys[:, head_slice, start:end, :] = 0
            if zero_values:
                layer.values[:, head_slice, start:end, :] = 0


def zero_attention_top_fraction_ranges(
    cache: Any,
    encoded: EncodedCase,
    block_ids: list[str],
    attention_token_scores: torch.Tensor,
    fraction: float,
    zero_keys: bool,
    zero_values: bool,
    layer_slice: str,
    head_idx: int | None = None,
) -> None:
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"attention top fraction must be in (0, 1], got {fraction}")
    if attention_token_scores.ndim != 1:
        raise ValueError("attention_token_scores must be a 1D tensor over prefix token positions")
    layers = selected_layers(cache.layers, layer_slice)
    for block_id in block_ids:
        if block_id not in encoded.block_ranges:
            continue
        block_start, block_end = encoded.block_ranges[block_id]
        block_len = block_end - block_start
        if block_len <= 0:
            continue
        k = max(1, int(torch.ceil(torch.tensor(block_len * fraction)).item()))
        block_scores = attention_token_scores[block_start:block_end]
        top_offsets = torch.topk(block_scores, k=min(k, block_len), largest=True, sorted=False).indices
        token_indices = torch.sort(top_offsets + block_start).values
        for layer in layers:
            head_slice = slice(None)
            if head_idx is not None:
                head_count = layer.values.shape[1]
                if head_idx < 0 or head_idx >= head_count:
                    raise IndexError(f"Head index {head_idx} outside 0..{head_count - 1}")
                head_slice = slice(head_idx, head_idx + 1)
            device_token_indices = token_indices.to(layer.values.device)
            if zero_keys:
                layer.keys[:, head_slice, device_token_indices, :] = 0
            if zero_values:
                layer.values[:, head_slice, device_token_indices, :] = 0


def scale_ranges(
    cache: Any,
    encoded: EncodedCase,
    block_ids: list[str],
    value_scale: float,
    layer_slice: str = "all",
    head_idx: int | None = None,
) -> None:
    if value_scale < 0:
        raise ValueError(f"value_scale must be non-negative, got {value_scale}")
    layers = selected_layers(cache.layers, layer_slice)
    for block_id in block_ids:
        if block_id not in encoded.block_ranges:
            continue
        start, end = encoded.block_ranges[block_id]
        for layer in layers:
            head_slice = slice(None)
            if head_idx is not None:
                head_count = layer.values.shape[1]
                if head_idx < 0 or head_idx >= head_count:
                    raise IndexError(f"Head index {head_idx} outside 0..{head_count - 1}")
                head_slice = slice(head_idx, head_idx + 1)
            layer.values[:, head_slice, start:end, :] *= value_scale


def zero_stale_and_scale_current_values(
    cache: Any,
    encoded: EncodedCase,
    stale_current_pairs: list[tuple[str, str]],
    value_scale: float,
    layer_slice: str = "all",
    head_idx: int | None = None,
) -> None:
    stale_ids = sorted({stale_id for stale_id, _ in stale_current_pairs})
    current_ids = sorted({current_id for _, current_id in stale_current_pairs})
    zero_ranges(
        cache,
        encoded,
        stale_ids,
        zero_keys=False,
        zero_values=True,
        layer_slice=layer_slice,
        head_idx=head_idx,
    )
    scale_ranges(
        cache,
        encoded,
        current_ids,
        value_scale=value_scale,
        layer_slice=layer_slice,
        head_idx=head_idx,
    )


def zero_stale_and_scale_current(
    cache: Any,
    encoded: EncodedCase,
    stale_current_pairs: list[tuple[str, str]],
    zero_keys: bool,
    zero_values: bool,
    current_value_scale: float,
    layer_slice: str = "all",
    head_idx: int | None = None,
) -> None:
    stale_ids = sorted({stale_id for stale_id, _ in stale_current_pairs})
    current_ids = sorted({current_id for _, current_id in stale_current_pairs})
    zero_ranges(
        cache,
        encoded,
        stale_ids,
        zero_keys=zero_keys,
        zero_values=zero_values,
        layer_slice=layer_slice,
        head_idx=head_idx,
    )
    scale_ranges(
        cache,
        encoded,
        current_ids,
        value_scale=current_value_scale,
        layer_slice=layer_slice,
        head_idx=head_idx,
    )


def graft_current_mean_values(
    cache: Any,
    encoded: EncodedCase,
    stale_current_pairs: list[tuple[str, str]],
    layer_slice: str = "all",
    head_idx: int | None = None,
    norm_match: bool = False,
) -> None:
    layers = selected_layers(cache.layers, layer_slice)
    for stale_id, current_id in stale_current_pairs:
        if stale_id not in encoded.block_ranges or current_id not in encoded.block_ranges:
            continue
        stale_start, stale_end = encoded.block_ranges[stale_id]
        current_start, current_end = encoded.block_ranges[current_id]
        stale_len = stale_end - stale_start
        for layer in layers:
            head_slice = slice(None)
            if head_idx is not None:
                head_count = layer.values.shape[1]
                if head_idx < 0 or head_idx >= head_count:
                    raise IndexError(f"Head index {head_idx} outside 0..{head_count - 1}")
                head_slice = slice(head_idx, head_idx + 1)
            current_mean = layer.values[:, head_slice, current_start:current_end, :].mean(dim=2, keepdim=True)
            graft_values = current_mean.expand(-1, -1, stale_len, -1)
            if norm_match:
                stale_values = layer.values[:, head_slice, stale_start:stale_end, :]
                stale_norm = stale_values.norm(dim=-1, keepdim=True)
                graft_norm = graft_values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                graft_values = graft_values * (stale_norm / graft_norm)
            layer.values[:, head_slice, stale_start:stale_end, :] = graft_values


def graft_current_resample_values(
    cache: Any,
    encoded: EncodedCase,
    stale_current_pairs: list[tuple[str, str]],
    layer_slice: str = "all",
    head_idx: int | None = None,
) -> None:
    layers = selected_layers(cache.layers, layer_slice)
    for stale_id, current_id in stale_current_pairs:
        if stale_id not in encoded.block_ranges or current_id not in encoded.block_ranges:
            continue
        stale_start, stale_end = encoded.block_ranges[stale_id]
        current_start, current_end = encoded.block_ranges[current_id]
        stale_len = stale_end - stale_start
        current_len = current_end - current_start
        if stale_len <= 0 or current_len <= 0:
            continue
        for layer in layers:
            head_slice = slice(None)
            if head_idx is not None:
                head_count = layer.values.shape[1]
                if head_idx < 0 or head_idx >= head_count:
                    raise IndexError(f"Head index {head_idx} outside 0..{head_count - 1}")
                head_slice = slice(head_idx, head_idx + 1)
            offsets = torch.arange(stale_len, device=layer.values.device) * current_len // stale_len
            current_indices = current_start + offsets
            graft_values = layer.values[:, head_slice, current_indices, :].clone()
            layer.values[:, head_slice, stale_start:stale_end, :] = graft_values


def parse_policy_factor(value: str) -> float:
    return float(value.replace("_", "."))


def partial_block_range(start: int, end: int, span_slice: str) -> tuple[int, int]:
    length = end - start
    if length <= 0:
        return start, start
    decile_match = re.fullmatch(r"decile_([0-9])", span_slice)
    if decile_match:
        idx = int(decile_match.group(1))
        return start + (length * idx) // 10, start + (length * (idx + 1)) // 10
    if span_slice == "first_half":
        return start, start + max(1, length // 2)
    if span_slice == "second_half":
        return start + (length // 2), end
    if span_slice == "last_quarter":
        return end - max(1, length // 4), end
    if span_slice == "last_token":
        return end - 1, end
    raise KeyError(span_slice)


def block_subrange(encoded: EncodedCase, block_id: str, span_slice: str) -> tuple[int, int]:
    if span_slice == "header":
        ranges = encoded.block_header_ranges or {}
        return ranges.get(block_id, encoded.block_ranges[block_id])
    if span_slice == "body":
        ranges = encoded.block_body_ranges or {}
        return ranges.get(block_id, encoded.block_ranges[block_id])
    return partial_block_range(*encoded.block_ranges[block_id], span_slice)


def selected_layers(layers: list[Any], layer_slice: str) -> list[Any]:
    return [layer for idx in selected_layer_indices(len(layers), layer_slice) if is_kv_editable_layer(layer := layers[idx])]


def is_kv_editable_layer(layer: Any) -> bool:
    return isinstance(getattr(layer, "keys", None), torch.Tensor) and isinstance(getattr(layer, "values", None), torch.Tensor)


def selected_layer_indices(layer_count: int, layer_slice: str) -> list[int]:
    if layer_slice == "all":
        return list(range(layer_count))
    quarter = max(1, layer_count // 4)
    midpoint = layer_count // 2
    if layer_slice == "lower_half":
        return list(range(0, midpoint))
    if layer_slice == "upper_half":
        return list(range(midpoint, layer_count))
    if layer_slice == "first_quarter":
        return list(range(0, quarter))
    if layer_slice == "second_quarter":
        return list(range(quarter, midpoint))
    if layer_slice == "third_quarter":
        return list(range(midpoint, layer_count - quarter))
    if layer_slice == "fourth_quarter":
        return list(range(layer_count - quarter, layer_count))
    if layer_slice == "last_quarter":
        start = layer_count - max(1, layer_count // 4)
        return list(range(start, layer_count))
    layer_match = re.fullmatch(r"layer_(\d+)", layer_slice)
    if layer_match:
        layer_idx = int(layer_match.group(1))
        if layer_idx < 0 or layer_idx >= layer_count:
            raise IndexError(f"Layer index {layer_idx} outside 0..{layer_count - 1}")
        return [layer_idx]
    layer_range_match = re.fullmatch(r"layers_(\d+)_(\d+)", layer_slice)
    if layer_range_match:
        start_idx = int(layer_range_match.group(1))
        end_idx = int(layer_range_match.group(2))
        if start_idx < 0 or end_idx < start_idx or end_idx >= layer_count:
            raise IndexError(f"Layer range {start_idx}..{end_idx} outside 0..{layer_count - 1}")
        return list(range(start_idx, end_idx + 1))
    raise KeyError(layer_slice)


def attention_policy_layer_slice(policy: str) -> str | None:
    match = re.fullmatch(
        r"zero_lineage_conflict_(?:values|keys|keys_values)_layers_(\d+)_(\d+)_attn_topfrac_[0-9]+(?:_[0-9]+)?",
        policy,
    )
    if not match:
        return None
    return f"layers_{int(match.group(1))}_{int(match.group(2))}"


def query_attention_token_scores(
    model: Any,
    cache: Any,
    prefix_len: int,
    suffix_ids: torch.Tensor,
    suffix_ranges: dict[str, tuple[int, int]],
    layer_slice: str,
) -> torch.Tensor:
    if "query" not in suffix_ranges:
        raise ValueError("attention-gated policies require an encoded query suffix range")
    query_start, query_end = suffix_ranges["query"]
    if query_start >= query_end:
        raise ValueError("attention-gated policies require at least one query token")
    attention_mask = torch.ones((1, prefix_len + suffix_ids.shape[1]), dtype=torch.long, device=suffix_ids.device)
    with torch.no_grad():
        output = model(
            input_ids=suffix_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
        )
    attentions = output.attentions
    if not attentions:
        raise ValueError(
            "Model did not return attentions. Try rerunning with --attn-implementation eager."
        )
    layer_indices = selected_layer_indices(len(attentions), layer_slice)
    scores = torch.zeros(prefix_len, dtype=torch.float32)
    for layer_idx in layer_indices:
        layer_attention = attentions[layer_idx]
        if layer_attention is None:
            raise ValueError(
                "Model returned empty attention tensors. Try rerunning with --attn-implementation eager."
            )
        prefix_attention = layer_attention[0, :, query_start:query_end, :prefix_len]
        scores += prefix_attention.float().mean(dim=(0, 1)).cpu()
    return scores / max(1, len(layer_indices))


def detected_conflict_stale_ids(
    item: BenchmarkItem,
    use_current_text_cues: bool = True,
    prefer_singleton_conflicts: bool = False,
) -> list[str]:
    claims: dict[tuple[str, str], list[tuple[str, str, str, bool]]] = {}
    for block in item.context_blocks:
        claim = extract_value_claim(block)
        if claim is None:
            continue
        entity, prop, value = claim
        is_current_cue = use_current_text_cues and has_current_text_cue(block.text)
        claims.setdefault((entity, prop), []).append((block.id, block.timestamp, value, is_current_cue))

    stale_ids: list[str] = []
    for group_claims in claims.values():
        values = {value for _, _, value, _ in group_claims}
        if len(values) < 2:
            continue
        current_values = {value for _, _, value, is_current in group_claims if is_current}
        if not current_values and prefer_singleton_conflicts:
            current_values = singleton_conflict_values(group_claims)
        if not current_values:
            latest_timestamp = max(timestamp for _, timestamp, _, _ in group_claims)
            current_values = {value for _, timestamp, value, _ in group_claims if timestamp == latest_timestamp}
        stale_ids.extend(block_id for block_id, _, value, _ in group_claims if value not in current_values)
    block_order = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    return sorted(set(stale_ids), key=lambda block_id: block_order.get(block_id, len(block_order)))


def singleton_conflict_values(group_claims: list[tuple[str, str, str, bool]]) -> set[str]:
    counts: dict[str, int] = {}
    for _, _, value, _ in group_claims:
        counts[value] = counts.get(value, 0) + 1
    singletons = {value for value, count in counts.items() if count == 1}
    repeated = {value for value, count in counts.items() if count > 1}
    if len(singletons) == 1 and repeated:
        return singletons
    return set()


def detected_lineage_stale_ids(item: BenchmarkItem) -> list[str]:
    block_ids = {block.id for block in item.context_blocks}
    stale_ids: set[str] = set()
    for block in item.context_blocks:
        metadata = block.metadata
        for superseded_id in metadata_id_list(metadata.get("supersedes")):
            if superseded_id in block_ids:
                stale_ids.add(superseded_id)
        if any(current_id in block_ids for current_id in metadata_id_list(metadata.get("superseded_by"))):
            stale_ids.add(block.id)
    block_order = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    return sorted(stale_ids, key=lambda block_id: block_order.get(block_id, len(block_order)))


def detected_lineage_pairs(item: BenchmarkItem) -> list[tuple[str, str]]:
    block_ids = {block.id for block in item.context_blocks}
    block_order = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    pairs: dict[str, str] = {}
    for block in item.context_blocks:
        metadata = block.metadata
        for superseded_id in metadata_id_list(metadata.get("supersedes")):
            if superseded_id in block_ids and superseded_id not in pairs:
                pairs[superseded_id] = block.id
        for current_id in metadata_id_list(metadata.get("superseded_by")):
            if current_id in block_ids and block.id not in pairs:
                pairs[block.id] = current_id
    return sorted(pairs.items(), key=lambda pair: block_order.get(pair[0], len(block_order)))


def detected_lineage_current_ids(item: BenchmarkItem) -> list[str]:
    block_order = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    return sorted(
        {current_id for _, current_id in detected_lineage_pairs(item)},
        key=lambda block_id: block_order.get(block_id, len(block_order)),
    )


def detected_query_ref_stale_ids(item: BenchmarkItem) -> list[str]:
    target = query_revision_target(item.query)
    if target is None:
        return []

    target_repo, target_path, target_ref = target
    candidates: list[tuple[str, str]] = []
    has_current_ref = False
    for block in item.context_blocks:
        metadata = block.metadata
        repo = normalize_query_ref_part(metadata.get("repo"))
        path = normalize_query_ref_path(metadata.get("path"))
        ref = normalize_query_ref_part(metadata.get("ref"))
        if not repo or not path or not ref:
            continue
        if repo != target_repo or path != target_path:
            continue
        candidates.append((block.id, ref))
        if ref == target_ref:
            has_current_ref = True

    if not has_current_ref:
        return []

    block_order = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    stale_ids = {block_id for block_id, ref in candidates if ref != target_ref}
    return sorted(stale_ids, key=lambda block_id: block_order.get(block_id, len(block_order)))


def detected_text_timestamp_stale_ids(item: BenchmarkItem) -> list[str]:
    candidates = text_timestamp_candidate_blocks(item)
    if len(candidates) < 2:
        return []
    latest_timestamp = max(block.parsed_timestamp for block in candidates)
    if sum(block.parsed_timestamp == latest_timestamp for block in candidates) != 1:
        return []
    block_order = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    stale_ids = {block.id for block in candidates if block.parsed_timestamp != latest_timestamp}
    return sorted(stale_ids, key=lambda block_id: block_order.get(block_id, len(block_order)))


def detected_text_timestamp_pairs(item: BenchmarkItem) -> list[tuple[str, str]]:
    candidates = text_timestamp_candidate_blocks(item)
    if len(candidates) < 2:
        return []
    latest_timestamp = max(block.parsed_timestamp for block in candidates)
    current_blocks = [block for block in candidates if block.parsed_timestamp == latest_timestamp]
    if len(current_blocks) != 1:
        return []
    current_id = current_blocks[0].id
    block_order = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    pairs = [
        (block.id, current_id)
        for block in candidates
        if block.parsed_timestamp != latest_timestamp
    ]
    return sorted(pairs, key=lambda pair: block_order.get(pair[0], len(block_order)))


def detected_text_timestamp_current_ids(item: BenchmarkItem) -> list[str]:
    block_order = {block.id: idx for idx, block in enumerate(item.context_blocks)}
    return sorted(
        {current_id for _, current_id in detected_text_timestamp_pairs(item)},
        key=lambda block_id: block_order.get(block_id, len(block_order)),
    )


def text_timestamp_candidate_blocks(item: BenchmarkItem) -> list[Any]:
    phrases = query_option_phrases(item.query)
    if phrases:
        candidates = [
            block
            for block in item.context_blocks
            if block.text.strip()
            and max((phrase_text_coverage(phrase, block.text) for phrase in phrases), default=0.0) >= 0.5
        ]
        if len(candidates) >= 2:
            return candidates

    memory_key = query_memory_key(item.query)
    if memory_key:
        candidates = [
            block
            for block in item.context_blocks
            if block.text.strip() and memory_key in block_memory_keys(block.text)
        ]
        if len(candidates) >= 2:
            return candidates

    doc_blocks = [
        block
        for block in item.context_blocks
        if block.text.strip() and "doc_revision" in str(block.type)
    ]
    if len(doc_blocks) == 2:
        return doc_blocks
    return []


def query_memory_key(query: str) -> str:
    patterns = [
        r"\bmemory\s+key\s+[`'\"](?P<key>[^`'\"]+)[`'\"]",
        r"\bmemory\s+for\s+key\s+[`'\"](?P<key>[^`'\"]+)[`'\"]",
        r"\bkey\s+[`'\"](?P<key>[^`'\"]+)[`'\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return normalize_memory_key(match.group("key"))
    return ""


def block_memory_keys(text: str) -> set[str]:
    keys: set[str] = set()
    key_patterns = [
        r"\bMemory\s+(?:key|slot)\s*[:=]\s*([A-Za-z0-9_:-]+(?:\.[A-Za-z0-9_:-]+)*)",
        r"\bmemory_key\s*=\s*([A-Za-z0-9_:-]+(?:\.[A-Za-z0-9_:-]+)*)",
    ]
    for key_pattern in key_patterns:
        for match in re.finditer(key_pattern, text, flags=re.IGNORECASE):
            key = normalize_memory_key(match.group(1))
            if key:
                keys.add(key)
    return keys


def normalize_memory_key(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,'\"`")


def query_option_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    pattern = re.compile(
        r"(?P<code>[A-Za-z0-9_.%-]+)\s*=\s*(?P<literal>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")\.",
    )
    for match in pattern.finditer(query):
        literal = match.group("literal")
        try:
            value = ast.literal_eval(literal)
        except (SyntaxError, ValueError):
            value = literal.strip("\"'")
        if value:
            phrases.append(str(value))
    return phrases


def phrase_text_coverage(phrase: str, text: str) -> float:
    phrase_tokens = normalized_text_tokens(phrase)
    if not phrase_tokens:
        return 0.0
    text_counts: dict[str, int] = {}
    for token in normalized_text_tokens(text):
        text_counts[token] = text_counts.get(token, 0) + 1
    overlap = 0
    phrase_counts: dict[str, int] = {}
    for token in phrase_tokens:
        phrase_counts[token] = phrase_counts.get(token, 0) + 1
    for token, count in phrase_counts.items():
        overlap += min(count, text_counts.get(token, 0))
    return overlap / max(1, len(phrase_tokens))


def normalized_text_tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value).lower())


def query_revision_target(query: str) -> tuple[str, str, str] | None:
    normalized = re.sub(r"\s+", " ", query).strip()
    match = re.search(
        r"\bIn\s+(?P<repo>\S+)\s+(?P<path>[^,]+),.*?\bafter revision\s+(?P<ref>[^?\s]+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    repo = normalize_query_ref_part(match.group("repo"))
    path = normalize_query_ref_path(match.group("path"))
    ref = normalize_query_ref_part(match.group("ref"))
    if not repo or not path or not ref:
        return None
    return repo, path, ref


def normalize_query_ref_path(value: Any) -> str:
    text = normalize_query_ref_part(value)
    return text.strip("/")


def normalize_query_ref_part(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,'\"")


def metadata_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def extract_value_claim(block: Any) -> tuple[str, str, str] | None:
    text = block.text
    match = re.search(
        r"for (?P<entity>[^:]+): (?P<prop>.+?) (?:was|is) '(?P<value>[^']+)'",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return (
            normalize_claim_part(match.group("entity")),
            normalize_claim_part(match.group("prop")),
            normalize_claim_part(match.group("value")),
        )
    entity = block.metadata.get("entity")
    prop = block.metadata.get("property")
    value = block.metadata.get("value")
    if entity and prop and value:
        return normalize_claim_part(entity), normalize_claim_part(prop), normalize_claim_part(value)
    return None


def normalize_claim_part(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def has_current_text_cue(text: str) -> bool:
    normalized = text.lower()
    return "current decision" in normalized or "supersedes" in normalized


def random_non_stale_blocks(encoded: EncodedCase, item: BenchmarkItem) -> list[str]:
    target_tokens = token_count_for_ids(encoded, item.stale_ids)
    excluded = set(item.stale_ids) | set(item.gold_evidence_ids)
    candidates = [block_id for block_id in encoded.block_ranges if block_id not in excluded]
    rng_seed = int(hashlib.sha256(item.id.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(rng_seed)
    rng.shuffle(candidates)
    selected: list[str] = []
    selected_tokens = 0
    for block_id in candidates:
        selected.append(block_id)
        start, end = encoded.block_ranges[block_id]
        selected_tokens += end - start
        if selected_tokens >= target_tokens:
            break
    return selected


def generate_from_cache(
    model: Any,
    tokenizer: Any,
    cache: Any,
    prefix_len: int,
    suffix_ids: torch.Tensor,
    max_new_tokens: int,
) -> str:
    attention_mask = torch.ones((1, prefix_len + suffix_ids.shape[1]), dtype=torch.long, device=suffix_ids.device)
    with torch.no_grad():
        output = model(input_ids=suffix_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
    cache = output.past_key_values
    next_logits = output.logits[:, -1, :]
    generated: list[int] = []
    total_len = prefix_len + suffix_ids.shape[1]
    eos = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(next_logits, dim=-1).item())
        if eos is not None and next_id == eos:
            break
        generated.append(next_id)
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if "}" in decoded and "{" in decoded and decoded.rfind("}") > decoded.find("{"):
            break
        token = torch.tensor([[next_id]], dtype=torch.long, device=suffix_ids.device)
        total_len += 1
        attention_mask = torch.ones((1, total_len), dtype=torch.long, device=suffix_ids.device)
        with torch.no_grad():
            output = model(input_ids=token, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        next_logits = output.logits[:, -1, :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def token_count_for_ids(encoded: EncodedCase, block_ids: list[str]) -> int:
    total = 0
    for block_id in block_ids:
        if block_id in encoded.block_ranges:
            start, end = encoded.block_ranges[block_id]
            total += end - start
    return total


if __name__ == "__main__":
    main()
