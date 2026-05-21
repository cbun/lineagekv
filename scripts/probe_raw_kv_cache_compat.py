#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_rot.datasets.io import read_jsonl
from kv_value_ablation_probe import apply_policy, clone_cache, encode_case, parse_torch_dtype


DEFAULT_POLICIES = (
    "zero_text_timestamp_conflict_values_layers_8_23,"
    "zero_text_timestamp_conflict_keys_values_layers_8_23,"
    "zero_text_timestamp_conflict_values_layers_8_23_boost_current_factor_1_0"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe raw checkpoint KV-cache edit compatibility without decoding.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="data/generated/agent_memory_action_test32.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    item = read_jsonl(ROOT / args.dataset)[args.item_index]
    started = time.perf_counter()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
        dtype=parse_torch_dtype(args.dtype),
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    model.eval()
    load_ms = (time.perf_counter() - started) * 1000

    encoded = encode_case(tokenizer, item, render_mode="temporal_blind", block_id_mode="neutral", prompt_style="plain")
    with torch.no_grad():
        started = time.perf_counter()
        prefill = model(input_ids=encoded.prefix_ids, use_cache=True)
        prefill_ms = (time.perf_counter() - started) * 1000

    cache = prefill.past_key_values
    layers = list(getattr(cache, "layers", []))
    editable_layers = editable_layer_records(layers)
    policy_results = []
    for policy in [value for value in args.policies.split(",") if value]:
        started = time.perf_counter()
        edit_cache = clone_cache(cache)
        clone_ms = (time.perf_counter() - started) * 1000
        try:
            started = time.perf_counter()
            apply_policy(edit_cache, encoded, item, policy)
            edit_ms = (time.perf_counter() - started) * 1000
            policy_results.append({"policy": policy, "ok": True, "clone_ms": clone_ms, "edit_ms": edit_ms})
        except Exception as exc:
            policy_results.append(
                {
                    "policy": policy,
                    "ok": False,
                    "clone_ms": clone_ms,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    result = {
        "model": args.model,
        "config_model_type": getattr(config, "model_type", None),
        "text_model_type": text_model_type(config),
        "load_ms": load_ms,
        "prefill_ms": prefill_ms,
        "prefix_tokens": int(encoded.prefix_ids.shape[1]),
        "suffix_tokens": int(encoded.suffix_ids.shape[1]),
        "cache_class": type(cache).__name__,
        "cache_layers": len(layers),
        "editable_layer_count": len(editable_layers),
        "editable_layers": editable_layers,
        "policy_results": policy_results,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def editable_layer_records(layers: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):
            records.append(
                {
                    "idx": idx,
                    "class": type(layer).__name__,
                    "keys": list(keys.shape),
                    "values": list(values.shape),
                }
            )
    return records


def text_model_type(config: Any) -> str | None:
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return None
    if isinstance(text_config, dict):
        value = text_config.get("model_type")
    else:
        value = getattr(text_config, "model_type", None)
    return str(value) if value is not None else None


if __name__ == "__main__":
    main()
