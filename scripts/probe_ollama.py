#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.models.ollama_adapter import OllamaModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a local Ollama model with a short JSON task.")
    parser.add_argument("--model", default="qwen3:8b")
    args = parser.parse_args()
    model = OllamaModel(args.model, max_tokens=96, temperature=0.0)
    output = model.generate(
        prompt='Return JSON only: {"answer":"ok","evidence_ids":["b1"],"used_stale_fact":false,"confidence":1,"abstain":false,"reasoning_summary":"probe"}',
        item=None,  # type: ignore[arg-type]
        compression=None,  # type: ignore[arg-type]
    )
    print(output)


if __name__ == "__main__":
    main()
