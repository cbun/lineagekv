# Reproducibility

## Install

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"
```

Optional extras:

```bash
python3 -m pip install -e ".[datasets]"      # fetch optional external datasets
python3 -m pip install -e ".[local-models]"  # MLX and embedding baselines
python3 -m pip install -e ".[transformers]"  # raw editable-KV probes
```

## Fast Verification

```bash
python3 -m compileall src scripts
python3 -m pytest -q
```

## Public Artifact Rebuild

```bash
bash scripts/run_kv_public_repro.sh
```

This rebuilds:

- `data/public/stale_memory_repair_benchmark_v1.jsonl`
- `data/public/stale_memory_universality_traces_v1.jsonl`
- `results/kv/kv_public_release_targeting_audit.csv`
- `results/kv/kv_public_runtime_curve.csv`
- `results/kv/kv_public_7b32_paired_stats.csv`
- `results/kv/kv_public_qwen3_action32_paired_stats.csv`
- `results/kv/kv_public_qwen25_agent_memory80_paired_stats.csv`

The script uses checked-in compact source artifacts; it does not download model
weights or rerun expensive inference.

## Paper Build

The paper uses LaTeX with TikZ and PGFPlots.

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error stale_memory_kv_cache_repair.tex
pdflatex -interaction=nonstopmode -halt-on-error stale_memory_kv_cache_repair.tex
```

## Expensive Model Runs

The public repro path avoids heavy inference. The raw editable-KV experiments
need local model checkpoints and hardware-specific runtimes:

- MLX runs need `mlx-lm` and Apple Silicon/Metal.
- Transformers raw-cache runs need `torch`, `transformers`, and enough RAM/VRAM
  for the selected checkpoint.
- The Qwen3.5-9B local CPU smoke is intentionally a small probe, not a full
  CUDA replication.

The no-model public artifact rebuild is the default reproducibility path for
reviewers who do not have the same local checkpoints or hardware.
