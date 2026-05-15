#!/usr/bin/env python3
"""Fetch the bounded dataset bundle for the context-rot/attention-hygiene MVP.

This intentionally avoids the largest optional files:
- LongMemEval_M (~2.7GB)
- LongBench v2 (~465MB single JSON)
- LoCoMo MC10 transformed files (~250MB each)

Run from anywhere:
    python3 scripts/fetch_datasets.py
"""
from pathlib import Path
from huggingface_hub import snapshot_download
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXTERNAL = ROOT / "external"
RAW.mkdir(parents=True, exist_ok=True)
EXTERNAL.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("kellyhongg/cleaned-longmemeval-s", ["README.md", "longmemeval_s_cleaned.csv"]),
    ("Percena/locomo-mc10", ["README.md", "LICENSE", "raw/locomo10.json"]),
    ("amodaresi/NoLiMa", ["README.md", "LICENSE", "needlesets/*", "haystack/rand_shuffle/*.txt", "haystack/rand_shuffle_long/*.txt"]),
    ("THUDM/LongBench", ["README.md", "LongBench.py", "data.zip"]),
]

for repo, patterns in DATASETS:
    dest = RAW / repo.replace("/", "__")
    print(f"\n== {repo}")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=patterns,
    )

lb = RAW / "THUDM__LongBench"
if (lb / "data.zip").exists() and not (lb / "data").exists():
    subprocess.run(["unzip", "-q", "data.zip"], cwd=lb, check=True)

ruler = EXTERNAL / "RULER"
if not (ruler / ".git").exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/RULER.git", str(ruler)], check=True)
else:
    subprocess.run(["git", "-C", str(ruler), "pull", "--ff-only"], check=True)

print(f"\nDone: {ROOT}")
