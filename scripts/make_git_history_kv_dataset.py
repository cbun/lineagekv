#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import write_jsonl
from context_rot.datasets.schema import BenchmarkItem, ContextBlock


@dataclass(frozen=True)
class FactSpec:
    path: str
    key_path: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.path}::{'.'.join(self.key_path)}"


@dataclass(frozen=True)
class FactRevision:
    commit: str
    timestamp: str
    value: str


DEFAULT_SPECS = [
    "package.json::version",
    "package.json::engines.node",
    "pyproject.toml::project.version",
    "pyproject.toml::project.requires-python",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KV stale/current benchmark items from real git revision history."
    )
    parser.add_argument("--repo", required=True, help="Path to a local git repository.")
    parser.add_argument("--output", default="data/generated/git_history_lineage.jsonl")
    parser.add_argument("--repo-name", default="", help="Human-readable repo name. Defaults to repo directory name.")
    parser.add_argument(
        "--specs",
        default=",".join(DEFAULT_SPECS),
        help="Comma-separated specs of the form path::key.path.",
    )
    parser.add_argument("--max-items", type=int, default=40)
    parser.add_argument("--max-events-per-spec", type=int, default=20)
    parser.add_argument(
        "--ref-mode",
        choices=["commits", "tags"],
        default="commits",
        help="Use every commit touching the file, or only git tags sorted by creation date.",
    )
    parser.add_argument(
        "--variants",
        default="current_first,stale_first",
        help="Comma-separated context-order variants.",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    specs = [parse_spec(text) for text in args.specs.split(",") if text]
    variants = [value for value in args.variants.split(",") if value]
    items = build_items(
        repo=repo,
        repo_name=args.repo_name or repo.name,
        specs=specs,
        variants=variants,
        max_items=args.max_items,
        max_events_per_spec=args.max_events_per_spec,
        ref_mode=args.ref_mode,
    )
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} git-history lineage items to {ROOT / args.output}")


def build_items(
    repo: Path,
    repo_name: str,
    specs: list[FactSpec],
    variants: list[str],
    max_items: int,
    max_events_per_spec: int,
    ref_mode: str = "commits",
) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    item_idx = 0
    for spec in specs:
        revisions = fact_revisions(repo, spec, ref_mode=ref_mode)
        transitions = list(zip(revisions, revisions[1:]))[-max_events_per_spec:]
        for stale_revision, current_revision in transitions:
            if stale_revision.value == current_revision.value:
                continue
            for variant in variants:
                item_idx += 1
                items.append(
                    build_item(
                        item_idx=item_idx,
                        repo_name=repo_name,
                        spec=spec,
                        stale_revision=stale_revision,
                        current_revision=current_revision,
                        variant=variant,
                    )
                )
                if len(items) >= max_items:
                    return items
    return items


def fact_revisions(repo: Path, spec: FactSpec, ref_mode: str = "commits") -> list[FactRevision]:
    if ref_mode == "commits":
        log_rows = git(repo, "log", "--all", "--date-order", "--reverse", "--format=%H%x09%cI", "--", spec.path)
    elif ref_mode == "tags":
        log_rows = git(
            repo,
            "for-each-ref",
            "--sort=creatordate",
            "--format=%(refname:short)%09%(creatordate:iso-strict)",
            "refs/tags",
        )
    else:
        raise KeyError(ref_mode)
    revisions: list[FactRevision] = []
    last_value: str | None = None
    for row in log_rows.splitlines():
        if not row.strip():
            continue
        commit, timestamp = row.split("\t", 1)
        value = value_at_commit(repo, commit, spec)
        if value is None or value == last_value:
            continue
        revisions.append(FactRevision(commit=commit, timestamp=timestamp, value=value))
        last_value = value
    return revisions


def value_at_commit(repo: Path, commit: str, spec: FactSpec) -> str | None:
    try:
        raw = git(repo, "show", f"{commit}:{spec.path}")
    except subprocess.CalledProcessError:
        return None
    try:
        if spec.path.endswith(".json"):
            data = json.loads(raw)
        elif spec.path.endswith(".toml"):
            data = tomllib.loads(raw)
        else:
            return None
    except (json.JSONDecodeError, tomllib.TOMLDecodeError):
        return None
    value: Any = data
    for part in spec.key_path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def build_item(
    item_idx: int,
    repo_name: str,
    spec: FactSpec,
    stale_revision: FactRevision,
    current_revision: FactRevision,
    variant: str,
) -> BenchmarkItem:
    safe_repo = slug(repo_name)
    safe_key = slug(spec.label)
    stale_id = f"g{item_idx:03d}_stale"
    current_id = f"g{item_idx:03d}_current"
    entity = f"{repo_name} {spec.path}"
    prop = ".".join(spec.key_path)
    stale = revision_block(
        block_id=stale_id,
        repo_name=repo_name,
        spec=spec,
        revision=stale_revision,
        entity=entity,
        prop=prop,
        status="unknown",
        lineage={"superseded_by": [current_id]},
    )
    current = revision_block(
        block_id=current_id,
        repo_name=repo_name,
        spec=spec,
        revision=current_revision,
        entity=entity,
        prop=prop,
        status="unknown",
        lineage={"supersedes": [stale_id]},
    )
    if variant == "current_first":
        blocks = [current, stale]
    elif variant == "stale_first":
        blocks = [stale, current]
    else:
        raise KeyError(variant)
    query = (
        f"According to the git revision history for {repo_name}, what is the current value of "
        f"{spec.label} after revision {display_ref(current_revision.commit)}?"
    )
    return BenchmarkItem(
        id=f"git_history_{safe_repo}_{safe_key}_{item_idx:03d}_{variant}",
        domain="git_history",
        question_type="revision_lineage",
        context_blocks=blocks,
        query=query,
        gold_answer=current_revision.value,
        gold_evidence_ids=[current_id],
        stale_ids=[stale_id],
        distractor_ids=[],
        metadata={
            "repo_name": repo_name,
            "path": spec.path,
            "key_path": ".".join(spec.key_path),
            "variant": variant,
            "distractor_ratio": 0.0,
            "target_value": current_revision.value,
            "stale_value": stale_revision.value,
            "stale_commit": stale_revision.commit,
            "current_commit": current_revision.commit,
            "stale_ref": stale_revision.commit,
            "current_ref": current_revision.commit,
            "lineage_source": "git_history",
        },
    )


def revision_block(
    block_id: str,
    repo_name: str,
    spec: FactSpec,
    revision: FactRevision,
    entity: str,
    prop: str,
    status: str,
    lineage: dict[str, list[str]],
) -> ContextBlock:
    metadata = {
        "repo": repo_name,
        "path": spec.path,
        "key_path": ".".join(spec.key_path),
        "commit": revision.commit,
        "entity": entity,
        "property": prop,
        "value": revision.value,
        **lineage,
    }
    return ContextBlock(
        id=block_id,
        timestamp=revision.timestamp,
        type="git_revision",
        trust="high",
        status=status,
        text=(
            f"Git revision fact for {repo_name} at revision {display_ref(revision.commit)}: "
            f"{spec.label} was '{revision.value}'."
        ),
        metadata=metadata,
    )


def parse_spec(text: str) -> FactSpec:
    if "::" not in text:
        raise ValueError(f"Spec must have form path::key.path: {text}")
    path, key_path = text.split("::", 1)
    parts = tuple(part for part in key_path.split(".") if part)
    if not path or not parts:
        raise ValueError(f"Spec must have form path::key.path: {text}")
    return FactSpec(path=path, key_path=parts)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL)


def slug(value: str) -> str:
    output = []
    for char in value.lower():
        if char.isalnum():
            output.append(char)
        elif output and output[-1] != "_":
            output.append("_")
    return "".join(output).strip("_")[:80]


def display_ref(ref: str) -> str:
    if len(ref) >= 12 and all(char in "0123456789abcdefABCDEF" for char in ref):
        return ref[:12]
    return ref


if __name__ == "__main__":
    main()
