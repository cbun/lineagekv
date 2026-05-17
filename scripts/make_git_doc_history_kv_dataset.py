#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import write_jsonl
from context_rot.datasets.schema import BenchmarkItem, ContextBlock
from make_git_history_kv_dataset import display_ref, git, slug


@dataclass(frozen=True)
class RevisionRef:
    ref: str
    timestamp: str


@dataclass(frozen=True)
class SentenceChange:
    path: str
    old_ref: RevisionRef
    new_ref: RevisionRef
    old_sentence: str
    new_sentence: str
    similarity: float


DEFAULT_PATHS = ["README.md", "docs/README.md", "CHANGELOG.md"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KV stale/current items from natural-language documentation sentence changes."
    )
    parser.add_argument("--repo", required=True, help="Path to a local git repository.")
    parser.add_argument("--repo-name", default="", help="Human-readable repo name. Defaults to repo directory name.")
    parser.add_argument("--output", default="data/generated/git_doc_history_lineage.jsonl")
    parser.add_argument("--paths", default=",".join(DEFAULT_PATHS), help="Comma-separated markdown paths to scan.")
    parser.add_argument("--max-items", type=int, default=40)
    parser.add_argument("--max-changes-per-transition", type=int, default=2)
    parser.add_argument("--variants", default="current_first,stale_first")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    paths = [path for path in args.paths.split(",") if path]
    variants = [variant for variant in args.variants.split(",") if variant]
    items = build_doc_items(
        repo=repo,
        repo_name=args.repo_name or repo.name,
        paths=paths,
        variants=variants,
        max_items=args.max_items,
        max_changes_per_transition=args.max_changes_per_transition,
    )
    count = write_jsonl(items, ROOT / args.output)
    print(f"wrote {count} git-doc lineage items to {ROOT / args.output}")


def build_doc_items(
    repo: Path,
    repo_name: str,
    paths: list[str],
    variants: list[str],
    max_items: int,
    max_changes_per_transition: int = 2,
) -> list[BenchmarkItem]:
    refs = tag_refs(repo)
    seen: set[tuple[str, str, str]] = set()
    items: list[BenchmarkItem] = []
    item_idx = 0
    for old_ref, new_ref in zip(refs, refs[1:]):
        for path in paths:
            for change in sentence_changes(repo, old_ref, new_ref, path)[:max_changes_per_transition]:
                key = (change.path, change.old_sentence, change.new_sentence)
                if key in seen:
                    continue
                seen.add(key)
                for variant in variants:
                    item_idx += 1
                    items.append(build_doc_item(item_idx, repo_name, change, variant))
                    if len(items) >= max_items:
                        return items
    return items


def tag_refs(repo: Path) -> list[RevisionRef]:
    rows = git(
        repo,
        "for-each-ref",
        "--sort=creatordate",
        "--format=%(refname:short)%09%(creatordate:iso-strict)",
        "refs/tags",
    )
    refs = []
    for row in rows.splitlines():
        if not row.strip():
            continue
        ref, timestamp = row.split("\t", 1)
        refs.append(RevisionRef(ref=ref, timestamp=timestamp))
    return refs


def sentence_changes(repo: Path, old_ref: RevisionRef, new_ref: RevisionRef, path: str) -> list[SentenceChange]:
    try:
        diff = git(repo, "diff", "--unified=0", old_ref.ref, new_ref.ref, "--", path)
    except subprocess.CalledProcessError:
        return []

    changes: list[SentenceChange] = []
    removed: list[str] = []
    added: list[str] = []
    for line in diff.splitlines():
        if line.startswith("@@"):
            changes.extend(pair_hunk(path, old_ref, new_ref, removed, added))
            removed = []
            added = []
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            cleaned = clean_markdown_line(line[1:])
            if valid_sentence(cleaned):
                removed.append(cleaned)
        elif line.startswith("+"):
            cleaned = clean_markdown_line(line[1:])
            if valid_sentence(cleaned):
                added.append(cleaned)
    changes.extend(pair_hunk(path, old_ref, new_ref, removed, added))
    return changes


def pair_hunk(
    path: str,
    old_ref: RevisionRef,
    new_ref: RevisionRef,
    removed: list[str],
    added: list[str],
) -> list[SentenceChange]:
    changes: list[SentenceChange] = []
    used_removed: set[int] = set()
    for new_sentence in added:
        best_idx = None
        best_score = 0.0
        for idx, old_sentence in enumerate(removed):
            if idx in used_removed:
                continue
            score = sentence_similarity(old_sentence, new_sentence)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None or best_score < 0.25 or best_score >= 0.96:
            continue
        used_removed.add(best_idx)
        changes.append(
            SentenceChange(
                path=path,
                old_ref=old_ref,
                new_ref=new_ref,
                old_sentence=removed[best_idx],
                new_sentence=new_sentence,
                similarity=best_score,
            )
        )
    return changes


def clean_markdown_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^#{1,6}\s*", "", line)
    line = re.sub(r"^[-*+]\s+", "", line)
    line = re.sub(r"^\d+[.)]\s+", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def valid_sentence(sentence: str) -> bool:
    if not 35 <= len(sentence) <= 220:
        return False
    lower = sentence.lower()
    if any(marker in lower for marker in ["http://", "https://", "npm install", "```"]):
        return False
    if any(char in sentence for char in "{}<>|"):
        return False
    if sentence.count("/") > 4:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", sentence)
    return len(words) >= 6


def sentence_similarity(left: str, right: str) -> float:
    token_left = set(re.findall(r"[a-z0-9]+", left.lower()))
    token_right = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not token_left or not token_right:
        return 0.0
    overlap = len(token_left & token_right) / len(token_left | token_right)
    return 0.5 * overlap + 0.5 * SequenceMatcher(None, left, right).ratio()


def build_doc_item(item_idx: int, repo_name: str, change: SentenceChange, variant: str) -> BenchmarkItem:
    stale_id = f"d{item_idx:03d}_stale"
    current_id = f"d{item_idx:03d}_current"
    stale = doc_block(
        block_id=stale_id,
        repo_name=repo_name,
        path=change.path,
        ref=change.old_ref,
        text=change.old_sentence,
        lineage={"superseded_by": [current_id]},
    )
    current = doc_block(
        block_id=current_id,
        repo_name=repo_name,
        path=change.path,
        ref=change.new_ref,
        text=change.new_sentence,
        lineage={"supersedes": [stale_id]},
    )
    if variant == "current_first":
        blocks = [current, stale]
    elif variant == "stale_first":
        blocks = [stale, current]
    else:
        raise KeyError(variant)
    query = (
        f"In the {repo_name} documentation file {change.path}, what is the current wording of the "
        f"changed sentence after revision {display_ref(change.new_ref.ref)}?"
    )
    return BenchmarkItem(
        id=f"git_doc_{slug(repo_name)}_{slug(change.path)}_{item_idx:03d}_{variant}",
        domain="git_doc_history",
        question_type="doc_revision_lineage",
        context_blocks=blocks,
        query=query,
        gold_answer=change.new_sentence,
        gold_evidence_ids=[current_id],
        stale_ids=[stale_id],
        distractor_ids=[],
        metadata={
            "repo_name": repo_name,
            "path": change.path,
            "variant": variant,
            "distractor_ratio": 0.0,
            "target_value": change.new_sentence,
            "stale_value": change.old_sentence,
            "stale_ref": change.old_ref.ref,
            "current_ref": change.new_ref.ref,
            "similarity": change.similarity,
            "lineage_source": "git_doc_history",
        },
    )


def doc_block(
    block_id: str,
    repo_name: str,
    path: str,
    ref: RevisionRef,
    text: str,
    lineage: dict[str, list[str]],
) -> ContextBlock:
    return ContextBlock(
        id=block_id,
        timestamp=ref.timestamp,
        type="doc_revision",
        trust="high",
        status="unknown",
        text=f"{text}",
        metadata={
            "repo": repo_name,
            "path": path,
            "ref": ref.ref,
            **lineage,
        },
    )


if __name__ == "__main__":
    main()
