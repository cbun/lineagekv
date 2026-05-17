#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_rot.datasets.io import read_jsonl, write_jsonl
from context_rot.datasets.schema import BenchmarkItem


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "then",
    "these",
    "this",
    "to",
    "was",
    "when",
    "with",
    "you",
}

GENERIC_CONTENT_TOKENS = {
    "add",
    "directly",
    "fix",
    "follow",
    "instead",
    "support",
    "update",
    "use",
    "work",
}


@dataclass(frozen=True)
class PhraseAudit:
    item: BenchmarkItem
    issues: list[str]
    sentence_token_f1: float
    sentence_char_ratio: float
    phrase_char_ratio: float
    target_content_tokens: int
    stale_content_tokens: int

    @property
    def passed(self) -> bool:
        return not self.issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and optionally filter git-doc phrase lineage items for revision-like phrase quality."
    )
    parser.add_argument("--input", action="append", required=True, help="Input phrase JSONL. Repeatable.")
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--filtered-output")
    parser.add_argument("--per-input-limit", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    audits: list[PhraseAudit] = []
    for input_path in args.input:
        input_audits = [audit_item(item) for item in read_jsonl(ROOT / input_path)]
        if args.per_input_limit:
            input_audits = input_audits[: args.per_input_limit]
        audits.extend(input_audits)
    if args.max_items:
        audits = audits[: args.max_items]

    write_audit_csv(audits, ROOT / args.audit_output)
    passed_items = [audit.item for audit in audits if audit.passed]
    if args.filtered_output:
        count = write_jsonl((with_quality_metadata(audit) for audit in audits if audit.passed), ROOT / args.filtered_output)
        print(f"wrote {count} filtered phrase items to {ROOT / args.filtered_output}")
    print_summary(audits, passed_items)


def audit_item(item: BenchmarkItem) -> PhraseAudit:
    target_phrase = str(item.metadata.get("target_value", item.gold_answer))
    stale_phrase = str(item.metadata.get("stale_value", ""))
    target_sentence = str(item.metadata.get("sentence_target_value", ""))
    stale_sentence = str(item.metadata.get("sentence_stale_value", ""))

    sentence_f1 = token_f1(content_tokens(target_sentence), content_tokens(stale_sentence))
    sentence_char_ratio = char_ratio(target_sentence, stale_sentence)
    phrase_char_ratio = char_ratio(target_phrase, stale_phrase)
    target_content = content_tokens(target_phrase)
    stale_content = content_tokens(stale_phrase)
    issues: list[str] = []

    if sentence_f1 < 0.45 and sentence_char_ratio < 0.62:
        issues.append("low_sentence_overlap")
    if not target_content:
        issues.append("generic_target_phrase")
    if not stale_content:
        issues.append("generic_stale_phrase")
    if len(target_content) == 1 and next(iter(target_content)) in GENERIC_CONTENT_TOKENS:
        issues.append("generic_target_phrase")
    if len(stale_content) == 1 and next(iter(stale_content)) in GENERIC_CONTENT_TOKENS:
        issues.append("generic_stale_phrase")
    if all(token.isdigit() for token in target_content) or all(token.isdigit() for token in stale_content):
        issues.append("numeric_only_phrase")
    if len(words(target_phrase)) > 8 or len(words(stale_phrase)) > 8:
        issues.append("phrase_too_long")
    if target_phrase.lower() == stale_phrase.lower():
        issues.append("identical_phrase")
    if phrase_char_ratio > 0.93 and target_phrase.lower() != stale_phrase.lower():
        issues.append("near_identical_phrase")

    return PhraseAudit(
        item=item,
        issues=issues,
        sentence_token_f1=sentence_f1,
        sentence_char_ratio=sentence_char_ratio,
        phrase_char_ratio=phrase_char_ratio,
        target_content_tokens=len(target_content),
        stale_content_tokens=len(stale_content),
    )


def with_quality_metadata(audit: PhraseAudit) -> BenchmarkItem:
    return audit.item.model_copy(
        update={
            "metadata": {
                **audit.item.metadata,
                "phrase_quality_pass": audit.passed,
                "phrase_sentence_token_f1": round(audit.sentence_token_f1, 4),
                "phrase_sentence_char_ratio": round(audit.sentence_char_ratio, 4),
                "phrase_char_ratio": round(audit.phrase_char_ratio, 4),
            }
        }
    )


def write_audit_csv(audits: Iterable[PhraseAudit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "passed",
        "issues",
        "repo_name",
        "path",
        "target_phrase",
        "stale_phrase",
        "sentence_token_f1",
        "sentence_char_ratio",
        "phrase_char_ratio",
        "target_content_tokens",
        "stale_content_tokens",
        "target_sentence",
        "stale_sentence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for audit in audits:
            item = audit.item
            writer.writerow(
                {
                    "item_id": item.id,
                    "passed": audit.passed,
                    "issues": " ".join(audit.issues),
                    "repo_name": item.metadata.get("repo_name", ""),
                    "path": item.metadata.get("path", ""),
                    "target_phrase": item.metadata.get("target_value", item.gold_answer),
                    "stale_phrase": item.metadata.get("stale_value", ""),
                    "sentence_token_f1": f"{audit.sentence_token_f1:.4f}",
                    "sentence_char_ratio": f"{audit.sentence_char_ratio:.4f}",
                    "phrase_char_ratio": f"{audit.phrase_char_ratio:.4f}",
                    "target_content_tokens": audit.target_content_tokens,
                    "stale_content_tokens": audit.stale_content_tokens,
                    "target_sentence": item.metadata.get("sentence_target_value", ""),
                    "stale_sentence": item.metadata.get("sentence_stale_value", ""),
                }
            )
    print(f"wrote audit CSV to {path}")


def print_summary(audits: list[PhraseAudit], passed_items: list[BenchmarkItem]) -> None:
    issue_counts: dict[str, int] = {}
    for audit in audits:
        for issue in audit.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    print("audited", len(audits), "passed", len(passed_items), "failed", len(audits) - len(passed_items))
    if issue_counts:
        print("issues", " ".join(f"{key}={value}" for key, value in sorted(issue_counts.items())))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9._'-]*", text)


def content_tokens(text: str) -> set[str]:
    tokens = set()
    for token in words(text.lower()):
        token = token.strip("'")
        if token in STOPWORDS:
            continue
        tokens.add(light_stem(token))
    return tokens


def light_stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def token_f1(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2 * precision * recall / (precision + recall)


def char_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def normalize(text: str) -> str:
    return " ".join(words(text.lower()))


if __name__ == "__main__":
    main()
