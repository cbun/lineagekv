from __future__ import annotations

import json
import re
from datetime import timedelta
from difflib import SequenceMatcher
from typing import Any

from context_rot.datasets.schema import BenchmarkItem, ContextBlock, GradingResult


def parse_model_json(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return _salvage_partial_json(text)
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return _salvage_partial_json(text)


def grade_output(item: BenchmarkItem, strategy: str, model_id: str, parsed_output: dict[str, Any]) -> GradingResult:
    if not isinstance(parsed_output, dict):
        parsed_output = {"answer": str(parsed_output), "evidence_ids": [], "used_stale_fact": False}
    answer = str(parsed_output.get("answer", ""))
    normalized_answer = _normalize(answer)
    target_value = _normalize(str(item.metadata.get("target_value", item.gold_answer)))
    stale_value = _normalize(str(item.metadata.get("stale_value", "")))
    gold = _normalize(item.gold_answer)
    evidence_ids = parsed_output.get("evidence_ids", [])
    if isinstance(evidence_ids, str):
        evidence_ids = [evidence_ids]
    evidence_set = {_normalize_evidence_id(str(eid)) for eid in evidence_ids}
    gold_set = set(item.gold_evidence_ids)
    stale_set = set(item.stale_ids)
    reported_stale = _as_bool(parsed_output.get("used_stale_fact", False))
    correct = _is_correct_answer(normalized_answer, target_value, gold, item.query) or _is_locomo_correct(
        item, normalized_answer
    )
    cited_stale = bool(evidence_set & stale_set)
    used_stale_fact = cited_stale or bool(stale_value and stale_value in normalized_answer)
    evidence_correct = gold_set.issubset(evidence_set) and not cited_stale
    contradiction_failure = bool(item.metadata.get("contradiction_count", 0)) and used_stale_fact
    abstain = bool(parsed_output.get("abstain", False))
    return GradingResult(
        item_id=item.id,
        strategy=strategy,
        model_id=model_id,
        correct=correct and not abstain,
        evidence_correct=evidence_correct,
        cited_stale_evidence=cited_stale,
        used_stale_fact=used_stale_fact,
        contradiction_failure=contradiction_failure,
        abstain_correct=None,
        metrics={
            "answer": answer,
            "evidence_ids": sorted(evidence_set),
            "gold_answer": item.gold_answer,
            "gold_evidence_ids": item.gold_evidence_ids,
            "reported_used_stale_fact": reported_stale,
        },
    )


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.%$ -]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_correct_answer(normalized_answer: str, target_value: str, gold: str, query: str = "") -> bool:
    candidates = [value for value in {target_value, gold} if value]
    for candidate in candidates:
        if candidate in normalized_answer:
            return True
        if len(normalized_answer) >= 4 and normalized_answer in candidate:
            return True
        if _numeric_value_matches(normalized_answer, candidate, query):
            return True
        candidate_tokens = _content_tokens(candidate)
        answer_tokens = _content_tokens(normalized_answer)
        if len(candidate_tokens) >= 3 and candidate_tokens.issubset(answer_tokens):
            return True
        if len(normalized_answer) >= 12 and SequenceMatcher(None, normalized_answer, candidate).ratio() >= 0.82:
            return True
    if normalized_answer in {"no", "nope", "not without approval"}:
        return any(
            phrase in candidate
            for candidate in candidates
            for phrase in ["require approval", "approval required", "requires approval"]
        )
    if normalized_answer in {"yes", "yes if directly requested"}:
        return any(
            phrase in candidate
            for candidate in candidates
            for phrase in ["allowed", "yes", "directly requested"]
        )
    return False


def _numeric_value_matches(answer: str, candidate: str, query: str = "") -> bool:
    query_terms = _numeric_query_terms(query)
    answer_mentions = _extract_numeric_mentions(answer)
    candidate_mentions = _extract_numeric_mentions(candidate)
    if not answer_mentions or not candidate_mentions:
        return False
    for answer_value, answer_context in answer_mentions:
        for candidate_value, candidate_context in candidate_mentions:
            if answer_value != candidate_value:
                continue
            if not query_terms:
                return True
            if query_terms & answer_context or query_terms & candidate_context:
                return True
    return False


def _extract_numeric_mentions(text: str) -> list[tuple[str, set[str]]]:
    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    mentions: list[tuple[str, set[str]]] = []
    tokens = re.findall(r"\$?\d[\d,]*(?:\.\d+)?|[a-z]+", text.lower())
    for idx, token in enumerate(tokens):
        value = None
        if re.fullmatch(r"\$?\d[\d,]*(?:\.\d+)?", token):
            value = token.replace("$", "").replace(",", "")
        elif token in number_words:
            value = str(number_words[token])
        if value is None:
            continue
        if value.endswith(".0"):
            value = value[:-2]
        context = {_unit_stem(part) for part in tokens[max(0, idx - 2) : idx] + tokens[idx + 1 : idx + 4]}
        mentions.append((value, context))
    for idx, (first, second) in enumerate(zip(tokens, tokens[1:])):
        if first in number_words and second in number_words and number_words[first] >= 20 and number_words[second] < 10:
            context = {_unit_stem(part) for part in tokens[max(0, idx - 2) : idx] + tokens[idx + 2 : idx + 5]}
            mentions.append((str(number_words[first] + number_words[second]), context))
    return mentions


def _numeric_query_terms(query: str) -> set[str]:
    stopwords = {
        "did",
        "different",
        "do",
        "does",
        "for",
        "have",
        "how",
        "i",
        "in",
        "many",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "total",
        "what",
        "when",
    }
    return {
        _unit_stem(token)
        for token in re.findall(r"[a-z]+", query.lower())
        if token not in stopwords and len(token) > 2
    }


def _unit_stem(token: str) -> str:
    token = token.strip().lower()
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "be",
        "for",
        "in",
        "is",
        "of",
        "on",
        "should",
        "the",
        "to",
        "with",
    }
    tokens = set()
    for token in text.split():
        token = token.strip(".,;:")
        if token in stopwords:
            continue
        tokens.add(_light_stem(token))
    return tokens


def _light_stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    return token


def _is_locomo_correct(item: BenchmarkItem, normalized_answer: str) -> bool:
    if item.domain != "external_locomo":
        return False
    gold = _normalize(item.gold_answer)
    if _token_f1(normalized_answer, gold) >= 0.66:
        return True
    return _relative_date_matches(item, normalized_answer)


def _token_f1(answer: str, gold: str) -> float:
    answer_tokens = _content_tokens(answer)
    gold_tokens = _content_tokens(gold)
    if not answer_tokens or not gold_tokens:
        return 0.0
    overlap = len(answer_tokens & gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _relative_date_matches(item: BenchmarkItem, normalized_answer: str) -> bool:
    gold_date = _gold_year_month_day(item.gold_answer)
    gold_year = _gold_year(item.gold_answer)
    if not gold_date and not gold_year:
        return False
    evidence_blocks = _gold_evidence_blocks(item)
    if "yesterday" in normalized_answer and gold_date:
        for block in evidence_blocks:
            if (block.parsed_timestamp.date() - timedelta(days=1)) == gold_date:
                return True
    if "last year" in normalized_answer and gold_year:
        return any((block.parsed_timestamp.year - 1) == gold_year for block in evidence_blocks)
    return False


def _gold_evidence_blocks(item: BenchmarkItem) -> list[ContextBlock]:
    gold_ids = set(item.gold_evidence_ids)
    blocks = [block for block in item.context_blocks if block.id in gold_ids]
    return blocks or item.context_blocks


def _gold_year_month_day(value: str):
    from datetime import datetime

    for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            pass
    return None


def _gold_year(value: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else None


def _normalize_evidence_id(evidence_id: str) -> str:
    evidence_id = evidence_id.strip()
    evidence_id = re.sub(r"^id=", "", evidence_id)
    evidence_id = evidence_id.strip("\"' ")
    return evidence_id


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "null", "none", "0", "no"}
    return bool(value)


def _salvage_partial_json(text: str) -> dict[str, Any]:
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)', text, re.DOTALL)
    evidence_match = re.search(r'"evidence_ids"\s*:\s*\[([^\]]*)', text, re.DOTALL)
    stale_match = re.search(r'"used_stale_fact"\s*:\s*([^,\n}]*)', text, re.DOTALL)
    evidence_ids: list[str] = []
    if evidence_match:
        evidence_ids = _parse_evidence_list(evidence_match.group(1))
    return {
        "answer": answer_match.group(1) if answer_match else text,
        "evidence_ids": evidence_ids,
        "used_stale_fact": _parse_partial_bool(stale_match.group(1)) if stale_match else False,
    }


def _parse_partial_bool(value: str) -> Any:
    value = value.strip().strip(",")
    if value.lower() in {"true", "false", "null"}:
        return value.lower() == "true"
    if value.startswith('"'):
        return value.strip('"')
    if value.startswith("{"):
        return True
    return value


def _parse_evidence_list(value: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"', value)
    if quoted:
        return quoted
    evidence_ids: list[str] = []
    for part in value.split(","):
        candidate = part.strip().strip("\"' ")
        candidate = re.sub(r"^id=", "", candidate)
        if re.fullmatch(r"[A-Za-z0-9_:-]+", candidate):
            evidence_ids.append(candidate)
    return evidence_ids
