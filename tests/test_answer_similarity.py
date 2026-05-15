from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_answer_similarity import classify, token_f1


def test_token_f1_accepts_wrapped_current_sentence() -> None:
    answer = "The current wording is: Fixed SlowBuffer support for res.send()."
    current = "Fixed SlowBuffer support for res.send()."
    stale = "Added support for res.contentType() literal"

    assert token_f1(answer, current) > 0.70
    assert token_f1(answer, stale) < 0.35
    assert classify(token_f1(answer, current), token_f1(answer, stale), 0.50, 0.08) == "current_like"


def test_token_f1_detects_stale_sentence() -> None:
    answer = "Added support for res.contentType() literal"
    current = "Fixed SlowBuffer support for res.send()."
    stale = "Added support for res.contentType() literal"

    assert classify(token_f1(answer, current), token_f1(answer, stale), 0.50, 0.08) == "stale_like"


def test_classifier_leaves_low_overlap_answers_ambiguous() -> None:
    answer = "This feature will be available in a future release."
    current = "Fixed SlowBuffer support for res.send()."
    stale = "Added support for res.contentType() literal"

    assert classify(token_f1(answer, current), token_f1(answer, stale), 0.50, 0.08) == "ambiguous_or_other"
