"""
Tests for the decision pipeline and Gemini output validation.
All Gemini calls are mocked — no live API calls needed.
"""
import json
import pytest
from unittest.mock import patch

from src.decision import _parse_and_validate, run_decision_pipeline
from src.retrieval import RetrievedChunk
from src.schemas import AIDecision

_CHUNKS = [RetrievedChunk(source="damaged_goods.md", text="Policy text.", score=0.9)]


# ── _parse_and_validate ───────────────────────────────────────────────────────

def _valid_json(action="APPROVE_REFUND", status="FINAL", confidence=0.9):
    return json.dumps({
        "action": action, "status": status, "confidence": confidence,
        "reason": "Policy satisfied.", "question": None,
        "required_information": [], "sources": ["damaged_goods.md"],
    })


def test_parse_valid_final():
    r = _parse_and_validate(_valid_json(), _CHUNKS)
    assert isinstance(r, AIDecision)
    assert r.action == "APPROVE_REFUND"
    assert r.status == "FINAL"


def test_parse_strips_markdown_fences():
    raw = "```json\n" + _valid_json() + "\n```"
    r = _parse_and_validate(raw, _CHUNKS)
    assert r.action == "APPROVE_REFUND"


def test_parse_in_progress_with_question():
    data = json.dumps({
        "action": "REQUEST_MORE_INFORMATION", "status": "IN_PROGRESS",
        "confidence": 0.8, "reason": "Need more details.",
        "question": "What is the order value?",
        "required_information": ["order_value"], "sources": [],
    })
    r = _parse_and_validate(data, _CHUNKS)
    assert r.status == "IN_PROGRESS"
    assert r.question == "What is the order value?"


def test_parse_malformed_json_raises():
    with pytest.raises(ValueError, match="non-JSON"):
        _parse_and_validate("not json at all {{{}}", _CHUNKS)


def test_parse_invalid_status_raises():
    bad = json.dumps({
        "action": "APPROVE_REFUND", "status": "UNKNOWN",
        "confidence": 0.9, "reason": "ok", "question": None,
        "required_information": [], "sources": [],
    })
    with pytest.raises(ValueError):
        _parse_and_validate(bad, _CHUNKS)


def test_parse_confidence_too_high_raises():
    bad = _valid_json(confidence=1.5)
    with pytest.raises(ValueError):
        _parse_and_validate(bad, _CHUNKS)


def test_parse_confidence_negative_raises():
    bad = _valid_json(confidence=-0.1)
    with pytest.raises(ValueError):
        _parse_and_validate(bad, _CHUNKS)


def test_parse_needs_more_information_accepted():
    data = json.dumps({
        "action": "NEEDS_MORE_INFORMATION", "status": "IN_PROGRESS",
        "confidence": 0.6, "reason": "Missing details.", "question": "Please clarify.",
        "required_information": ["product_type"], "sources": [],
    })
    r = _parse_and_validate(data, _CHUNKS)
    assert r.action == "NEEDS_MORE_INFORMATION"


def test_parse_backfills_sources_from_chunks():
    data = json.dumps({
        "action": "APPROVE_REFUND", "status": "FINAL",
        "confidence": 0.8, "reason": "ok", "question": None,
        "required_information": [], "sources": [],
    })
    r = _parse_and_validate(data, _CHUNKS)
    assert "damaged_goods.md" in r.sources


def test_parse_action_uppercased():
    data = json.dumps({
        "action": "approve_refund", "status": "FINAL",
        "confidence": 0.8, "reason": "ok", "question": None,
        "required_information": [], "sources": ["refunds.md"],
    })
    r = _parse_and_validate(data, _CHUNKS)
    assert r.action == "APPROVE_REFUND"


# ── run_decision_pipeline ─────────────────────────────────────────────────────

def test_pipeline_returns_decision():
    mock_raw = _valid_json()
    with (
        patch("src.decision.retrieve_relevant_chunks", return_value=_CHUNKS),
        patch("src.decision.generate_decision", return_value=mock_raw),
    ):
        r = run_decision_pipeline("My product is damaged.")
    assert r.action == "APPROVE_REFUND"


def test_pipeline_with_history():
    mock_raw = json.dumps({
        "action": "APPROVE_REFUND", "status": "FINAL",
        "confidence": 0.95, "reason": "Photos confirm damage.",
        "question": None, "required_information": [], "sources": ["damaged_goods.md"],
    })
    history = [
        {"role": "user", "content": "My product is damaged."},
        {"role": "assistant", "content": "Please provide the order value and photos."},
        {"role": "user", "content": "Order value is ₹3,500."},
    ]
    with (
        patch("src.decision.retrieve_relevant_chunks", return_value=_CHUNKS),
        patch("src.decision.generate_decision", return_value=mock_raw),
    ):
        r = run_decision_pipeline("My product is damaged.", conversation_history=history)
    assert r.status == "FINAL"


def test_pipeline_propagates_gemini_error():
    with (
        patch("src.decision.retrieve_relevant_chunks", return_value=_CHUNKS),
        patch("src.decision.generate_decision", side_effect=RuntimeError("API down")),
    ):
        with pytest.raises(RuntimeError, match="API down"):
            run_decision_pipeline("My product is damaged.")


def test_pipeline_raises_on_invalid_response():
    with (
        patch("src.decision.retrieve_relevant_chunks", return_value=_CHUNKS),
        patch("src.decision.generate_decision", return_value="{{invalid}}"),
    ):
        with pytest.raises(ValueError):
            run_decision_pipeline("My product is damaged.")
