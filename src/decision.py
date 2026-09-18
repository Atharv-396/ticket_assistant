"""
Multi-turn decision pipeline: RAG → Gemini → Pydantic validation.

run_decision_pipeline() is called:
  - When a NEW ticket is created (no prior history).
  - When the user sends a FOLLOW-UP message (full history provided).
  - After evidence is uploaded (evidence metadata included).

The pipeline always:
  1. Retrieves relevant policy chunks for the combined context.
  2. Formats the prompt with conversation + evidence + policy.
  3. Calls Gemini.
  4. Validates and returns the structured AIDecision.

An unvalidated LLM response is NEVER stored.
"""
import json
import logging
import re
from typing import Optional

from src.retrieval import format_chunks_for_prompt, retrieve_relevant_chunks
from src.schemas import AIDecision
from src.services.gemini import generate_decision

logger = logging.getLogger(__name__)


def run_decision_pipeline(
    initial_message: str,
    conversation_history: Optional[list[dict]] = None,
    evidence_files: Optional[list[dict]] = None,
) -> AIDecision:
    """
    Execute the full RAG → Gemini → validate pipeline.

    Parameters
    ----------
    initial_message : str
        The customer's original complaint (used as the primary retrieval query).
    conversation_history : list[dict], optional
        All previous turns: [{"role": "user"|"assistant", "content": "..."}]
        Include every turn so Gemini has full context.
    evidence_files : list[dict], optional
        Evidence metadata: [{"filename": "photo.jpg", "content_type": "image/jpeg"}]

    Returns
    -------
    AIDecision
        Fully validated Pydantic model ready for database storage.

    Raises
    ------
    RuntimeError  — Gemini unavailable or unrecoverable API error.
    ValueError    — Gemini returned malformed/invalid output.
    FileNotFoundError — Embeddings not yet ingested.
    """
    history = conversation_history or []

    # Step 1: Build a rich retrieval query from all available context
    query = _build_retrieval_query(initial_message, history)
    chunks = retrieve_relevant_chunks(query)
    logger.info("Retrieved %d policy chunks for ticket", len(chunks))

    # Step 2: Format policy context for the prompt
    policy_context = format_chunks_for_prompt(chunks)

    # Step 3: Build evidence summary
    evidence_summary = _build_evidence_summary(evidence_files or [])

    # Step 4: Call Gemini with the full context + image files
    raw_response = generate_decision(
        initial_message=initial_message,
        conversation_history=history,
        policy_context=policy_context,
        evidence_summary=evidence_summary,
        evidence_files=evidence_files or [],   # pass full metadata including file_path
    )
    logger.debug("Raw Gemini response: %.200s", raw_response)

    # Step 5: Parse and validate
    decision = _parse_and_validate(raw_response, chunks)
    logger.info(
        "Decision: action=%s status=%s confidence=%.2f",
        decision.action, decision.status, decision.confidence,
    )
    return decision


def _build_retrieval_query(
    initial_message: str,
    history: list[dict],
) -> str:
    """
    Combine initial message and the latest user turn for a richer query.

    Using only the initial message misses context added in follow-ups
    (e.g. "product price is ₹3,500" changes which policy chunks are relevant).
    """
    parts = [initial_message]
    # Add the last 2 user messages for context
    user_turns = [h["content"] for h in history if h.get("role") == "user"]
    parts.extend(user_turns[-2:])
    return " ".join(parts)


def _build_evidence_summary(evidence_files: list[dict]) -> str:
    """Format evidence metadata into a readable list for the prompt."""
    if not evidence_files:
        return ""
    lines = []
    for ev in evidence_files:
        fname = ev.get("filename", "unknown")
        ctype = ev.get("content_type", "unknown")
        lines.append(f"  - {fname} ({ctype})")
    return "The customer has uploaded the following files:\n" + "\n".join(lines)


def _parse_and_validate(raw: str, chunks) -> AIDecision:
    """
    Strip Gemini markdown fences, parse JSON, validate with Pydantic.

    Raises ValueError on any parse or validation failure.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned non-JSON output: {exc}\nRaw: {raw[:300]}"
        ) from exc

    # Back-fill sources from retrieved chunks if Gemini left them empty
    if not data.get("sources"):
        data["sources"] = list({c.source for c in chunks})

    # Ensure required_information is present
    if "required_information" not in data:
        data["required_information"] = []

    try:
        return AIDecision(**data)
    except Exception as exc:
        raise ValueError(
            f"Gemini response failed validation: {exc}\nData: {data}"
        ) from exc
