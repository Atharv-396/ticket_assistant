"""
Gemini API wrapper — v2.1 with Vision support.

Public functions
────────────────
  embed_text(text)               → list[float]   query embedding
  embed_document_chunk(text)     → list[float]   document chunk embedding
  generate_decision(...)         → str           raw JSON decision from Gemini

Vision support
──────────────
  When evidence_files is provided and contains readable image paths,
  the images are loaded and sent to Gemini as inline image parts so the
  model can observe actual damage, defects, or wrong items.

  This uses the same gemini-2.5-flash model (it supports vision natively).

  Important: we never claim photos "prove" anything beyond what the model
  observes. Image analysis supplements but does not replace policy grounding.

Why send images directly?
  google-generativeai supports inline image bytes via PIL.Image or
  raw bytes + MIME type. No separate Files API call needed for images
  under ~20 MB, which keeps the implementation simple.
"""
import base64
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from src.config import settings

logger = logging.getLogger(__name__)

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set — Gemini calls will fail.")

# MIME types we will attempt to send as inline image data
_VISION_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """Query embedding (retrieval_query task type)."""
    _require_key()
    try:
        result = genai.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]
    except Exception as exc:
        raise RuntimeError(f"Gemini embedding failed: {exc}") from exc


def embed_document_chunk(text: str) -> list[float]:
    """Document chunk embedding (retrieval_document task type)."""
    _require_key()
    try:
        result = genai.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]
    except Exception as exc:
        raise RuntimeError(f"Gemini document embedding failed: {exc}") from exc


# ── Decision generation ───────────────────────────────────────────────────────

def generate_decision(
    initial_message: str,
    conversation_history: list[dict],
    policy_context: str,
    evidence_summary: str = "",
    evidence_files: Optional[list[dict]] = None,
) -> str:
    """
    Send the full conversation context + policy chunks (+ images) to Gemini.

    Parameters
    ----------
    initial_message : str
        The customer's original ticket message.
    conversation_history : list[dict]
        All previous turns: [{"role": "user"|"assistant", "content": "..."}]
    policy_context : str
        Formatted retrieved policy chunks.
    evidence_summary : str
        Human-readable list of uploaded file metadata (always included).
    evidence_files : list[dict], optional
        Each item: {"filename": str, "file_path": str, "content_type": str}
        Images are loaded from file_path and sent inline to Gemini.

    Returns
    -------
    str
        Raw text response from Gemini (caller must parse/validate as JSON).
    """
    _require_key()

    history_text = _format_history(conversation_history)

    evidence_section = ""
    if evidence_summary:
        evidence_section = f"\n## Uploaded Evidence\n{evidence_summary}\n"

    prompt_text = f"""You are a customer support decision assistant for an e-commerce company.

Your job is to decide what action to take for a support ticket using ONLY the
provided company policy excerpts below. Never invent policies or customer facts.

## Original Ticket
{initial_message}

## Conversation History
{history_text if history_text else "(no previous messages)"}
{evidence_section}
## Relevant Company Policies
{policy_context}

## Your Task
Analyze the ticket, conversation, uploaded evidence (including any images shown
below), and the policies above.

Decide one of:
1. You need more information → status=IN_PROGRESS, appropriate REQUEST_* action,
   write a targeted question for the customer.
2. You have enough information → status=FINAL, conclusive action.

## Valid Actions
REQUEST_MORE_INFORMATION, REQUEST_PHOTOS, REQUEST_ORDER_DETAILS,
REQUEST_DELIVERY_DETAILS, NEEDS_MORE_INFORMATION,
APPROVE_REFUND, REJECT_REFUND, APPROVE_REFUND_OR_REPLACEMENT,
APPROVE_RETURN, REJECT_RETURN, REJECT_OUTSIDE_WINDOW,
REJECT_FOOD_RETURN, REJECT_OPENED_ITEM,
REPLACE_CORRECT_ITEM, APPROVE_REPLACEMENT, REQUEST_DEFECT_EVIDENCE,
OPEN_SHIPPING_INVESTIGATION, WAIT_AND_TRACK,
OFFER_REPLACEMENT_OR_REFUND, CANCEL_AND_REFUND,
CANNOT_CANCEL_AFTER_DISPATCH

## Rules
- Use ONLY the supplied policy context. Do not invent company rules.
- If photos are required by policy (e.g. high-value damage claims) and have been
  provided, evaluate what you can observe in them.
- Only describe what is visually evident; do not over-claim certainty.
- If photos are required but not yet provided, use REQUEST_PHOTOS.
- When status=IN_PROGRESS the "question" field must contain a clear question.
- When status=FINAL set "question" to null.
- Sources must be the actual policy filenames provided above.
- confidence is your certainty the action is correct (0.0–1.0).

## Response Format
Respond with ONLY this JSON — no markdown, no extra text:

{{
  "action": "<ACTION>",
  "status": "<IN_PROGRESS|FINAL>",
  "confidence": <0.0-1.0>,
  "reason": "<explanation referencing the policy>",
  "question": "<targeted question or null>",
  "required_information": ["<item1>"],
  "sources": ["<policy_filename.md>"]
}}
"""

    # Build content parts: text prompt + optional inline images
    content_parts: list = [prompt_text]
    image_count = 0

    if evidence_files:
        for ev in evidence_files:
            mime = ev.get("content_type", "")
            fpath = ev.get("file_path", "")
            fname = ev.get("filename", "unknown")

            if mime in _VISION_TYPES and fpath and Path(fpath).exists():
                try:
                    img_bytes = Path(fpath).read_bytes()
                    # Gemini SDK accepts {"mime_type": ..., "data": bytes}
                    content_parts.append({"mime_type": mime, "data": img_bytes})
                    image_count += 1
                    logger.info("Added image to prompt: %s (%d bytes)", fname, len(img_bytes))
                except Exception as exc:
                    logger.warning("Could not load image %s: %s", fname, exc)

    if image_count:
        logger.info("Sending %d image(s) to Gemini Vision", image_count)
    
    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content(content_parts)
        return response.text
    except Exception as exc:
        raise RuntimeError(f"Gemini generation failed: {exc}") from exc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_key() -> None:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Set it in your .env file."
        )


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    return "\n".join(
        f"{t.get('role','user').upper()}: {t.get('content','')}"
        for t in history
    )
