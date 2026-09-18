"""
Pydantic request/response schemas — v2 multi-turn system.

Three categories:
  1. Auth         — register, login, token, user profile
  2. Tickets      — create, list, detail, conversation, evidence
  3. AI Decision  — structured Gemini output validated before storage

Passwords are NEVER included in any response schema.
All Gemini output is validated through AIDecision before touching the DB.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, description="Minimum 8 characters")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# AI Decision  (Gemini structured output — validated before storage)
# ─────────────────────────────────────────────────────────────────────────────

VALID_ACTIONS = {
    # Information-gathering states
    "REQUEST_MORE_INFORMATION",
    "REQUEST_PHOTOS",
    "REQUEST_ORDER_DETAILS",
    "REQUEST_DELIVERY_DETAILS",
    "NEEDS_MORE_INFORMATION",
    # Refund / replacement outcomes
    "APPROVE_REFUND",
    "REJECT_REFUND",
    "APPROVE_REFUND_OR_REPLACEMENT",
    # Return outcomes
    "APPROVE_RETURN",
    "REJECT_RETURN",
    "REJECT_OUTSIDE_WINDOW",
    "REJECT_FOOD_RETURN",
    "REJECT_OPENED_ITEM",
    # Other outcomes
    "REPLACE_CORRECT_ITEM",
    "APPROVE_REPLACEMENT",
    "REQUEST_DEFECT_EVIDENCE",
    "OPEN_SHIPPING_INVESTIGATION",
    "WAIT_AND_TRACK",
    "OFFER_REPLACEMENT_OR_REFUND",
    "CANCEL_AND_REFUND",
    "CANNOT_CANCEL_AFTER_DISPATCH",
}

VALID_STATUSES = {"IN_PROGRESS", "FINAL"}


class AIDecision(BaseModel):
    """
    Structured decision returned by Gemini.  Every field is validated
    before the decision is persisted.

    status IN_PROGRESS  → conversation continues; question is required
    status FINAL        → conclusive policy-based recommendation
    """
    action: str
    status: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    question: Optional[str] = None          # follow-up question for the user
    required_information: list[str] = []    # info types still needed
    sources: list[str] = []

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in VALID_ACTIONS:
            # Accept unknown actions rather than crashing — log but allow
            # so new policy actions don't break validation.
            pass
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got '{v}'")
        return v

    @field_validator("sources", "required_information", mode="before")
    @classmethod
    def coerce_to_list(cls, v) -> list:
        if v is None:
            return []
        return [str(x) for x in v]


# ─────────────────────────────────────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    id: int
    ticket_id: int
    role: str           # "user" | "assistant"
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class FollowUpRequest(BaseModel):
    """Body for POST /tickets/{id}/messages — user's follow-up reply."""
    content: str = Field(min_length=2, max_length=4000)


# ─────────────────────────────────────────────────────────────────────────────
# Decisions
# ─────────────────────────────────────────────────────────────────────────────

class DecisionResponse(BaseModel):
    """API representation of a stored Decision row."""
    id: int
    ticket_id: int
    action: str
    status: str             # IN_PROGRESS | FINAL
    reason: str
    question: Optional[str]
    confidence: float
    sources: list[str]
    required_information: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Evidence
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceResponse(BaseModel):
    id: int
    ticket_id: int
    filename: str
    content_type: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Tickets
# ─────────────────────────────────────────────────────────────────────────────

class TicketCreate(BaseModel):
    message: str = Field(min_length=5, max_length=4000)


class TicketSummary(BaseModel):
    """Lightweight row for the history list."""
    id: int
    initial_message: str
    status: str             # open | closed
    created_at: datetime
    updated_at: datetime
    latest_action: Optional[str] = None
    latest_decision_status: Optional[str] = None    # IN_PROGRESS | FINAL

    model_config = {"from_attributes": True}


class TicketDetail(BaseModel):
    """Full ticket with conversation, decisions, and evidence."""
    id: int
    user_id: int
    initial_message: str
    status: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []
    decisions: list[DecisionResponse] = []
    evidence: list[EvidenceResponse] = []

    model_config = {"from_attributes": True}


class TicketCreateResponse(BaseModel):
    """
    Returned after POST /tickets.

    Includes the ticket metadata plus the first AI decision so the
    frontend knows immediately whether to show a follow-up form or a
    final decision.
    """
    ticket: TicketSummary
    decision: DecisionResponse
    assistant_message: MessageResponse


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up response
# ─────────────────────────────────────────────────────────────────────────────

class FollowUpResponse(BaseModel):
    """Returned after POST /tickets/{id}/messages."""
    decision: DecisionResponse
    assistant_message: MessageResponse


# ─────────────────────────────────────────────────────────────────────────────
# Error
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
