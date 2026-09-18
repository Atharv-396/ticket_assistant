"""
FastAPI application — v2 multi-turn support ticket assistant.

Endpoints
─────────
POST  /register                   — create account
POST  /login                      — issue JWT
GET   /me                         — authenticated user profile

POST  /tickets                    — create ticket + first AI decision
GET   /tickets                    — list user's tickets (summary)
GET   /tickets/{id}               — full ticket with conversation + decisions
POST  /tickets/{id}/messages      — submit follow-up → new AI decision
POST  /tickets/{id}/evidence      — upload a photo/file
GET   /tickets/{id}/messages      — conversation history
GET   /tickets/{id}/decision      — latest decision

All /tickets/* endpoints are JWT-protected and ownership-enforced.
"""
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.auth import create_access_token, hash_password, verify_password
from src.database import get_db, init_db
from src.dependencies import get_current_user
from src.models import Decision, Evidence, Message, Ticket, User
from src.schemas import (
    AIDecision,
    DecisionResponse,
    ErrorResponse,
    EvidenceResponse,
    FollowUpRequest,
    FollowUpResponse,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    TicketCreate,
    TicketCreateResponse,
    TicketDetail,
    TicketSummary,
    TokenResponse,
    UserResponse,
)
from src.config import settings

logger = logging.getLogger(__name__)

UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif",
    "application/pdf",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    logger.info("Database initialised")
    yield


app = FastAPI(
    title="Support Ticket Decision Assistant",
    description="Multi-turn AI-powered support triage with RAG + Gemini.",
    version="2.0.0",
    lifespan=lifespan,
)

# Serve uploaded files (read-only, for displaying evidence in the UI)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/register", response_model=UserResponse, status_code=201,
          summary="Register a new account")
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> User:
    user = User(email=body.email, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    return user


@app.post("/login", response_model=TokenResponse, summary="Log in and receive a JWT")
def login(body: LoginRequest, db: Session = Depends(get_db)) -> dict:
    user = db.query(User).filter(User.email == body.email).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    return {"access_token": create_access_token(user.id), "token_type": "bearer"}


@app.get("/me", response_model=UserResponse, summary="Current user profile")
def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


# ── Create ticket ─────────────────────────────────────────────────────────────

@app.post("/tickets", response_model=TicketCreateResponse, status_code=201,
          summary="Submit a new support ticket — triggers first AI decision")
def create_ticket(
    body: TicketCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TicketCreateResponse:
    """
    1. Persist ticket.
    2. Store user's initial message.
    3. RAG-retrieve policy chunks.
    4. Call Gemini with ticket + policy.
    5. Validate + store decision.
    6. Store assistant reply as a message.
    7. Return ticket + decision + assistant message.
    """
    from src.decision import run_decision_pipeline

    # Persist ticket
    ticket = Ticket(user_id=current_user.id, initial_message=body.message, status="open")
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    # Store user's first message
    user_msg = Message(ticket_id=ticket.id, role="user", content=body.message)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # Run AI pipeline
    try:
        ai: AIDecision = run_decision_pipeline(
            initial_message=body.message,
            conversation_history=[{"role": "user", "content": body.message}],
        )
    except Exception as exc:
        logger.error("Decision pipeline failed for ticket %d: %s", ticket.id, exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"AI decision service unavailable: {exc}")

    # Store decision
    decision = _store_decision(db, ticket, ai)

    # Store assistant message
    asst_content = _decision_to_message(ai)
    asst_msg = Message(ticket_id=ticket.id, role="assistant", content=asst_content)
    db.add(asst_msg)

    # Close ticket if final
    if ai.status == "FINAL":
        ticket.status = "closed"

    db.commit()
    db.refresh(ticket)
    db.refresh(decision)
    db.refresh(asst_msg)

    return TicketCreateResponse(
        ticket=_ticket_summary(ticket),
        decision=_decision_schema(decision),
        assistant_message=MessageResponse.model_validate(asst_msg),
    )


# ── List tickets ──────────────────────────────────────────────────────────────

@app.get("/tickets", response_model=list[TicketSummary],
         summary="List the authenticated user's tickets")
def list_tickets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[TicketSummary]:
    tickets = (
        db.query(Ticket)
        .filter(Ticket.user_id == current_user.id)
        .order_by(Ticket.created_at.desc())
        .all()
    )
    return [_ticket_summary(t) for t in tickets]


# ── Get ticket ────────────────────────────────────────────────────────────────

@app.get("/tickets/{ticket_id}", response_model=TicketDetail,
         summary="Full ticket with conversation, decisions and evidence")
def get_ticket(
    ticket_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TicketDetail:
    ticket = _get_owned_ticket(ticket_id, current_user.id, db)
    return _ticket_detail(ticket)


# ── Submit follow-up message ──────────────────────────────────────────────────

@app.post("/tickets/{ticket_id}/messages", response_model=FollowUpResponse,
          summary="Submit a follow-up reply to an in-progress ticket")
def add_message(
    ticket_id: int,
    body: FollowUpRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FollowUpResponse:
    """
    1. Verify ownership.
    2. Store user reply.
    3. Reload full conversation.
    4. Re-run RAG + Gemini with full context.
    5. Store new decision + assistant reply.
    """
    from src.decision import run_decision_pipeline

    ticket = _get_owned_ticket(ticket_id, current_user.id, db)

    if ticket.status == "closed":
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Ticket is closed — a final decision has already been made.")

    # Store user reply
    user_msg = Message(ticket_id=ticket.id, role="user", content=body.content)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    db.refresh(ticket)

    # Build conversation history
    history = [{"role": m.role, "content": m.content} for m in ticket.messages]

    # Build evidence metadata — include file_path so Gemini Vision can load images
    ev_meta = [
        {
            "filename": e.filename,
            "content_type": e.content_type,
            "file_path": e.file_path,
        }
        for e in ticket.evidence
    ]

    # Run pipeline
    try:
        ai: AIDecision = run_decision_pipeline(
            initial_message=ticket.initial_message,
            conversation_history=history,
            evidence_files=ev_meta,
        )
    except Exception as exc:
        logger.error("Follow-up pipeline failed for ticket %d: %s", ticket.id, exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"AI decision service unavailable: {exc}")

    # Store decision + assistant message
    decision = _store_decision(db, ticket, ai)
    asst_content = _decision_to_message(ai)
    asst_msg = Message(ticket_id=ticket.id, role="assistant", content=asst_content)
    db.add(asst_msg)

    if ai.status == "FINAL":
        ticket.status = "closed"

    db.commit()
    db.refresh(decision)
    db.refresh(asst_msg)

    return FollowUpResponse(
        decision=_decision_schema(decision),
        assistant_message=MessageResponse.model_validate(asst_msg),
    )


# ── Upload evidence ───────────────────────────────────────────────────────────

@app.post("/tickets/{ticket_id}/evidence", response_model=EvidenceResponse,
          status_code=201, summary="Upload a photo or document as evidence")
async def upload_evidence(
    ticket_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EvidenceResponse:
    """
    Accepts images (JPEG, PNG, WebP, GIF) and PDFs up to 10 MB.
    Stores the file in uploads/ and records metadata in the DB.
    """
    ticket = _get_owned_ticket(ticket_id, current_user.id, db)

    # Validate content type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{content_type}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    # Read and validate size
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    # Save with a safe unique filename
    ext = Path(file.filename or "upload").suffix or ".bin"
    safe_name = f"{ticket_id}_{uuid.uuid4().hex}{ext}"
    file_path = UPLOADS_DIR / safe_name
    file_path.write_bytes(data)

    # Persist metadata
    ev = Evidence(
        ticket_id=ticket.id,
        filename=file.filename or safe_name,
        file_path=str(file_path),
        content_type=content_type,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)

    return EvidenceResponse.model_validate(ev)


# ── Get messages ──────────────────────────────────────────────────────────────

@app.get("/tickets/{ticket_id}/messages", response_model=list[MessageResponse],
         summary="Conversation history for a ticket")
def get_messages(
    ticket_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[MessageResponse]:
    ticket = _get_owned_ticket(ticket_id, current_user.id, db)
    return [MessageResponse.model_validate(m) for m in ticket.messages]


# ── Get latest decision ───────────────────────────────────────────────────────

@app.get("/tickets/{ticket_id}/decision", response_model=DecisionResponse,
         summary="Latest AI decision for a ticket")
def get_decision(
    ticket_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DecisionResponse:
    ticket = _get_owned_ticket(ticket_id, current_user.id, db)
    if not ticket.latest_decision:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No decision recorded yet.")
    return _decision_schema(ticket.latest_decision)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_owned_ticket(ticket_id: int, user_id: int, db: Session) -> Ticket:
    """Load a ticket and verify ownership. Returns 404 for missing or foreign tickets."""
    ticket = db.get(Ticket, ticket_id)
    if ticket is None or ticket.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    return ticket


def _store_decision(db: Session, ticket: Ticket, ai: AIDecision) -> Decision:
    """Persist a validated AIDecision to the decisions table."""
    d = Decision(
        ticket_id=ticket.id,
        action=ai.action,
        status=ai.status,
        reason=ai.reason,
        question=ai.question,
        confidence=ai.confidence,
    )
    d.set_sources(ai.sources)
    d.set_required_information(ai.required_information)
    db.add(d)
    db.flush()   # get the id without committing
    return d


def _decision_to_message(ai: AIDecision) -> str:
    """Convert an AIDecision into a human-readable assistant message."""
    if ai.status == "FINAL":
        return (
            f"**Decision: {ai.action}**\n\n"
            f"{ai.reason}\n\n"
            f"*Confidence: {ai.confidence:.0%}*\n"
            f"*Sources: {', '.join(ai.sources) if ai.sources else 'N/A'}*"
        )
    # IN_PROGRESS — has a question
    question_part = f"\n\n{ai.question}" if ai.question else ""
    return f"{ai.reason}{question_part}"


def _ticket_summary(ticket: Ticket) -> TicketSummary:
    ld = ticket.latest_decision
    return TicketSummary(
        id=ticket.id,
        initial_message=ticket.initial_message,
        status=ticket.status,
        created_at=ticket.created_at,
        updated_at=ticket.updated_at,
        latest_action=ld.action if ld else None,
        latest_decision_status=ld.status if ld else None,
    )


def _decision_schema(d: Decision) -> DecisionResponse:
    return DecisionResponse(
        id=d.id,
        ticket_id=d.ticket_id,
        action=d.action,
        status=d.status,
        reason=d.reason,
        question=d.question,
        confidence=d.confidence,
        sources=d.get_sources(),
        required_information=d.get_required_information(),
        created_at=d.created_at,
    )


def _ticket_detail(ticket: Ticket) -> TicketDetail:
    return TicketDetail(
        id=ticket.id,
        user_id=ticket.user_id,
        initial_message=ticket.initial_message,
        status=ticket.status,
        created_at=ticket.created_at,
        updated_at=ticket.updated_at,
        messages=[MessageResponse.model_validate(m) for m in ticket.messages],
        decisions=[_decision_schema(d) for d in ticket.decisions],
        evidence=[EvidenceResponse.model_validate(e) for e in ticket.evidence],
    )
