"""
SQLAlchemy ORM models — v2 (multi-turn conversation system).

Tables
──────
  users         — registered accounts
  tickets       — support cases (status: open / closed)
  messages      — conversation turns (role: user / assistant)
  decisions     — AI decisions per turn (status: IN_PROGRESS / FINAL)
  evidence      — uploaded files associated with a ticket

Relationships
─────────────
  User  1──* Ticket
  Ticket 1──* Message
  Ticket 1──* Decision   (one per turn; latest is the current decision)
  Ticket 1──* Evidence

JSON helpers
────────────
  Columns that store lists (sources, required_information) use JSON text
  because SQLite has no native array type.
"""
import json
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Users ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    tickets: Mapped[list["Ticket"]] = relationship(
        "Ticket", back_populates="owner", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)


# ── Tickets ───────────────────────────────────────────────────────────────────

class Ticket(Base):
    """
    A support case.

    status:
      open   — conversation still in progress (AI has not yet issued a FINAL decision)
      closed — a FINAL decision has been recorded
    """
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    initial_message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="open", index=True
    )  # "open" | "closed"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    owner: Mapped["User"] = relationship("User", back_populates="tickets")
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="ticket",
        cascade="all, delete-orphan", order_by="Message.created_at",
    )
    decisions: Mapped[list["Decision"]] = relationship(
        "Decision", back_populates="ticket",
        cascade="all, delete-orphan", order_by="Decision.created_at",
    )
    evidence: Mapped[list["Evidence"]] = relationship(
        "Evidence", back_populates="ticket",
        cascade="all, delete-orphan", order_by="Evidence.created_at",
    )

    @property
    def latest_decision(self) -> "Decision | None":
        """The most recently created decision for this ticket."""
        return self.decisions[-1] if self.decisions else None


# ── Messages ──────────────────────────────────────────────────────────────────

class Message(Base):
    """
    A single conversation turn.

    role:
      user      — message from the customer
      assistant — message from the AI
    """
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticket_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="messages")


# ── Decisions ─────────────────────────────────────────────────────────────────

class Decision(Base):
    """
    An AI decision recorded at a specific point in the conversation.

    status:
      IN_PROGRESS — the AI needs more information (conversation continues)
      FINAL       — the AI has produced a conclusive policy-based recommendation

    JSON fields (stored as text):
      sources              — list of policy filenames used
      required_information — list of information types still needed
    """
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticket_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)   # IN_PROGRESS | FINAL
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)  # AI's follow-up question
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    sources: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    required_information: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="decisions")

    def get_sources(self) -> list[str]:
        return json.loads(self.sources)

    def set_sources(self, v: list[str]) -> None:
        self.sources = json.dumps(v)

    def get_required_information(self) -> list[str]:
        return json.loads(self.required_information)

    def set_required_information(self, v: list[str]) -> None:
        self.required_information = json.dumps(v)


# ── Evidence ──────────────────────────────────────────────────────────────────

class Evidence(Base):
    """
    Metadata for an uploaded file (photo, document) associated with a ticket.

    Only metadata is stored in SQLite.
    The actual file lives in uploads/<filename>.
    """
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticket_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="evidence")
