"""
Tests for ticket creation, follow-up, persistence, and conversation storage.
"""
import pytest
from unittest.mock import patch
from src.schemas import AIDecision

_IN_PROGRESS = AIDecision(
    action="REQUEST_MORE_INFORMATION",
    status="IN_PROGRESS",
    confidence=0.7,
    reason="Need the order value.",
    question="What is the order value?",
    required_information=["order_value"],
    sources=["damaged_goods.md"],
)

_FINAL = AIDecision(
    action="APPROVE_REFUND",
    status="FINAL",
    confidence=0.93,
    reason="Damage confirmed, value ₹3,500 > ₹2,000 threshold, photos provided.",
    question=None,
    required_information=[],
    sources=["damaged_goods.md", "refunds.md"],
)


def _reg_login(client, email="u@x.com", pwd="pass1234") -> str:
    client.post("/register", json={"email": email, "password": pwd})
    return client.post("/login", json={"email": email, "password": pwd}).json()["access_token"]


# ── Creation ──────────────────────────────────────────────────────────────────

def test_create_ticket_returns_201(client):
    tok = _reg_login(client)
    with patch("src.decision.run_decision_pipeline", return_value=_IN_PROGRESS):
        r = client.post("/tickets", json={"message": "My product is damaged."},
                        headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 201
    body = r.json()
    assert "ticket" in body
    assert "decision" in body
    assert "assistant_message" in body


def test_create_ticket_unauthenticated(client):
    r = client.post("/tickets", json={"message": "My product is damaged."})
    assert r.status_code in (401, 403)


def test_create_ticket_too_short(client):
    tok = _reg_login(client)
    r = client.post("/tickets", json={"message": "hi"},
                    headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 422


def test_in_progress_decision_stored(client):
    tok = _reg_login(client)
    with patch("src.decision.run_decision_pipeline", return_value=_IN_PROGRESS):
        r = client.post("/tickets", json={"message": "Product damaged."},
                        headers={"Authorization": f"Bearer {tok}"})
    dec = r.json()["decision"]
    assert dec["status"] == "IN_PROGRESS"
    assert dec["action"] == "REQUEST_MORE_INFORMATION"
    assert dec["question"] is not None


def test_final_decision_closes_ticket(client):
    tok = _reg_login(client)
    with patch("src.decision.run_decision_pipeline", return_value=_FINAL):
        r = client.post("/tickets", json={"message": "Damaged product, ₹3500, photo attached."},
                        headers={"Authorization": f"Bearer {tok}"})
    assert r.json()["ticket"]["status"] == "closed"
    assert r.json()["decision"]["status"] == "FINAL"


# ── Persistence ───────────────────────────────────────────────────────────────

def test_ticket_persisted_in_db(client):
    tok = _reg_login(client)
    with patch("src.decision.run_decision_pipeline", return_value=_IN_PROGRESS):
        r = client.post("/tickets", json={"message": "Product arrived broken."},
                        headers={"Authorization": f"Bearer {tok}"})
    tid = r.json()["ticket"]["id"]
    r2 = client.get(f"/tickets/{tid}", headers={"Authorization": f"Bearer {tok}"})
    assert r2.status_code == 200
    assert r2.json()["id"] == tid


def test_initial_message_in_conversation(client):
    tok = _reg_login(client)
    with patch("src.decision.run_decision_pipeline", return_value=_IN_PROGRESS):
        r = client.post("/tickets", json={"message": "Product arrived broken."},
                        headers={"Authorization": f"Bearer {tok}"})
    tid = r.json()["ticket"]["id"]
    msgs = client.get(f"/tickets/{tid}/messages",
                      headers={"Authorization": f"Bearer {tok}"}).json()
    roles = [m["role"] for m in msgs]
    assert "user" in roles
    assert "assistant" in roles


# ── Follow-up ─────────────────────────────────────────────────────────────────

def test_follow_up_message_advances_conversation(client):
    tok = _reg_login(client)
    with patch("src.decision.run_decision_pipeline", return_value=_IN_PROGRESS):
        r = client.post("/tickets", json={"message": "Product is damaged."},
                        headers={"Authorization": f"Bearer {tok}"})
    tid = r.json()["ticket"]["id"]

    with patch("src.decision.run_decision_pipeline", return_value=_FINAL):
        r2 = client.post(f"/tickets/{tid}/messages",
                         json={"content": "Order value is ₹3,500, delivered yesterday."},
                         headers={"Authorization": f"Bearer {tok}"})
    assert r2.status_code == 200
    assert r2.json()["decision"]["status"] == "FINAL"


def test_follow_up_to_closed_ticket_rejected(client):
    tok = _reg_login(client)
    with patch("src.decision.run_decision_pipeline", return_value=_FINAL):
        r = client.post("/tickets", json={"message": "Damaged ₹3500 product with photos."},
                        headers={"Authorization": f"Bearer {tok}"})
    tid = r.json()["ticket"]["id"]

    r2 = client.post(f"/tickets/{tid}/messages",
                     json={"content": "More info."},
                     headers={"Authorization": f"Bearer {tok}"})
    assert r2.status_code == 400


def test_multiple_decisions_stored(client):
    tok = _reg_login(client)
    with patch("src.decision.run_decision_pipeline", return_value=_IN_PROGRESS):
        r = client.post("/tickets", json={"message": "Damaged product."},
                        headers={"Authorization": f"Bearer {tok}"})
    tid = r.json()["ticket"]["id"]

    with patch("src.decision.run_decision_pipeline", return_value=_FINAL):
        client.post(f"/tickets/{tid}/messages",
                    json={"content": "₹3500, delivered yesterday."},
                    headers={"Authorization": f"Bearer {tok}"})

    detail = client.get(f"/tickets/{tid}", headers={"Authorization": f"Bearer {tok}"}).json()
    assert len(detail["decisions"]) == 2


# ── History ───────────────────────────────────────────────────────────────────

def test_history_shows_only_own_tickets(client):
    tok1 = _reg_login(client, "u1@x.com")
    tok2 = _reg_login(client, "u2@x.com")
    with patch("src.decision.run_decision_pipeline", return_value=_IN_PROGRESS):
        client.post("/tickets", json={"message": "User 1 ticket."},
                    headers={"Authorization": f"Bearer {tok1}"})
        client.post("/tickets", json={"message": "User 2 ticket."},
                    headers={"Authorization": f"Bearer {tok2}"})

    r = client.get("/tickets", headers={"Authorization": f"Bearer {tok1}"})
    assert len(r.json()) == 1


def test_history_empty_for_new_user(client):
    tok = _reg_login(client)
    r = client.get("/tickets", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json() == []
