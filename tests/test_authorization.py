"""
Authorization tests — Alice cannot access Bob's ticket.

Alice creates a ticket; Bob creates a ticket.
We prove cross-user access is blocked at every endpoint.
"""
from unittest.mock import patch
from src.schemas import AIDecision

_MOCK = AIDecision(
    action="REQUEST_MORE_INFORMATION",
    status="IN_PROGRESS",
    confidence=0.5,
    reason="Need more info.",
    question="What is the product value?",
    required_information=["product_value"],
    sources=["damaged_goods.md"],
)


def _reg_login(client, email, pwd="pass1234") -> str:
    client.post("/register", json={"email": email, "password": pwd})
    return client.post("/login", json={"email": email, "password": pwd}).json()["access_token"]


def _new_ticket(client, token, msg="My product is damaged.") -> int:
    r = client.post("/tickets", json={"message": msg},
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 201, r.json()
    return r.json()["ticket"]["id"]


def test_alice_can_access_own_ticket(client):
    with patch("src.decision.run_decision_pipeline", return_value=_MOCK):
        tok = _reg_login(client, "alice@x.com")
        tid = _new_ticket(client, tok)
    r = client.get(f"/tickets/{tid}", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200


def test_alice_cannot_access_bobs_ticket(client):
    with patch("src.decision.run_decision_pipeline", return_value=_MOCK):
        alice = _reg_login(client, "alice@x.com")
        bob = _reg_login(client, "bob@x.com")
        bob_tid = _new_ticket(client, bob)
    r = client.get(f"/tickets/{bob_tid}", headers={"Authorization": f"Bearer {alice}"})
    assert r.status_code == 404


def test_bob_cannot_access_alices_ticket(client):
    with patch("src.decision.run_decision_pipeline", return_value=_MOCK):
        alice = _reg_login(client, "alice@x.com")
        bob = _reg_login(client, "bob@x.com")
        alice_tid = _new_ticket(client, alice)
    r = client.get(f"/tickets/{alice_tid}", headers={"Authorization": f"Bearer {bob}"})
    assert r.status_code == 404


def test_alice_list_excludes_bobs_tickets(client):
    with patch("src.decision.run_decision_pipeline", return_value=_MOCK):
        alice = _reg_login(client, "alice@x.com")
        bob = _reg_login(client, "bob@x.com")
        alice_tid = _new_ticket(client, alice)
        _new_ticket(client, bob)
    r = client.get("/tickets", headers={"Authorization": f"Bearer {alice}"})
    assert r.status_code == 200
    ids = [t["id"] for t in r.json()]
    assert alice_tid in ids
    assert len(ids) == 1


def test_cannot_send_message_to_others_ticket(client):
    with patch("src.decision.run_decision_pipeline", return_value=_MOCK):
        alice = _reg_login(client, "alice@x.com")
        bob = _reg_login(client, "bob@x.com")
        bob_tid = _new_ticket(client, bob)
    r = client.post(
        f"/tickets/{bob_tid}/messages",
        json={"content": "Some follow-up."},
        headers={"Authorization": f"Bearer {alice}"},
    )
    assert r.status_code == 404


def test_cannot_get_decision_for_others_ticket(client):
    with patch("src.decision.run_decision_pipeline", return_value=_MOCK):
        alice = _reg_login(client, "alice@x.com")
        bob = _reg_login(client, "bob@x.com")
        bob_tid = _new_ticket(client, bob)
    r = client.get(
        f"/tickets/{bob_tid}/decision",
        headers={"Authorization": f"Bearer {alice}"},
    )
    assert r.status_code == 404
