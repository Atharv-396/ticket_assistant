"""
Tests for evidence upload endpoint.

Covers: valid image, invalid file type, oversized file.
"""
import io
from unittest.mock import patch

from src.schemas import AIDecision

_MOCK = AIDecision(
    action="REQUEST_PHOTOS", status="IN_PROGRESS", confidence=0.8,
    reason="Photos needed.", question="Please upload photos.",
    required_information=["photos"], sources=["damaged_goods.md"],
)


def _reg_login(client, email="u@x.com", pwd="pass1234") -> str:
    client.post("/register", json={"email": email, "password": pwd})
    return client.post("/login", json={"email": email, "password": pwd}).json()["access_token"]


def _new_ticket(client, token) -> int:
    with patch("src.decision.run_decision_pipeline", return_value=_MOCK):
        r = client.post("/tickets", json={"message": "My product is damaged."},
                        headers={"Authorization": f"Bearer {token}"})
    return r.json()["ticket"]["id"]


def test_valid_image_upload(client):
    tok = _reg_login(client)
    tid = _new_ticket(client, tok)
    fake_image = io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 100)  # minimal JPEG header
    r = client.post(
        f"/tickets/{tid}/evidence",
        headers={"Authorization": f"Bearer {tok}"},
        files={"file": ("photo.jpg", fake_image, "image/jpeg")},
    )
    assert r.status_code == 201
    assert r.json()["filename"] == "photo.jpg"
    assert r.json()["content_type"] == "image/jpeg"


def test_invalid_file_type_rejected(client):
    tok = _reg_login(client)
    tid = _new_ticket(client, tok)
    r = client.post(
        f"/tickets/{tid}/evidence",
        headers={"Authorization": f"Bearer {tok}"},
        files={"file": ("script.exe", io.BytesIO(b"MZ\x00"), "application/octet-stream")},
    )
    assert r.status_code == 415


def test_oversized_file_rejected(client):
    tok = _reg_login(client)
    tid = _new_ticket(client, tok)
    big = io.BytesIO(b"\xff" * (11 * 1024 * 1024))  # 11 MB
    r = client.post(
        f"/tickets/{tid}/evidence",
        headers={"Authorization": f"Bearer {tok}"},
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert r.status_code == 413


def test_png_upload_accepted(client):
    tok = _reg_login(client)
    tid = _new_ticket(client, tok)
    png = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    r = client.post(
        f"/tickets/{tid}/evidence",
        headers={"Authorization": f"Bearer {tok}"},
        files={"file": ("damage.png", png, "image/png")},
    )
    assert r.status_code == 201


def test_evidence_stored_in_ticket(client):
    tok = _reg_login(client)
    tid = _new_ticket(client, tok)
    img = io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 50)
    client.post(
        f"/tickets/{tid}/evidence",
        headers={"Authorization": f"Bearer {tok}"},
        files={"file": ("proof.jpg", img, "image/jpeg")},
    )
    detail = client.get(f"/tickets/{tid}",
                        headers={"Authorization": f"Bearer {tok}"}).json()
    filenames = [e["filename"] for e in detail["evidence"]]
    assert "proof.jpg" in filenames
