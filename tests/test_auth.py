"""Tests for registration, login, JWT, and /me."""
import pytest
from src.auth import (
    create_access_token, decode_access_token,
    hash_password, verify_password,
)
from src.database import get_db
from src.main import app
from src.models import User


# ── Password helpers ──────────────────────────────────────────────────────────

def test_hash_is_not_plain():
    assert hash_password("secret123") != "secret123"


def test_verify_correct():
    h = hash_password("secret123")
    assert verify_password("secret123", h) is True


def test_verify_wrong():
    h = hash_password("secret123")
    assert verify_password("wrong", h) is False


# ── JWT helpers ───────────────────────────────────────────────────────────────

def test_token_roundtrip():
    token = create_access_token(99)
    assert decode_access_token(token) == 99


def test_invalid_token_raises():
    with pytest.raises(ValueError):
        decode_access_token("bad.token.here")


# ── /register ────────────────────────────────────────────────────────────────

def test_register_success(client):
    r = client.post("/register", json={"email": "a@b.com", "password": "pass1234"})
    assert r.status_code == 201
    body = r.json()
    assert body["email"] == "a@b.com"
    assert "password" not in body
    assert "password_hash" not in body


def test_register_duplicate(client):
    p = {"email": "a@b.com", "password": "pass1234"}
    client.post("/register", json=p)
    r = client.post("/register", json=p)
    assert r.status_code == 409


def test_register_short_password(client):
    r = client.post("/register", json={"email": "a@b.com", "password": "short"})
    assert r.status_code == 422


def test_password_stored_hashed(client):
    client.post("/register", json={"email": "a@b.com", "password": "pass1234"})
    db_gen = app.dependency_overrides[get_db]()
    db = next(db_gen)
    user = db.query(User).filter(User.email == "a@b.com").first()
    try:
        next(db_gen)
    except StopIteration:
        pass
    assert user is not None
    assert user.password_hash != "pass1234"
    assert verify_password("pass1234", user.password_hash)


# ── /login ────────────────────────────────────────────────────────────────────

def test_login_success(client):
    client.post("/register", json={"email": "a@b.com", "password": "pass1234"})
    r = client.post("/login", json={"email": "a@b.com", "password": "pass1234"})
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_login_wrong_password(client):
    client.post("/register", json={"email": "a@b.com", "password": "pass1234"})
    r = client.post("/login", json={"email": "a@b.com", "password": "wrong"})
    assert r.status_code == 401


def test_login_unknown_email(client):
    r = client.post("/login", json={"email": "nobody@x.com", "password": "pass1234"})
    assert r.status_code == 401


# ── /me ───────────────────────────────────────────────────────────────────────

def _token(client, email="a@b.com", pwd="pass1234") -> str:
    client.post("/register", json={"email": email, "password": pwd})
    return client.post("/login", json={"email": email, "password": pwd}).json()["access_token"]


def test_me_valid(client):
    tok = _token(client)
    r = client.get("/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json()["email"] == "a@b.com"


def test_me_no_token(client):
    r = client.get("/me")
    assert r.status_code in (401, 403)


def test_me_invalid_token(client):
    r = client.get("/me", headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401
