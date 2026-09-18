"""
Authentication helpers: password hashing and JWT creation/verification.

Why bcrypt?  Industry-standard adaptive hashing — intentionally slow to
resist brute-force attacks.  passlib wraps it with a clean API.

Why JWT?  Stateless — the server doesn't need to store sessions.  The token
carries the user identity (sub = user id) and an expiry claim.
"""
from datetime import datetime, timedelta, timezone

import warnings

from jose import JWTError, jwt
from passlib.context import CryptContext

from src.config import settings

# Suppress a benign passlib/bcrypt version-detection warning that appears
# when bcrypt >= 4.x is installed (passlib reads __about__ which was removed).
warnings.filterwarnings(
    "ignore",
    message=".*error reading bcrypt version.*",
    category=UserWarning,
)

# bcrypt is the hashing scheme; deprecated="auto" will warn if an old scheme
# is encountered so we can rehash transparently.
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*.  Never store the plain value."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*."""
    return _pwd_context.verify(plain, hashed)


# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(user_id: int) -> str:
    """
    Create a signed JWT containing the user's id as the 'sub' claim.

    The token expires after JWT_EXPIRE_MINUTES (default 60 minutes).
    Signed with HS256 using JWT_SECRET from the environment.
    """
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.JWT_EXPIRE_MINUTES
    )
    payload = {
        "sub": str(user_id),
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    """
    Decode and verify a JWT.  Returns the user id (int) on success.

    Raises ValueError if the token is invalid, expired, or malformed.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        sub: str | None = payload.get("sub")
        if sub is None:
            raise ValueError("Token missing 'sub' claim")
        return int(sub)
    except JWTError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc
