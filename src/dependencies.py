"""
Reusable FastAPI dependencies.

get_current_user
    Extracts and validates the Bearer JWT from the Authorization header,
    then loads the matching User row from the database.

    Used as:  user: User = Depends(get_current_user)

    Why inject from the token rather than a query param?
    - The token is signed — we can trust the user_id it contains.
    - We never let callers pass their own user_id; that would be an IDOR risk.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.auth import decode_access_token
from src.database import get_db
from src.models import User

_bearer = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """
    Validate the Bearer token and return the authenticated User.

    Raises HTTP 401 if the token is missing, invalid, or expired.
    Raises HTTP 401 if the user_id in the token no longer exists in the DB.
    """
    try:
        user_id = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
