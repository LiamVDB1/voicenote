from __future__ import annotations
import hashlib
import secrets
from datetime import timedelta
from typing import Annotated

import bcrypt
from fastapi import Cookie, Depends, Header, HTTPException, Request, Response, status
from itsdangerous import BadSignature, TimestampSigner
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .db import ApiKey, User, get_session


# ----- Passwords -----
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# Constant-cost dummy hash to defeat username-enumeration timing attacks.
# Generated once at import time; checking against it costs the same as a real bcrypt.
_DUMMY_HASH = hash_password("voicenote-dummy-password-for-timing-equalization")


def verify_password_constant_time(plain: str, hashed: str | None) -> bool:
    """Run bcrypt even when the user does not exist, so login latency is uniform."""
    return verify_password(plain, hashed or _DUMMY_HASH) and hashed is not None


# ----- Sessions (signed cookie) -----
_signer = TimestampSigner(settings.secret_key)


def issue_session(response: Response, user_id: int) -> None:
    token = _signer.sign(str(user_id).encode("utf-8")).decode("utf-8")
    response.set_cookie(
        settings.session_cookie_name,
        token,
        max_age=settings.session_max_age_days * 24 * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )


def clear_session(response: Response) -> None:
    response.delete_cookie(settings.session_cookie_name, path="/")


def _verify_session(token: str) -> int | None:
    try:
        raw = _signer.unsign(
            token, max_age=settings.session_max_age_days * 24 * 3600
        )
        return int(raw.decode("utf-8"))
    except (BadSignature, ValueError):
        return None


# ----- API keys -----
API_KEY_PREFIX = "vn_"


def generate_api_key() -> tuple[str, str, str]:
    """Returns (full_key, prefix, hash). Show full_key once, store prefix+hash."""
    raw = secrets.token_urlsafe(32)
    full = f"{API_KEY_PREFIX}{raw}"
    prefix = full[:10]  # vn_ + 7 chars for display
    h = hashlib.sha256(full.encode("utf-8")).hexdigest()
    return full, prefix, h


def hash_api_key(full: str) -> str:
    return hashlib.sha256(full.encode("utf-8")).hexdigest()


# ----- Dependencies -----
async def current_user(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    vn_session: Annotated[str | None, Cookie(alias=None)] = None,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> User:
    # 1. Cookie session
    cookie_val = request.cookies.get(settings.session_cookie_name)
    if cookie_val:
        uid = _verify_session(cookie_val)
        if uid is not None:
            user = await session.get(User, uid)
            if user is not None:
                return user

    # 2. API key (Authorization: Bearer ... OR X-API-Key)
    api_key = None
    if authorization and authorization.lower().startswith("bearer "):
        api_key = authorization.split(" ", 1)[1].strip()
    elif x_api_key:
        api_key = x_api_key.strip()

    if api_key:
        h = hash_api_key(api_key)
        result = await session.execute(
            select(ApiKey).where(ApiKey.key_hash == h, ApiKey.revoked == False)  # noqa: E712
        )
        rec = result.scalar_one_or_none()
        if rec is not None:
            user = await session.get(User, rec.user_id)
            if user is not None:
                # Throttle last_used_at writes — every request would thrash SQLite
                # under load. Only persist if it's been at least 5 minutes.
                from datetime import datetime, timedelta, timezone
                now = datetime.now(timezone.utc)
                if rec.last_used_at is None or (now - rec.last_used_at) > timedelta(minutes=5):
                    rec.last_used_at = now
                    try:
                        await session.commit()
                    except Exception:
                        await session.rollback()
                return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Niet ingelogd",
    )
