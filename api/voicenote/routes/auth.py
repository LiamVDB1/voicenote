from __future__ import annotations
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import (
    clear_session,
    current_user,
    generate_api_key,
    hash_api_key,
    issue_session,
    verify_password_constant_time,
)
from ..db import ApiKey, User, get_session

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class UserInfo(BaseModel):
    id: int
    username: str
    display_name: str

    @classmethod
    def from_orm(cls, u: User) -> "UserInfo":
        return cls(id=u.id, username=u.username, display_name=u.display_name or u.username)


class CreateKeyRequest(BaseModel):
    name: str = Field("", max_length=128)


class CreatedKey(BaseModel):
    id: int
    name: str
    prefix: str
    key: str  # full key, shown once


class KeyInfo(BaseModel):
    id: int
    name: str
    prefix: str
    created_at: str
    last_used_at: str | None
    revoked: bool


@router.post("/v1/auth/login")
async def login(
    body: LoginRequest,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> UserInfo:
    result = await session.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()
    # Always run bcrypt — including against a dummy hash when user is missing —
    # so login latency doesn't reveal whether the username exists.
    ok = verify_password_constant_time(body.password, user.password_hash if user else None)
    if not ok or user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Gebruikersnaam of wachtwoord klopt niet",
        )
    issue_session(response, user.id)
    return UserInfo.from_orm(user)


@router.post("/v1/auth/logout")
async def logout(response: Response) -> dict:
    clear_session(response)
    return {"ok": True}


@router.get("/v1/me")
async def me(user: Annotated[User, Depends(current_user)]) -> UserInfo:
    return UserInfo.from_orm(user)


@router.get("/v1/keys")
async def list_keys(
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[KeyInfo]:
    result = await session.execute(
        select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc())
    )
    keys = result.scalars().all()
    return [
        KeyInfo(
            id=k.id, name=k.name, prefix=k.prefix,
            created_at=k.created_at.isoformat(),
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
            revoked=k.revoked,
        )
        for k in keys
    ]


@router.post("/v1/keys", status_code=201)
async def create_key(
    body: CreateKeyRequest,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CreatedKey:
    full, prefix, h = generate_api_key()
    rec = ApiKey(user_id=user.id, name=body.name or "untitled", prefix=prefix, key_hash=h)
    session.add(rec)
    await session.commit()
    await session.refresh(rec)
    return CreatedKey(id=rec.id, name=rec.name, prefix=rec.prefix, key=full)


@router.delete("/v1/keys/{key_id}")
async def revoke_key(
    key_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    rec = await session.get(ApiKey, key_id)
    if rec is None or rec.user_id != user.id:
        raise HTTPException(status_code=404, detail="Sleutel niet gevonden")
    rec.revoked = True
    await session.commit()
    return {"ok": True}
