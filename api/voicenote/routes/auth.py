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
    hash_password,
    issue_session,
    verify_password,
    verify_password_constant_time,
)
from ..db import ApiKey, User, get_session, normalize_username

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class UserInfo(BaseModel):
    id: int
    username: str
    display_name: str
    is_admin: bool = False

    @classmethod
    def from_orm(cls, u: User) -> "UserInfo":
        return cls(
            id=u.id,
            username=u.username,
            display_name=u.display_name or u.username,
            is_admin=bool(u.is_admin),
        )


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=6, max_length=256)


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    display_name: str = Field("", max_length=128)
    password: str = Field(..., min_length=6, max_length=256)
    is_admin: bool = False


class UpdateUserRequest(BaseModel):
    display_name: str | None = Field(None, max_length=128)
    password: str | None = Field(None, min_length=6, max_length=256)
    is_admin: bool | None = None


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
    uname = normalize_username(body.username)
    result = await session.execute(select(User).where(User.username == uname))
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


# ----- Self-service password change -----
@router.post("/v1/me/password")
async def change_password(
    body: ChangePasswordRequest,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status_code=401, detail="Huidig wachtwoord klopt niet")
    user.password_hash = hash_password(body.new_password)
    await session.commit()
    return {"ok": True}


# ----- Admin user management -----
async def _require_admin(user: User) -> None:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Alleen voor beheerders")


@router.get("/v1/admin/users")
async def list_users(
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[UserInfo]:
    await _require_admin(user)
    result = await session.execute(select(User).order_by(User.created_at.asc()))
    return [UserInfo.from_orm(u) for u in result.scalars().all()]


@router.post("/v1/admin/users", status_code=201)
async def create_user(
    body: CreateUserRequest,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> UserInfo:
    await _require_admin(user)
    uname = normalize_username(body.username)
    if not uname:
        raise HTTPException(status_code=400, detail="Gebruikersnaam mag niet leeg zijn")
    existing = await session.execute(select(User).where(User.username == uname))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Gebruikersnaam bestaat al")
    new_user = User(
        username=uname,
        password_hash=hash_password(body.password),
        display_name=(body.display_name or uname).strip(),
        is_admin=bool(body.is_admin),
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return UserInfo.from_orm(new_user)


@router.patch("/v1/admin/users/{user_id}")
async def update_user(
    user_id: int,
    body: UpdateUserRequest,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> UserInfo:
    await _require_admin(user)
    target = await session.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Gebruiker niet gevonden")
    if body.display_name is not None:
        target.display_name = body.display_name.strip()
    if body.password is not None:
        target.password_hash = hash_password(body.password)
    if body.is_admin is not None:
        # Don't let an admin demote themselves into a state with zero admins.
        if not body.is_admin and target.id == user.id:
            count = await session.execute(
                select(User).where(User.is_admin == True)  # noqa: E712
            )
            admins = count.scalars().all()
            if len(admins) <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="Kan jezelf niet uit beheer halen — er moet minstens één beheerder blijven",
                )
        target.is_admin = body.is_admin
    await session.commit()
    await session.refresh(target)
    return UserInfo.from_orm(target)


@router.delete("/v1/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    await _require_admin(user)
    if user_id == user.id:
        raise HTTPException(status_code=400, detail="Kan jezelf niet verwijderen")
    target = await session.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Gebruiker niet gevonden")
    await session.delete(target)
    await session.commit()
    return {"ok": True}
