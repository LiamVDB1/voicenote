"""
Seed or update a user. Usage:

  python -m voicenote.scripts.seed_user --username liam --display-name "Liam" --password "..."
  python -m voicenote.scripts.seed_user --username mama  --display-name "Mama" --password "..."

If the user exists, password and display_name are updated.
"""
from __future__ import annotations
import argparse
import asyncio
import getpass
import sys

from sqlalchemy import select

from voicenote.auth import hash_password
from voicenote.db import User, _SessionLocal, init_db, normalize_username


async def upsert_user(username: str, display_name: str, password: str, admin: bool | None) -> str:
    await init_db()
    uname = normalize_username(username)
    async with _SessionLocal() as session:
        result = await session.execute(select(User).where(User.username == uname))
        user = result.scalar_one_or_none()
        if user is None:
            # First user gets admin automatically unless --no-admin is passed.
            any_user = (await session.execute(select(User).limit(1))).scalar_one_or_none()
            if admin is None:
                admin = any_user is None
            user = User(
                username=uname,
                password_hash=hash_password(password),
                display_name=(display_name or uname).strip(),
                is_admin=bool(admin),
            )
            session.add(user)
            await session.commit()
            tag = " (admin)" if user.is_admin else ""
            return f"created user '{uname}'{tag}"
        user.password_hash = hash_password(password)
        if display_name:
            user.display_name = display_name.strip()
        if admin is not None:
            user.is_admin = bool(admin)
        await session.commit()
        tag = " (admin)" if user.is_admin else ""
        return f"updated user '{uname}'{tag}"


def main() -> None:
    p = argparse.ArgumentParser(description="Seed or update a voicenote user")
    p.add_argument("--username", required=True)
    p.add_argument("--display-name", default="")
    p.add_argument("--password", default=None, help="omit to be prompted")
    args = p.parse_args()

    pw = args.password
    if not pw:
        pw = getpass.getpass("Password: ")
        if not pw:
            print("password required", file=sys.stderr)
            sys.exit(1)

    msg = asyncio.run(upsert_user(args.username, args.display_name, pw))
    print(msg)


if __name__ == "__main__":
    main()
