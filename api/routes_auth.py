"""Accounts and login sessions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import AuthSession, User

from .context import SESSION_COOKIE, current_user, get_db
from .schemas import LoginIn, RegisterIn, UserOut
from .security import hash_password, new_session_token, token_hash, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])

SHORT_SESSION = timedelta(hours=12)   # "Remember me" unticked (cookie also ends with the browser)
LONG_SESSION = timedelta(days=30)     # "Remember me" ticked


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, db: Session = Depends(get_db)) -> User:
    email = body.email.strip().lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = User(email=email, name=body.name.strip(), password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return user


@router.post("/login", response_model=UserOut)
def login(body: LoginIn, response: Response, db: Session = Depends(get_db)) -> User:
    user = db.scalar(select(User).where(User.email == body.email.strip().lower()))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong email or password")

    lifetime = LONG_SESSION if body.remember else SHORT_SESSION
    token = new_session_token()
    db.add(AuthSession(token_hash=token_hash(token), user_id=user.id,
                       expires_at=datetime.now(timezone.utc) + lifetime))
    db.commit()
    response.set_cookie(
        SESSION_COOKIE, token,
        max_age=int(lifetime.total_seconds()) if body.remember else None,
        httponly=True, samesite="lax", path="/",
    )
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> None:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        sess = db.get(AuthSession, token_hash(token))
        if sess:
            db.delete(sess)
            db.commit()
    response.delete_cookie(SESSION_COOKIE, path="/")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> User:
    return user
