# app/auth.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .database import get_db
from .models import User
from .auth_utils import (
    hash_password, verify_password, create_access_token,
    verify_google_token
)
from .models import (
    UserRegisterRequest, UserLoginRequest, AuthResponse,
    UserInfoResponse, LoginResponse
)
from .dependencies import get_current_user

router = APIRouter(prefix="/api/auth", tags=["authentication"])

@router.post("/register", response_model=AuthResponse)
async def register_user(
    req: UserRegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    # Check if email exists
    result = await db.execute(select(User).where(User.email == req.email))
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered")

    hashed = hash_password(req.password) if req.password else None    
    user = User(
        name=req.name,
        email=req.email,
        password=hashed,
        google_id=req.google_id,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    access_token = create_access_token(data={"sub": str(user.id)})
    return AuthResponse(access_token=access_token)


@router.post("/login", response_model=LoginResponse)
async def login_user(
    req: UserLoginRequest,
    db: AsyncSession = Depends(get_db),
):
    # --- Google login path ---
    # Treat common placeholders (0, "0", empty, "false", "null") as no Google ID.
    raw_google = req.google_id
    normalized_google = None
    if raw_google is not None:
        try:
            normalized_google = str(raw_google).strip()
        except Exception:
            normalized_google = None

    # Decide whether the provided value should be treated as a real Google ID/token.
    placeholders = ("0", "", "false", "none", "null")
    is_google_provided = False
    if normalized_google and normalized_google.lower() not in placeholders:
        # Heuristic: Google ID tokens are JWTs (contain '.'), or the
        # provider `sub` is a reasonably long string. Treat short values
        # like "0" or similar as placeholders.
        if "." in normalized_google or len(normalized_google) >= 10:
            is_google_provided = True

    if is_google_provided:
        # Client may send either the raw Google `sub` (google_id) or an
        # ID token JWT. If it's a token (contains '.'), verify it with
        # Google and extract the `sub`, `email`, and `name`.
        google_sub = normalized_google
        google_email = req.email
        google_payload = None

        if "." in (req.google_id or ""):
            google_payload = await verify_google_token(req.google_id)
            if not google_payload:
                return LoginResponse(login=False, message="Invalid Google token")
            google_sub = google_payload.get("sub")
            google_email = google_payload.get("email")

        if not google_sub:
            return LoginResponse(login=False, message="Invalid Google credentials")

        result = await db.execute(select(User).where(User.google_id == google_sub))
        user = result.scalar_one_or_none()
        if not user and google_email:
            result = await db.execute(select(User).where(User.email == google_email))
            user = result.scalar_one_or_none()
            if user and not user.google_id:
                user.google_id = google_sub
                db.add(user)
                await db.commit()
                await db.refresh(user)

        if not user:
            return LoginResponse(login=False, message="User not found")

        if not user.is_active:
            return LoginResponse(login=False, message="User account is inactive")
        access_token = create_access_token(data={"sub": str(user.id)})
        return LoginResponse(login=True, access_token=access_token, message="Login successful")

    # --- Email/password login path ---
    if not req.password:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Password required for email/password login")

    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()
    if not user:
        return LoginResponse(login=False, message="Email not found")

    if not user.password:
        return LoginResponse(login=False, message="Account uses Google Sign-In. Please use Google login.")

    if not verify_password(req.password, user.password):
        return LoginResponse(login=False, message="Invalid password")

    if not user.is_active:
        return LoginResponse(login=False, message="User account is inactive")

    access_token = create_access_token(data={"sub": str(user.id)})
    return LoginResponse(login=True, access_token=access_token, message="Login successful")


@router.get("/me", response_model=UserInfoResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserInfoResponse(
        id=str(current_user.id),
        name=current_user.name,
        email=current_user.email,
        is_active=current_user.is_active,
    )


@router.post("/logout")
async def logout():
    # JWT is stateless – client discards the token.
    return {"message": "Logged out successfully"}
