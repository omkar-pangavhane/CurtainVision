# models.py
"""
Single source of truth for:
  - SQLAlchemy ORM models: User, Fabric, Room
  - Pydantic request/response schemas used by route.py and auth.py

Fabric/Room only ever store S3 URL + object key metadata -- base64 payloads
and image binaries are never persisted to MySQL.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence

from pydantic import BaseModel, EmailStr, Field, computed_field, field_validator
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from . import config
from .database import Base


# ===========================================================================
# SQLAlchemy ORM models
# ===========================================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=True)        # bcrypt hash; nullable for Google users
    google_id = Column(String(255), nullable=True, unique=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    fabrics = relationship("Fabric", back_populates="uploaded_by_user")
    rooms = relationship("Room", back_populates="uploaded_by_user")

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "email": self.email,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Fabric(Base):
    __tablename__ = "fabrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    #description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    #s3_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    s3_object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    image_hash: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
    uploaded_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    uploaded_by_user = relationship("User", back_populates="fabrics")


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    #description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    #s3_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    s3_object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    image_hash: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
    uploaded_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    uploaded_by_user = relationship("User", back_populates="rooms")


# ===========================================================================
# Pydantic schemas -- Room processing / catalog helpers
# ===========================================================================

class CurtainInfo(BaseModel):
    curtain_id: str
    bbox: List[int]
    area_px: int
    confidence: float


class UploadRoomResponse(BaseModel):
    room_token: str
    model_key: str = "curtain"
    preview_image_base64: str
    curtains_detected: int
    curtains: List[CurtainInfo]
    expires_in_seconds: int


class FabricMetadata(BaseModel):
    fabric_id: str
    name: str
    pattern: str
    color: str
    color_hex: Optional[str] = None
    texture_image_url: str
    tags: List[str] = Field(default_factory=list)
    is_custom_upload: bool = False


class RoomCatalogItem(BaseModel):
    room_id: str
    name: str
    image_url: str
    tags: List[str] = Field(default_factory=list)
    is_custom_upload: bool = False


class FabricSearchRequest(BaseModel):
    query: Optional[str] = None
    pattern: Optional[str] = None
    color: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 50


class RenderCurtainRequest(BaseModel):
    room_token: str
    fabric_id: str
    curtain_ids: Optional[List[str]] = None
    pattern_scale: float = Field(default=1.0, ge=0.05, le=5.0)
    pattern_rotation_deg: float = Field(default=0.0, ge=-360.0, le=360.0)
    pattern_offset_x: float = Field(default=0.0, ge=-10.0, le=10.0)
    pattern_offset_y: float = Field(default=0.0, ge=-10.0, le=10.0)
    color_hue_shift_deg: float = Field(default=0.0, ge=-360.0, le=360.0)

    @field_validator("fabric_id", mode="before")
    @classmethod
    def normalize_fabric_id(cls, value):
        if isinstance(value, int):
            return str(value)
        return value


class RenderCurtainResponse(BaseModel):
    room_token: str
    fabric_id: str
    rendered_image_base64: str
    render_time_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class ModelListResponse(BaseModel):
    models: List[dict]


# ===========================================================================
# Pydantic schemas -- Auth
# ===========================================================================

class UserRegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: Optional[str] = None
    google_id: Optional[str] = None


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: Optional[str] = None      # nullable for Google login
    google_id: Optional[str] = None     # nullable for email/password login


class LoginResponse(BaseModel):
    login: bool
    message: str
    access_token: Optional[str] = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserInfoResponse(BaseModel):
    id: str
    name: str
    email: str
    is_active: bool


# ===========================================================================
# Pydantic schemas -- Fabric
# ===========================================================================

class FabricUploadRequest(BaseModel):
    name: str
    description: Optional[str] = None
    image_base64: str = Field(..., description="Base64 image data, optionally a data: URI")


class FabricUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    image_base64: Optional[str] = Field(
        default=None, description="If provided, replaces the existing S3 image"
    )
    is_active: Optional[bool] = None


class FabricResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    s3_object_key: str
    uploaded_by: int
    is_active: bool

    model_config = {"from_attributes": True}

    # @computed_field
    # @property
    # def s3_url(self) -> str:
    #     """Generate S3 URL from object key."""
    #     return f"https://s3.{config.AWS_REGION}.amazonaws.com/{config.S3_BUCKET_NAME}/{self.s3_object_key}"


class FabricListResponse(BaseModel):
    fabrics: Sequence[FabricResponse]
    total: int


# ===========================================================================
# Pydantic schemas -- Room
# ===========================================================================

class RoomUploadRequest(BaseModel):
    name: str
    description: Optional[str] = None
    image_base64: str = Field(..., description="Base64 image data, optionally a data: URI")


class RoomUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    image_base64: Optional[str] = Field(
        default=None, description="If provided, replaces the existing S3 image"
    )
    is_active: Optional[bool] = None


class RoomResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    s3_object_key: str
    uploaded_by: int
    is_active: bool

    model_config = {"from_attributes": True}

    # @computed_field
    # @property
    # def s3_url(self) -> str:
    #     """Generate S3 URL from object key."""
    #     return f"https://s3.{config.AWS_REGION}.amazonaws.com/{config.S3_BUCKET_NAME}/{self.s3_object_key}"


class RoomListResponse(BaseModel):
    rooms: Sequence[RoomResponse]
    total: int
