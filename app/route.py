# route.py
"""All Fabric and Room CRUD APIs. Every endpoint requires a valid JWT
(via the existing get_current_user dependency). Images are received as
base64, uploaded to S3, and only the resulting URL/key are persisted in
MySQL -- no image binary or base64 payload is ever stored in the database
or on local disk.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import config, curtain_service, fabric_catalog, room_catalog, s3_service
from .database import get_db
from .dependencies import get_current_user
from .model_manager import ModelManager
from .models import (
    CurtainInfo,
    ErrorResponse,
    Fabric,
    FabricListResponse,
    FabricResponse,
    FabricSearchRequest,
    FabricUpdateRequest,
    FabricUploadRequest,
    ModelListResponse,
    RenderCurtainRequest,
    RenderCurtainResponse,
    Room,
    RoomListResponse,
    RoomResponse,
    RoomUpdateRequest,
    RoomUploadRequest,
    UploadRoomResponse,
    User,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["catalog"])


def _read_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file")
    return image


def _resize_large_image(image_bgr: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return image_bgr
    scale = max_dim / max(h, w)
    resized = cv2.resize(
        image_bgr,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def _encode_bgr_to_base64(image_bgr: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _upload_or_400(image_base64: str,name: str, prefix: str):
    try:
        return s3_service.upload_base64_image(image_base64, name, prefix)
    except s3_service.InvalidBase64ImageError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except s3_service.S3UploadError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e))


def _validate_model_key(model_key: str) -> None:
    if model_key not in ModelManager.available_models():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_key}'. Available: {ModelManager.available_models()}",
        )


# ===========================================================================
# Fabric
# ===========================================================================

@router.post("/fabrics", response_model=FabricResponse, status_code=status.HTTP_201_CREATED)
async def upload_fabric(
    req: FabricUploadRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Decode image and compute SHA-256 to prevent duplicate uploads
    try:
        image_bytes, _ext = s3_service.decode_base64_image(req.image_base64)
    except s3_service.InvalidBase64ImageError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    image_hash = hashlib.sha256(image_bytes).hexdigest()
    stmt = select(Fabric).where(Fabric.image_hash == image_hash)
    existing = (await db.execute(stmt)).scalars().first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Image already exists.")

    stmt = select(Fabric).where(Fabric.name == req.name)
    existing_name = (await db.execute(stmt)).scalars().first()
    if existing_name:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Image with this name already exists.")

    s3_url, s3_key = _upload_or_400(req.image_base64, req.name, config.S3_FABRIC_PREFIX)

    fabric = Fabric(
        name=req.name,
        # description=req.description,
        s3_object_key=s3_key,
        image_hash=image_hash,
        uploaded_by=current_user.id,
        is_active=True,
    )
    db.add(fabric)
    await db.commit()
    await db.refresh(fabric)
    return fabric


@router.get("/fabrics", response_model=FabricListResponse)
async def list_fabrics(
    include_inactive: bool = False,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = select(Fabric)
    if not include_inactive:
        stmt = stmt.where(Fabric.is_active == True)  # noqa: E712
    stmt = stmt.order_by(Fabric.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    fabrics = list(result.scalars())
    fabric_responses = [FabricResponse.model_validate(fabric, from_attributes=True) for fabric in fabrics]
    return FabricListResponse(fabrics=fabric_responses, total=len(fabric_responses))


@router.put("/fabrics/{fabric_id}", response_model=FabricResponse)
async def update_fabric(
    fabric_id: int,
    req: FabricUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    fabric = await db.get(Fabric, fabric_id)
    if fabric is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Fabric {fabric_id} not found")

    if req.name is not None:
        stmt = select(Fabric).where(Fabric.name == req.name)
        existing_name = (await db.execute(stmt)).scalars().first()
        if existing_name is not None and existing_name.id != fabric_id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Fabric with this name already exists.")
        fabric.name = req.name
    # if req.description is not None:
    #     fabric.description = req.description
    if req.is_active is not None:
        fabric.is_active = req.is_active

    if req.image_base64 is not None:
        try:
            image_bytes, _ext = s3_service.decode_base64_image(req.image_base64)
        except s3_service.InvalidBase64ImageError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        stmt = select(Fabric).where(Fabric.image_hash == image_hash)
        existing = (await db.execute(stmt)).scalars().first()
        if existing is not None and existing.id != fabric_id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Image already exists.")

        new_url, new_key = _upload_or_400(req.image_base64, req.name or fabric.name, config.S3_FABRIC_PREFIX)
        old_key = fabric.s3_object_key
        fabric.s3_object_key = new_key
        fabric.image_hash = image_hash
        try:
            s3_service.delete_object(old_key)
        except s3_service.S3UploadError:
            logger.warning("Old fabric image %s could not be deleted from S3", old_key)

    db.add(fabric)
    await db.commit()
    await db.refresh(fabric)
    return fabric


@router.delete("/fabrics/{fabric_id}")
async def delete_fabric(
    fabric_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    fabric = await db.get(Fabric, fabric_id)
    if fabric is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Fabric {fabric_id} not found")

    try:
        s3_service.delete_object(fabric.s3_object_key)
    except s3_service.S3UploadError:
        logger.warning("Fabric image %s could not be deleted from S3", fabric.s3_object_key)

    await db.delete(fabric)
    await db.commit()
    return {"message": f"Fabric {fabric_id} deleted"}


# ===========================================================================
# Room
# ===========================================================================

@router.post("/rooms", response_model=RoomResponse, status_code=status.HTTP_201_CREATED)
async def create_room(
    req: RoomUploadRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Decode image and compute SHA-256 to prevent duplicate uploads
    try:
        image_bytes, _ext = s3_service.decode_base64_image(req.image_base64)
    except s3_service.InvalidBase64ImageError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    image_hash = hashlib.sha256(image_bytes).hexdigest()
    stmt = select(Room).where(Room.image_hash == image_hash)
    existing = (await db.execute(stmt)).scalars().first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Image already exists.")

    stmt = select(Room).where(Room.name == req.name)
    existing_name = (await db.execute(stmt)).scalars().first()
    if existing_name:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Room with this name already exists.")

    s3_url, s3_key = _upload_or_400(req.image_base64, req.name, config.S3_ROOM_PREFIX)

    room = Room(
        name=req.name,
        #description=req.description,
        s3_object_key=s3_key,
        image_hash=image_hash,
        uploaded_by=current_user.id,
        is_active=True,
    )
    db.add(room)
    await db.commit() 
    await db.refresh(room)
    return room

@router.get("/rooms", response_model=RoomListResponse)
async def list_rooms(
    include_inactive: bool = False,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = select(Room)
    if not include_inactive:
        stmt = stmt.where(Room.is_active == True)  # noqa: E712
    stmt = stmt.order_by(Room.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    rooms = list(result.scalars())
    room_responses = [RoomResponse.model_validate(room, from_attributes=True) for room in rooms]
    return RoomListResponse(rooms=room_responses, total=len(room_responses))


@router.get("/rooms/{room_id}", response_model=RoomResponse)
async def get_room(
    room_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    room = await db.get(Room, room_id)
    if room is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Room {room_id} not found")
    return room


@router.put("/rooms/{room_id}", response_model=RoomResponse)
async def update_room(
    room_id: int,
    req: RoomUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    room = await db.get(Room, room_id)
    if room is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Room {room_id} not found")

    if req.name is not None:
        stmt = select(Room).where(Room.name == req.name)
        existing_name = (await db.execute(stmt)).scalars().first()
        if existing_name is not None and existing_name.id != room_id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Image already exists.")
        room.name = req.name
    # if req.description is not None:
    #     room.description = req.description
    if req.is_active is not None:
        room.is_active = req.is_active

    if req.image_base64 is not None:
        try:
            image_bytes, _ext = s3_service.decode_base64_image(req.image_base64)
        except s3_service.InvalidBase64ImageError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        stmt = select(Room).where(Room.image_hash == image_hash)
        existing = (await db.execute(stmt)).scalars().first()
        if existing is not None and existing.id != room_id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Image already exists.")

        new_url, new_key = _upload_or_400(req.image_base64, req.name or room.name, config.S3_ROOM_PREFIX)
        old_key = room.s3_object_key
        room.s3_object_key = new_key
        room.image_hash = image_hash
        try:
            s3_service.delete_object(old_key)
        except s3_service.S3UploadError:
            logger.warning("Old room image %s could not be deleted from S3", old_key)

    db.add(room)
    await db.commit()
    await db.refresh(room)
    return room


@router.delete("/rooms/{room_id}")
async def delete_room(
    room_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    room = await db.get(Room, room_id)
    if room is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Room {room_id} not found")

    try:
        s3_service.delete_object(room.s3_object_key)
    except s3_service.S3UploadError:
        logger.warning("Room image %s could not be deleted from S3", room.s3_object_key)

    await db.delete(room)
    await db.commit()
    return {"message": f"Room {room_id} deleted"}


# ---------------------------------------------------------------------------
# Use an existing saved room: process the saved image and return a room_token
# so the UI can render fabrics against it without re-uploading from client.
# ---------------------------------------------------------------------------
@router.post("/use-room/{room_id}", response_model=UploadRoomResponse)
async def use_room(room_id: int, db: AsyncSession = Depends(get_db)):
    from . import curtain_service, room_catalog

    item = room_catalog.get_room(str(room_id))
    if item is not None:
        path = room_catalog.resolve_room_image_path(item)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Room image file missing on disk")

        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to read saved room image")
    else:
        room = await db.get(Room, room_id)
        if room is None:
            raise HTTPException(status_code=404, detail=f"room_id '{room_id}' not found")

        try:
            image_bytes = s3_service.download_object(room.s3_object_key)
        except s3_service.S3UploadError as e:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"Failed to download room image: {e}")

        image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=500, detail="Failed to decode saved room image")

    image_bgr = _resize_large_image(image_bgr, max_dim=1280)

    try:
        room = curtain_service.process_upload(image_bgr, model_key="curtain")
    except curtain_service.NoCurtainsDetectedError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Processing saved room failed")
        raise HTTPException(status_code=500, detail=str(e))

    curtains_info = [
        CurtainInfo(
            curtain_id=c.curtain_id,
            bbox=c.bbox,
            area_px=int(cv2.countNonZero(c.mask)),
            confidence=c.confidence,
        )
        for c in room.curtains.values()
    ]

    return UploadRoomResponse(
        room_token=room.room_token,
        model_key="curtain",
        preview_image_base64=_encode_bgr_to_base64(room.original_image),
        curtains_detected=len(room.curtains),
        curtains=curtains_info,
        expires_in_seconds=config.ROOM_CACHE_TTL_SECONDS,
    )


# ---------------------------------------------------------------------------
# GET /api/models  -- lets the frontend populate a garment-type selector
# ---------------------------------------------------------------------------
@router.get("/models", response_model=ModelListResponse)
async def list_models():
    models = [
        {
            "model_key": key,
            "display_name": spec.display_name,
            "render_enabled": spec.render_enabled,
        }
        for key, spec in config.MODEL_REGISTRY.items()
    ]
    return ModelListResponse(models=models)


# ---------------------------------------------------------------------------
# POST /api/upload-room
#
# model_key is optional and defaults to "curtain" -- existing clients that
# never send it get byte-for-byte the same behavior as before this change.
# ---------------------------------------------------------------------------
@router.post("/upload-room", response_model=UploadRoomResponse, responses={400: {"model": ErrorResponse}})
async def upload_room_file(
    file: UploadFile = File(...),
    previous_room_token: str | None = Form(default=None),
    model_key: str = Form(default="curtain"),
):
    _validate_model_key(model_key)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in config.ALLOWED_IMAGE_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'")

    file_bytes = await file.read()
    if len(file_bytes) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {config.MAX_UPLOAD_MB}MB limit")

    image_bgr = _read_upload_to_bgr(file_bytes)
    image_bgr = _resize_large_image(image_bgr, max_dim=1280)

    try:
        room = curtain_service.process_upload(
            image_bgr, model_key=model_key, previous_room_token=previous_room_token,
        )
    except curtain_service.NoCurtainsDetectedError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        logger.exception("Room analysis failed: missing model file")
        raise HTTPException(status_code=500, detail=f"Model file missing: {e}")
    except RuntimeError as e:
        logger.exception("Room analysis failed: missing runtime dependency")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Room analysis failed")
        raise HTTPException(status_code=500, detail=f"Room analysis failed: {str(e)}")

    curtains_info = [
        CurtainInfo(
            curtain_id=c.curtain_id,
            bbox=c.bbox,
            area_px=int(cv2.countNonZero(c.mask)),
            confidence=c.confidence,
        )
        for c in room.curtains.values()
    ]

    return UploadRoomResponse(
        room_token=room.room_token,
        model_key=model_key,
        preview_image_base64=_encode_bgr_to_base64(room.original_image),
        curtains_detected=len(room.curtains),
        curtains=curtains_info,
        expires_in_seconds=config.ROOM_CACHE_TTL_SECONDS,
    )


# ---------------------------------------------------------------------------
# POST /api/render-curtain
#
# Works for any model_key whose ModelSpec has render_enabled=True. For a
# detection-only model (e.g. kurta today) this returns 409 with a clear
# message rather than a distorted render.
# ---------------------------------------------------------------------------
@router.post(
    "/render-curtain",
    response_model=RenderCurtainResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def render_curtain(req: RenderCurtainRequest, db: AsyncSession = Depends(get_db)):
    try:
        image, elapsed_ms = await curtain_service.render_fabric_on_item(
            room_token=req.room_token,
            fabric_id=req.fabric_id,
            item_ids=req.curtain_ids,
            db=db,
        )
    except curtain_service.RoomNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except curtain_service.FabricNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except curtain_service.RenderingNotSupportedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Render failed")
        raise HTTPException(status_code=500, detail="Render failed")

    return RenderCurtainResponse(
        room_token=req.room_token,
        fabric_id=req.fabric_id,
        rendered_image_base64=_encode_bgr_to_base64(image),
        render_time_ms=round(elapsed_ms, 2),
    )
