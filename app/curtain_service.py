"""
Service layer -- the only module routes.py should call into.

Two entrypoints matching the two performance paths in the spec:
  - process_upload(): expensive path, runs ONCE per uploaded image
    (detection + analysis), populates the cache, returns room_token.
  - render_fabric_on_item(): cheap path, runs on EVERY fabric change,
    reads only from cache, never touches YOLO/SAM2/depth again.

MULTI-MODEL NOTE
-----------------
Both entrypoints take a model_key (default "curtain" for backward
compatibility with existing callers/clients that don't send one).
analysis.analyze_item() automatically does full geometry+shading analysis
only for models with render_enabled=True; models with render_enabled=False
(kurta today) still get cached bbox/mask detections, but calling
render_fabric_on_item() against one of them raises
RenderingNotSupportedError instead of silently producing distorted output.

process_room_upload() / render_fabric_on_room() are kept as thin
backward-compatible wrappers around the generic functions, hardcoded to
model_key="curtain", so existing code (including routes.py before this
refactor) keeps working unchanged.
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

import cv2
import numpy as np

from . import analysis, config, s3_service
from .cache import RoomData, room_cache
from .database import AsyncSession
from .detection import detect_objects
from .models import Fabric
from .model_manager import ModelManager, UnknownModelError
from .rendering import render_and_time
from sqlalchemy import select

logger = logging.getLogger(__name__)


class NoCurtainsDetectedError(Exception):
    pass


class RoomNotFoundError(Exception):
    pass


class FabricNotFoundError(Exception):
    pass


class RenderingNotSupportedError(Exception):
    """Raised when render_fabric_on_item() is called for a model_key whose
    ModelSpec has render_enabled=False (e.g. kurta today) -- that model has
    no validated re-texturing technique yet, so we refuse rather than run
    curtain-shaped geometry math on non-curtain content."""


def process_upload(image_bgr: np.ndarray, model_key: str = "curtain",
                    previous_room_token: Optional[str] = None) -> RoomData:
    """Run detection + analysis exactly once for the given model_key, cache
    the result, return it.

    If previous_room_token is provided (client re-uses a session), its cache
    entry is evicted immediately per the "cache expires immediately on new
    upload" requirement.
    """
    if previous_room_token:
        room_cache.delete(previous_room_token)

    try:
        raw_detections = detect_objects(image_bgr, model_key)
    except UnknownModelError as e:
        raise ValueError(str(e))

    if not raw_detections:
        spec = ModelManager.get_spec(model_key)
        raise NoCurtainsDetectedError(f"No {spec.display_name.lower()} detected in the uploaded image")

    items = {}
    for i, det in enumerate(raw_detections):
        item_id = f"{model_key}_{i+1}"
        items[item_id] = analysis.analyze_item(image_bgr, det, item_id, model_key)

    room_token = room_cache.new_token()
    room_data = RoomData(room_token=room_token, original_image=image_bgr, curtains=items, model_key=model_key)
    room_cache.set(room_data)
    return room_data


def process_room_upload(image_bgr: np.ndarray, previous_room_token: Optional[str] = None) -> RoomData:
    """Backward-compatible wrapper; equivalent to
    process_upload(image_bgr, "curtain", previous_room_token)."""
    return process_upload(image_bgr, model_key="curtain", previous_room_token=previous_room_token)


async def render_fabric_on_item(
    room_token: str,
    fabric_id: str,
    item_ids: Optional[List[str]] = None,
    db: AsyncSession = None,
):
    """Cache-only path: load cached room + fabric image, render, return (image, elapsed_ms).

    Raises RenderingNotSupportedError if the room's model_key doesn't support
    rendering (config.MODEL_REGISTRY[model_key].render_enabled is False).
    """
    room = room_cache.get(room_token)
    if room is None:
        raise RoomNotFoundError(f"room_token '{room_token}' not found or expired; re-upload the image")

    if db is None:
        raise ValueError("An AsyncSession db is required to load fabric metadata")

    model_key = getattr(room, "model_key", "curtain")
    spec = ModelManager.get_spec(model_key)
    if not spec.render_enabled:
        raise RenderingNotSupportedError(
            f"'{spec.display_name}' does not support fabric rendering yet -- "
            f"this room_token only has detection data. Use the detection results directly."
        )

    try:
        fabric_id_int = int(fabric_id)
    except (TypeError, ValueError):
        raise FabricNotFoundError(f"fabric_id '{fabric_id}' not found")

    stmt = select(Fabric).where(Fabric.id == fabric_id_int)
    result = await db.execute(stmt)
    fabric = result.scalars().first()
    if fabric is None or not fabric.s3_object_key:
        raise FabricNotFoundError(f"fabric_id '{fabric_id}' not found")

    try:
        image_bytes = s3_service.download_object(fabric.s3_object_key)
    except s3_service.S3UploadError as e:
        raise FabricNotFoundError(f"fabric_id '{fabric_id}' not found") from e

    fabric_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if fabric_bgr is None:
        raise FabricNotFoundError(f"fabric_id '{fabric_id}' not found")

    if item_ids:
        missing = [iid for iid in item_ids if iid not in room.curtains]
        if missing:
            raise ValueError(f"Unknown item_ids for this room_token: {missing}")

    image, elapsed_ms = render_and_time(room, fabric_bgr, curtain_ids=item_ids)
    return image, elapsed_ms


async def render_fabric_on_room(
    room_token: str,
    fabric_id: str,
    curtain_ids: Optional[List[str]] = None,
    db: AsyncSession = None,
):
    """Backward-compatible wrapper; equivalent to
    render_fabric_on_item(room_token, fabric_id, curtain_ids)."""
    return await render_fabric_on_item(
        room_token, fabric_id, item_ids=curtain_ids, db=db
    )


def get_room_or_raise(room_token: str) -> RoomData:
    room = room_cache.get(room_token)
    if room is None:
        raise RoomNotFoundError(f"room_token '{room_token}' not found or expired")
    return room
