"""Room catalogue: stores uploaded room images for reuse in the UI.

This mirrors the fabric/curtain catalogue pattern but stores whole-room
photographs (the images users upload when they want to try fabrics on
existing rooms). Entries are exposed under `/api/rooms` and saved to
`config.ROOM_IMAGE_DIR` so they're visible via the `/static/rooms/...`
mount.
"""
from __future__ import annotations

import os
import uuid
from typing import List, Optional

from . import config
from .models import RoomCatalogItem


_ROOM_CATALOG: dict[str, RoomCatalogItem] = {}


def list_rooms() -> List[RoomCatalogItem]:
    return list(_ROOM_CATALOG.values())


def get_room(room_id: str) -> Optional[RoomCatalogItem]:
    return _ROOM_CATALOG.get(room_id)


def search_rooms(query: Optional[str] = None, tags: Optional[List[str]] = None, limit: int = 50) -> List[RoomCatalogItem]:
    results = list(_ROOM_CATALOG.values())
    if query:
        q = query.lower().strip()
        results = [r for r in results if q in r.name.lower()]
    if tags:
        tag_set = {t.lower() for t in tags}
        results = [r for r in results if tag_set.intersection({t.lower() for t in r.tags})]
    return results[:limit]


def register_custom_room(name: str, image_bytes: bytes, ext: str = ".jpg", tags: Optional[List[str]] = None) -> RoomCatalogItem:
    os.makedirs(config.ROOM_IMAGE_DIR, exist_ok=True)
    room_id = f"room_{uuid.uuid4().hex}"
    filename = f"{room_id}{ext}"
    filepath = os.path.join(config.ROOM_IMAGE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    if not os.path.exists(filepath):
        raise RuntimeError(f"Failed to save room image to {filepath}")

    item = RoomCatalogItem(
        room_id=room_id,
        name=name,
        image_url=f"/static/rooms/{filename}",
        tags=tags or [],
        is_custom_upload=True,
    )
    _ROOM_CATALOG[room_id] = item
    return item


def resolve_room_image_path(item: RoomCatalogItem) -> str:
    filename = os.path.basename(item.image_url)
    return os.path.join(config.ROOM_IMAGE_DIR, filename)
