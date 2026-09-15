# fabric_catalog.py
"""
Fabric catalogue store.

In-memory dict for now, structured so swapping to a real DB (Postgres/Mongo)
later only touches this file -- callers (routes.py) only use the functions
below, never the internal dict.
"""
from __future__ import annotations

import os
import uuid
from typing import List, Optional

from . import config
from .models import FabricMetadata

_CATALOGUE: dict[str, FabricMetadata] = {}


def seed_catalogue(items: List[FabricMetadata]) -> None:
    for item in items:
        _CATALOGUE[item.fabric_id] = item


def list_fabrics() -> List[FabricMetadata]:
    return list(_CATALOGUE.values())


def get_fabric(fabric_id: str) -> Optional[FabricMetadata]:
    return _CATALOGUE.get(fabric_id)


def search_fabrics(
    query: Optional[str] = None,
    pattern: Optional[str] = None,
    color: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 50,
) -> List[FabricMetadata]:
    results = list(_CATALOGUE.values())

    if query:
        q = query.lower().strip()
        results = [f for f in results if q in f.name.lower() or q in f.pattern.lower() or q in f.color.lower()]

    if pattern:
        p = pattern.lower().strip()
        results = [f for f in results if f.pattern.lower() == p]

    if color:
        c = color.lower().strip()
        results = [f for f in results if c in f.color.lower() or (f.color_hex or "").lower() == c]

    if tags:
        tag_set = {t.lower() for t in tags}
        results = [f for f in results if tag_set.intersection({t.lower() for t in f.tags})]

    return results[:limit]


def register_custom_fabric(
    name: str,
    pattern: str,
    color: str,
    image_bytes: bytes,
    ext: str = ".jpg",
    color_hex: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> FabricMetadata:
    os.makedirs(config.FABRIC_IMAGE_DIR, exist_ok=True)
    fabric_id = f"custom_{uuid.uuid4().hex}"
    filename = f"{fabric_id}{ext}"
    filepath = os.path.join(config.FABRIC_IMAGE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    metadata = FabricMetadata(
        fabric_id=fabric_id,
        name=name,
        pattern=pattern,
        color=color,
        color_hex=color_hex,
        texture_image_url=f"/static/fabrics/{filename}",
        tags=tags or [],
        is_custom_upload=True,
    )
    _CATALOGUE[fabric_id] = metadata
    return metadata


def upsert_fabric(metadata: FabricMetadata) -> None:
    """Insert or replace a fabric entry directly.

    Used by curtain_catalog.py to mirror curated curtain-catalogue designs
    into this store under the same id, so the existing /api/render-curtain
    endpoint (which resolves textures via fabric_id) can render a catalogue
    curtain without any changes to curtain_service.py.
    """
    _CATALOGUE[metadata.fabric_id] = metadata


def resolve_fabric_image_path(fabric: FabricMetadata) -> str:
    """Map a fabric's texture_image_url back to a filesystem path for reading."""
    filename = os.path.basename(fabric.texture_image_url)
    return os.path.join(config.FABRIC_IMAGE_DIR, filename)
