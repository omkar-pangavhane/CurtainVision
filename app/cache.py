"""
In-memory, thread-safe, TTL-based cache for cached room analysis.

Design notes for tds:
- Keyed by room_token (uuid4 hex).
- Stores heavy numpy arrays (masks, maps) directly -- fine for single-process
  deployment. For multi-worker/multi-process deployment, swap this out for
  Redis + a serialization layer (msgpack/np.save to bytes) without touching
  callers, since the public interface (get/set/delete/touch) stays the same.
- A background sweep thread evicts expired entries so memory doesn't grow
  unbounded even if users never trigger a new lookup for a stale room_token.
- `uploading a new room image` should call `cache.delete(old_token)` (or just
  let it expire) per the "cache expires immediately on new room upload" spec --
  in practice each upload gets a fresh token, so old tokens simply age out;
  see routes.upload_room for the explicit-delete variant if the client
  resends the same session id.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config


@dataclass
class CurtainMaskData:
    """Per-curtain cached geometry/appearance data."""
    curtain_id: str
    bbox: List[int]                          # [x1, y1, x2, y2] in room image coords
    mask: np.ndarray                         # uint8 HxW, 0/255, full room-image size
    contour: np.ndarray                      # Nx1x2 int32 points (largest contour of mask)
    quad_corners: np.ndarray                 # 4x2 float32, ordered TL,TR,BR,BL -- fitted curtain quad
    perspective_matrix: np.ndarray           # 3x3 float32, canonical-rect -> curtain quad
    canonical_size: tuple                    # (w, h) of the canonical rectangle the matrix warps from
    row_left_x: np.ndarray                   # float32, length = bbox height. Actual mask left edge x per row
                                              # (in crop-local coords), smoothed. Drives the silhouette-following
                                              # warp -- this is what lets the pattern fan/taper with the REAL
                                              # curtain shape instead of one flat idealized quad.
    row_right_x: np.ndarray                  # float32, length = bbox height. Actual mask right edge x per row.
    shading_ratio: np.ndarray                # float32 HxW (cropped to bbox), multiplicative shading factor
                                              # (original_L / blurred(original_L), clipped). Encodes folds,
                                              # pleats, shadows AND highlights in one map. Multiplying the
                                              # NEW fabric's own luminance by this preserves the new fabric's
                                              # color/saturation while still draping like the original curtain.
    fold_displacement_x: np.ndarray          # float32 HxW (cropped to bbox), horizontal pixel displacement
                                              # derived from the original curtain's fold ridges/valleys. Applied
                                              # to the new pattern via remap so pattern lines physically bend
                                              # around folds instead of lying flat on the perspective plane.
    depth_map: Optional[np.ndarray] = None   # float32 HxW (cropped to bbox), optional
    confidence: float = 0.0


@dataclass
class RoomData:
    room_token: str
    original_image: np.ndarray               # BGR uint8, full room image
    curtains: Dict[str, CurtainMaskData]      # curtain_id -> data
    model_key: str = "curtain"               # which registered model produced this room analysis
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_accessed_at = time.time()

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.last_accessed_at) > ttl_seconds


class RoomCache:
    """Thread-safe TTL store, one entry per room_token."""

    def __init__(self, ttl_seconds: int = config.ROOM_CACHE_TTL_SECONDS,
                 max_entries: int = config.ROOM_CACHE_MAX_ENTRIES,
                 sweep_interval: int = config.ROOM_CACHE_SWEEP_INTERVAL_SECONDS):
        self._store: Dict[str, RoomData] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._sweep_interval = sweep_interval
        self._stop_event = threading.Event()
        self._sweeper = threading.Thread(target=self._sweep_loop, daemon=True)
        self._sweeper.start()

    # -- public API -----------------------------------------------------
    def new_token(self) -> str:
        return uuid.uuid4().hex

    def set(self, room_data: RoomData) -> None:
        with self._lock:
            if len(self._store) >= self._max_entries:
                self._evict_oldest_locked()
            self._store[room_data.room_token] = room_data

    def get(self, room_token: str) -> Optional[RoomData]:
        with self._lock:
            entry = self._store.get(room_token)
            if entry is None:
                return None
            if entry.is_expired(self._ttl):
                del self._store[room_token]
                return None
            entry.touch()
            return entry

    def delete(self, room_token: str) -> None:
        with self._lock:
            self._store.pop(room_token, None)

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def shutdown(self) -> None:
        self._stop_event.set()

    # -- internals --------------------------------------------------------
    def _evict_oldest_locked(self) -> None:
        if not self._store:
            return
        oldest_token = min(self._store, key=lambda t: self._store[t].last_accessed_at)
        del self._store[oldest_token]

    def _sweep_loop(self) -> None:
        while not self._stop_event.wait(self._sweep_interval):
            with self._lock:
                expired = [t for t, d in self._store.items() if d.is_expired(self._ttl)]
                for t in expired:
                    del self._store[t]


# Module-level singleton -- import this from routes/service layer.
room_cache = RoomCache()
