"""
Prepares an uploaded fabric swatch for rendering onto a curtain.

A fabric image is just a sample swatch (e.g. 512x512 px of a repeating
pattern). Before it can be warped onto a curtain of arbitrary canonical
size, we need to tile it to cover that size while:
  - keeping the physical pattern scale correct (not stretching the swatch
    to fit -- that would make a small floral print look huge or tiny)
  - preserving orientation (no arbitrary rotation/flipping)
  - avoiding hard seams between tiles (mirror-tiling)
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from . import config


def prepare_tiled_texture(
    fabric_bgr: np.ndarray,
    target_size: Tuple[int, int],
    fabric_ppi: Optional[float] = None,
    mirror_tile: bool = False,
) -> np.ndarray:
    """
    Args:
        fabric_bgr: the raw fabric swatch image, BGR.
        target_size: (width, height) in pixels of the canonical curtain
            rectangle this texture will be warped onto.
        fabric_ppi: pixels-per-inch the swatch was captured/scanned at, if
            known (e.g. from fabric metadata). Falls back to
            config.FABRIC_TARGET_PPI so pattern scale stays consistent
            across fabrics that don't carry DPI metadata.
        mirror_tile: use mirror-flip tiling (True) vs. plain repeat (False).
            Mirror tiling removes hard seam lines for most weaves/prints;
            turn off for directional patterns (e.g. one-way stripes/damask)
            where a mirrored tile would look wrong -- gate this off fabric
            metadata's `pattern` field upstream if needed.

    Returns:
        BGR image of exactly `target_size`, tiled from the swatch.
    """
    target_w, target_h = target_size
    target_w, target_h = max(int(target_w), 1), max(int(target_h), 1)

    swatch = _normalize_scale(fabric_bgr, target_w, fabric_ppi)
    tile_h, tile_w = swatch.shape[:2]

    tiles_x = int(np.ceil(target_w / tile_w)) + 1
    tiles_y = int(np.ceil(target_h / tile_h)) + 1

    if mirror_tile:
        canvas = _mirror_tile(swatch, tiles_x, tiles_y)
    else:
        canvas = np.tile(swatch, (tiles_y, tiles_x, 1))

    # center-crop to exact target size so the pattern isn't biased to one edge
    ch, cw = canvas.shape[:2]
    y0 = max((ch - target_h) // 2, 0)
    x0 = max((cw - target_w) // 2, 0)
    cropped = canvas[y0:y0 + target_h, x0:x0 + target_w]

    # guard against rounding shortfalls
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return cropped


def _normalize_scale(fabric_bgr: np.ndarray, target_w: int, fabric_ppi: Optional[float]) -> np.ndarray:
    """Resize the swatch to a physically-plausible scale, WITHOUT stretching
    (uniform scale factor only).

    - If fabric_ppi is known (real DPI metadata): scale so FABRIC_TARGET_PPI
      is honored -- the fabric prints at a consistent physical size on every
      curtain, same as a real fabric would.
    - Otherwise (the common case: no DPI metadata on an uploaded swatch):
      scale so the swatch repeats config.FABRIC_TARGET_REPEATS_ACROSS_WIDTH
      times across THIS curtain's canonical width. This adapts to the actual
      curtain size instead of tiling at the swatch's raw pixel resolution,
      which is what produced randomly too-wide or too-narrow stripes before.
    """
    h, w = fabric_bgr.shape[:2]

    if fabric_ppi and fabric_ppi > 0:
        scale = config.FABRIC_TARGET_PPI / fabric_ppi
    else:
        if config.FABRIC_TARGET_REPEATS_ACROSS_WIDTH > 0:
            desired_tile_w = target_w / config.FABRIC_TARGET_REPEATS_ACROSS_WIDTH
        else:
            desired_tile_w = target_w * config.FABRIC_AUTO_TILE_WIDTH_FRACTION
            desired_tile_w = np.clip(
                desired_tile_w,
                config.FABRIC_AUTO_TILE_MIN_PX,
                config.FABRIC_AUTO_TILE_MAX_PX,
            )
        scale = desired_tile_w / w

    if abs(scale - 1.0) < 1e-3:
        return fabric_bgr
    new_w, new_h = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(fabric_bgr, (new_w, new_h), interpolation=interp)


def _mirror_tile(swatch: np.ndarray, tiles_x: int, tiles_y: int) -> np.ndarray:
    """Build a seamless-ish canvas by flipping alternate tiles (mirror tiling)."""
    rows = []
    for ty in range(tiles_y):
        row_tiles = []
        for tx in range(tiles_x):
            t = swatch
            if tx % 2 == 1:
                t = cv2.flip(t, 1)  # horizontal flip
            if ty % 2 == 1:
                t = cv2.flip(t, 0)  # vertical flip
            row_tiles.append(t)
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)