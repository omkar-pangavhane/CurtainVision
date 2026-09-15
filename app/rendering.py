"""
Renders a new fabric onto a curtain using ONLY cached room analysis data
(no re-detection, no re-analysis). This is what runs on every fabric switch
and must stay fast (<1s target per the spec).

Technique -- silhouette-following warp (replaces a single flat perspective
transform, which can only represent a flat tilted plane and therefore can
never make a pattern fan, taper, or bulge the way a real curtain does):
  1. Tile the new fabric to the curtain's average canonical width.
  2. For EVERY row of the curtain, read the actual left/right mask edge
     (cached from analysis.py, smoothed) and stretch that row of the tiled
     fabric to exactly fit the real photographed width at that row. Where
     the curtain fans out, the pattern fans out with it; where it narrows
     into a pleat, the pattern compresses. This is built with a single
     vectorized cv2.remap -- no per-row Python loop -- so it stays fast
     enough for the <1s cache-only render path.
  3. Convert warped fabric to LAB. Multiply/blend its OWN L channel using
     the cached shading_ratio map via an adaptive dark-fabric-aware blend:
         new_L = adaptive_relight(fabric_L, shading_ratio)
     This scales the fabric's own tone rather than replacing it with an
     absolute lighting value, so saturated AND dark colors both stay true
     to the selected fabric while still showing the original curtain's
     folds, pleats and highlights.
  3b. Remap the relit pattern horizontally using fold_displacement_x, so
      fine wrinkle-level pattern lines (e.g. stripes) additionally bend
      around individual fold ridges, on top of the macro silhouette warp
      from step 2.
  4. Recompose LAB -> BGR, feather the (already-smoothed) mask edges, and
     alpha-composite into the full room image.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, Optional

import cv2
import numpy as np

from . import config
from .cache import CurtainMaskData, RoomData
from .texture import prepare_tiled_texture


def render_curtains(
    room: RoomData,
    fabric_bgr: np.ndarray,
    fabric_ppi: Optional[float] = None,
    curtain_ids: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Render the selected fabric onto one or more cached curtains and return
    the full composited room image (BGR uint8)."""
    output = room.original_image.copy()

    targets: Dict[str, CurtainMaskData]
    if curtain_ids is None:
        targets = room.curtains
    else:
        targets = {cid: room.curtains[cid] for cid in curtain_ids if cid in room.curtains}

    for curtain in targets.values():
        output = _render_single_curtain(output, curtain, fabric_bgr, fabric_ppi)

    return output


def _adaptive_relight(L: np.ndarray, shading_ratio: np.ndarray) -> np.ndarray:
    """Blend multiplicative and additive shading transfer based on how dark
    the fabric pixel is, so folds/highlights stay visible on black/navy/dark
    grey fabrics without artificially brightening the fabric's true color.

    - multiplicative = L * shading_ratio  -- exact color preservation, but
      contrast collapses toward zero as L -> 0 (proven correct on light/mid
      fabrics; unchanged for them here).
    - additive = L + (shading_ratio - 1) * gain -- contrast stays visible
      regardless of L, because it's not scaled by L. delta is signed and
      roughly zero-mean across the whole curtain, so this doesn't shift the
      fabric's overall brightness -- it only adds local highlight/shadow
      contrast where the original curtain actually had it.

    dark_blend goes from 0 (pure multiplicative, fabrics at/above the
    threshold luminance) to 1 (pure additive, pure black) as L decreases.
    """
    multiplicative = L * shading_ratio
    delta = shading_ratio - 1.0
    additive = L + delta * config.DARK_FABRIC_ADDITIVE_GAIN

    threshold = max(config.DARK_FABRIC_BLEND_THRESHOLD_L, 1e-4)
    dark_blend = np.clip(1.0 - L / threshold, 0.0, 1.0)

    blended = (1.0 - dark_blend) * multiplicative + dark_blend * additive
    return np.clip(blended, 0.0, 1.0)


def _render_single_curtain(
    room_image: np.ndarray,
    curtain: CurtainMaskData,
    fabric_bgr: np.ndarray,
    fabric_ppi: Optional[float],
) -> np.ndarray:
    x1, y1, x2, y2 = curtain.bbox
    canon_w, canon_h = curtain.canonical_size

    # 1. tile fabric to canonical flat size (scaled to the curtain's own average width)
    tiled = prepare_tiled_texture(fabric_bgr, (canon_w, canon_h), fabric_ppi=fabric_ppi)

    # 2. silhouette-following warp: shift each row's sampling to hug the
    #    actual mask edge, but at a CONSTANT scale across all rows -- using
    #    a per-row scale (dividing by that row's own width) would squeeze
    #    the whole pattern to fit whatever width each row happens to have,
    #    so any row-to-row noise in the detected edge directly warps the
    #    pattern's zoom level and shows up as visible rippling/stretching.
    #    A constant scale keeps pattern SIZE fixed everywhere; only the
    #    alignment (left edge) shifts per row to follow a taper/fan/sway --
    #    which also matches how a real folded curtain works: a narrower
    #    row shows FEWER pattern repeats, not the same repeats squished.
    crop_h = y2 - y1
    crop_w = x2 - x1
    left_x = curtain.row_left_x       # shape (crop_h,)
    right_x = curtain.row_right_x     # shape (crop_h,)
    avg_row_width = float(np.mean(right_x - left_x))
    scale_x = canon_w / max(avg_row_width, 1e-3)

    dest_x = np.arange(crop_w, dtype=np.float32)[None, :]              # (1, crop_w)
    map_x = (dest_x - left_x[:, None]) * scale_x                       # (crop_h, crop_w) -> texture x
    map_x = map_x.astype(np.float32)

    y_frac = (np.arange(crop_h, dtype=np.float32) / max(crop_h - 1, 1))[:, None]
    map_y = np.broadcast_to(y_frac * (canon_h - 1), (crop_h, crop_w)).astype(np.float32)

    warped_crop = cv2.remap(tiled, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    mask_crop = curtain.mask[y1:y2, x1:x2]
    mask_norm = (mask_crop.astype(np.float32) / 255.0)

    # 3. multiplicative shading-ratio transfer -- preserves fabric's own hue/saturation
    lab = cv2.cvtColor(warped_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    new_L = _adaptive_relight(L, curtain.shading_ratio)

    lab_out = np.dstack([new_L * 255.0, A, B]).astype(np.uint8)
    relit_crop = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # 3b. bend the pattern around the original curtain's fold ridges/valleys
    #     (this is what breaks the "flat paper" look -- a straight stripe
    #     now visibly curves around each fold instead of staying rigid)
    crop_h, crop_w = relit_crop.shape[:2]
    y_grid, x_grid = np.indices((crop_h, crop_w), dtype=np.float32)
    map_x = x_grid + curtain.fold_displacement_x
    map_y = y_grid
    relit_crop = cv2.remap(relit_crop, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 4. feathered alpha composite back into the room image (mask was already
    #    morphologically smoothed in detection.py, this just softens the edge)
    feather = max(1, config.MASK_FEATHER_PX)
    alpha = cv2.GaussianBlur(mask_norm, (_odd(feather * 2 + 1), _odd(feather * 2 + 1)), 0)
    alpha3 = np.repeat(alpha[:, :, None], 3, axis=2)

    dest_crop = room_image[y1:y2, x1:x2].astype(np.float32)
    composited = alpha3 * relit_crop.astype(np.float32) + (1 - alpha3) * dest_crop
    room_image[y1:y2, x1:x2] = np.clip(composited, 0, 255).astype(np.uint8)

    return room_image


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def render_and_time(room: RoomData, fabric_bgr: np.ndarray, **kwargs):
    t0 = time.perf_counter()
    image = render_curtains(room, fabric_bgr, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return image, elapsed_ms