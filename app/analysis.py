"""
Room/item analysis performed exactly once per upload.

MULTI-MODEL NOTE
-----------------
Two tiers of analysis, chosen by config.MODEL_REGISTRY[model_key].render_enabled:

  - render_enabled=True (curtain today): full geometry + appearance pipeline
    below (quad fit, shading_ratio, fold_displacement, optional depth). This
    assumes a flat draped sheet of cloth and is what the fabric-swap
    renderer needs.
  - render_enabled=False (kurta today): lightweight record with just
    bbox/mask/confidence -- correct detection output without pretending to
    support a render technique that hasn't been designed/validated for a
    garment on a 3D body. Calling the renderer on one of these raises
    RenderingNotSupportedError in curtain_service.py rather than producing
    distorted output.

For each detected curtain mask (render_enabled=True path), this module
computes everything the renderer needs to reuse on every subsequent fabric
change:

  - quad_corners / perspective_matrix : maps a flat canonical rectangle onto
    the curtain's real, perspective-distorted shape in the room photo.
  - shading_ratio  : a single multiplicative map = original_L / blurred(original_L),
    clipped to a sane range. This is the standard "ratio image" / multiply-blend
    technique for re-texturing a photographed surface: it captures folds,
    pleats, wrinkles, shadows AND highlights all at once as a per-pixel
    brightness *factor* (around 1.0 = neutral, <1 = shadowed fold, >1 = highlight).
    On render, the NEW fabric's own luminance is multiplied by this ratio,
    instead of being replaced by an absolute lighting value -- this is what
    keeps a red fabric looking red instead of washing out to pastel, because
    we're scaling the fabric's own tone rather than overwriting it.
  - depth_map      : optional, from Depth Anything V2.

All outputs are cropped to each item's bbox to keep the cache compact.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from . import config
from .cache import CurtainMaskData
from .detection import RawDetection
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

_depth_pipeline = None


def analyze_item(image_bgr: np.ndarray, det: RawDetection, item_id: str, model_key: str) -> CurtainMaskData:
    """Generic entry point for any registered model. Dispatches to the full
    curtain-style geometry/shading pipeline only if that model_key has
    render_enabled=True; otherwise returns a lightweight detection-only
    record with the same data shape (extra fields left as None/empty)."""
    spec = ModelManager.get_spec(model_key)
    if spec.render_enabled:
        return _analyze_full(image_bgr, det, item_id)
    return _analyze_detection_only(det, item_id)


def analyze_curtain(image_bgr: np.ndarray, det: RawDetection, curtain_id: str) -> CurtainMaskData:
    """Backward-compatible wrapper. Existing callers keep working unchanged;
    equivalent to analyze_item(image_bgr, det, curtain_id, "curtain")."""
    return _analyze_full(image_bgr, det, curtain_id)


def _analyze_detection_only(det: RawDetection, item_id: str) -> CurtainMaskData:
    """Lightweight record for models without a validated render pipeline.
    Populates only what upload-response/detection consumers need; every
    render-only field is left empty so render code fails loudly (via
    RenderingNotSupportedError in curtain_service.py) instead of silently
    operating on garbage geometry."""
    x1, y1, x2, y2 = det.bbox
    return CurtainMaskData(
        curtain_id=item_id,
        bbox=[x1, y1, x2, y2],
        mask=det.mask,
        contour=None,
        quad_corners=None,
        perspective_matrix=None,
        canonical_size=None,
        row_left_x=None,
        row_right_x=None,
        shading_ratio=None,
        fold_displacement_x=None,
        depth_map=None,
        confidence=det.confidence,
    )


def _analyze_full(image_bgr: np.ndarray, det: RawDetection, item_id: str) -> CurtainMaskData:
    x1, y1, x2, y2 = det.bbox
    mask_full = det.mask  # 0/255, full image size

    contour = _largest_contour(mask_full)
    quad = _fit_quad(contour, fallback_bbox=(x1, y1, x2, y2))
    canonical_size, persp_matrix = _fit_perspective(quad)

    crop = image_bgr[y1:y2, x1:x2]
    mask_crop = mask_full[y1:y2, x1:x2]

    row_left_x, row_right_x = _compute_row_bounds(mask_crop)
    avg_width = float(np.mean(row_right_x - row_left_x))
    canonical_size = (max(int(round(avg_width)), 1), mask_crop.shape[0])

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0

    shading_ratio = _compute_shading_ratio(L)
    fold_displacement_x = _compute_fold_displacement(L)

    ratio_std = float(np.std(shading_ratio[mask_crop > 0])) if np.any(mask_crop > 0) else 0.0
    disp_max = float(np.max(np.abs(fold_displacement_x))) if fold_displacement_x.size else 0.0
    logger.info(
        "item_id=%s fold signal: shading_ratio_std=%.4f fold_displacement_max_px=%.2f "
        "(low std/near-zero displacement means the source crop has little visible fold "
        "shading for this technique to transfer -- check mask tightness and source photo contrast)",
        item_id, ratio_std, disp_max,
    )

    depth_map = _compute_depth_map(crop) if config.DEPTH_ENABLED else None

    if depth_map is not None:
        m = (mask_crop > 0).astype(np.float32)
        depth_map *= m

    return CurtainMaskData(
        curtain_id=item_id,
        bbox=[x1, y1, x2, y2],
        mask=mask_full,
        contour=contour,
        quad_corners=quad,
        perspective_matrix=persp_matrix,
        canonical_size=canonical_size,
        row_left_x=row_left_x,
        row_right_x=row_right_x,
        shading_ratio=shading_ratio,
        fold_displacement_x=fold_displacement_x,
        depth_map=depth_map,
        confidence=det.confidence,
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _compute_row_bounds(mask_crop: np.ndarray):
    """Read the ACTUAL left/right edge of the curtain mask at every row.
    This is what drives the silhouette-following warp in rendering.py --
    instead of assuming the curtain is a flat quad, every row gets its own
    true width and position straight from the detected mask, so a fan,
    taper, or bulge in the real photo carries through to the new pattern."""
    h, w = mask_crop.shape
    left_x = np.full(h, -1.0, dtype=np.float32)
    right_x = np.full(h, -1.0, dtype=np.float32)

    row_any = mask_crop > 0
    for y in range(h):
        cols = np.where(row_any[y])[0]
        if cols.size > 0:
            left_x[y] = cols[0]
            right_x[y] = cols[-1] + 1  # exclusive edge

    valid = left_x >= 0
    idx = np.arange(h)
    if not np.any(valid):
        # degenerate: mask had zero pixels in this crop (shouldn't happen,
        # area was already checked) -- fall back to full crop width
        left_x[:] = 0
        right_x[:] = w
    elif not np.all(valid):
        # interpolate rows with no mask pixels (e.g. above/below a tapered
        # curtain within its own bbox) from the nearest valid rows
        left_x = np.interp(idx, idx[valid], left_x[valid]).astype(np.float32)
        right_x = np.interp(idx, idx[valid], right_x[valid]).astype(np.float32)

    k = _odd(min(config.ROW_BOUNDS_SMOOTH_KSIZE, max(3, h // 4)))
    left_x = cv2.GaussianBlur(left_x.reshape(-1, 1), (1, k), 0).flatten()
    right_x = cv2.GaussianBlur(right_x.reshape(-1, 1), (1, k), 0).flatten()

    # guard against smoothing collapsing width to ~0 on a degenerate row
    min_width = 2.0
    too_narrow = (right_x - left_x) < min_width
    if np.any(too_narrow):
        mid = (left_x + right_x) / 2.0
        left_x[too_narrow] = mid[too_narrow] - min_width / 2.0
        right_x[too_narrow] = mid[too_narrow] + min_width / 2.0

    return left_x, right_x


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Mask produced no contour")
    return max(contours, key=cv2.contourArea)


def _fit_quad(contour: np.ndarray, fallback_bbox) -> np.ndarray:
    """Approximate the silhouette as a 4-point quad (TL,TR,BR,BL).

    Curtains are rarely perfect rectangles once draped, so we use the
    minimum-area rotated rect as a robust default and refine with
    approxPolyDP when the contour happens to approximate a quad well
    (common for pleated curtains photographed head-on).
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)  # 4x2, order not guaranteed

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        box = approx.reshape(4, 2).astype(np.float32)

    return _order_quad_points(box.astype(np.float32))


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _fit_perspective(quad: np.ndarray):
    """Build the matrix that warps a flat canonical rectangle -> item quad."""
    top_w = np.linalg.norm(quad[1] - quad[0])
    bot_w = np.linalg.norm(quad[2] - quad[3])
    left_h = np.linalg.norm(quad[3] - quad[0])
    right_h = np.linalg.norm(quad[2] - quad[1])

    canon_w = max(int(round(max(top_w, bot_w))), 1)
    canon_h = max(int(round(max(left_h, right_h))), 1)

    src = np.array([[0, 0], [canon_w - 1, 0], [canon_w - 1, canon_h - 1], [0, canon_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, quad)
    return (canon_w, canon_h), matrix


# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------
def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _compute_shading_ratio(L: np.ndarray) -> np.ndarray:
    """original_L / blurred(original_L), clipped -- the multiplicative
    fold/shadow/highlight map. blurred(L) acts as the 'flat, unshaded' base
    tone; dividing by it isolates the *relative* lighting variation only,
    independent of the original item's own color/brightness. That
    independence is exactly what lets this be reapplied to a completely
    different fabric color without washing it out.

    The raw ratio is further boosted via SHADING_CONTRAST_GAIN -- a plain
    ratio is often visually too subtle to read as "real folds" once applied
    to a different-colored fabric; amplifying deviation from 1.0 makes the
    fold shadows/highlights read clearly instead of looking flat/paper-like."""
    k = _odd(min(config.SHADING_BASE_BLUR_KSIZE, max(3, min(L.shape[:2]) // 2)))
    base = cv2.GaussianBlur(L, (k, k), 0)
    ratio = L / np.clip(base, config.SHADING_RATIO_EPS, None)
    ratio = 1.0 + (ratio - 1.0) * config.SHADING_CONTRAST_GAIN
    return np.clip(ratio, config.SHADING_RATIO_MIN, config.SHADING_RATIO_MAX).astype(np.float32)


def _compute_fold_displacement(L: np.ndarray) -> np.ndarray:
    """Derive a horizontal pixel-displacement map from the original item's
    fold ridges/valleys, so a NEW fabric's pattern can be warped to bend
    around the same folds instead of lying flat on the perspective plane.

    Folds in a hanging curtain run roughly vertically (top to bottom), so:
      - smoothing heavily along y (tall kernel) isolates the left-right
        undulation profile common to a whole fold, averaging out photo
        noise and horizontal weave/pattern detail from the ORIGINAL fabric
        (which we don't want to bake in -- only its 3D shape).
      - keeping the x kernel narrow preserves where individual ridges are.
    A horizontal Sobel gradient of that profile approximates local surface
    tilt; scaling it gives a plausible (not physically exact, but visually
    convincing) per-pixel horizontal displacement.

    NOTE: this vertical-fold assumption is specific to hanging curtains.
    Do not reuse for render_enabled models without checking it still makes
    sense for that garment's fold orientation."""
    h_k = _odd(min(config.FOLD_HORIZONTAL_SMOOTH_KSIZE, max(3, L.shape[1] // 2)))
    v_k = _odd(min(config.FOLD_VERTICAL_SMOOTH_KSIZE, max(3, L.shape[0] // 2)))
    fold_profile = cv2.GaussianBlur(L, (h_k, v_k), 0)

    grad_x = cv2.Sobel(fold_profile, cv2.CV_32F, dx=1, dy=0, ksize=config.FOLD_GRADIENT_SOBEL_KSIZE)
    max_abs = np.abs(grad_x).max()
    if max_abs > 1e-6:
        grad_x = grad_x / max_abs  # normalize to [-1, 1]

    return (grad_x * config.FOLD_DISPLACEMENT_STRENGTH_PX).astype(np.float32)


def _compute_depth_map(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    global _depth_pipeline
    try:
        if _depth_pipeline is None:
            from transformers import pipeline
            logger.info("Loading depth model %s", config.DEPTH_MODEL_NAME)
            _depth_pipeline = pipeline(task="depth-estimation", model=config.DEPTH_MODEL_NAME)
        from PIL import Image
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = _depth_pipeline(Image.fromarray(rgb))
        depth = np.array(result["depth"], dtype=np.float32)
        depth = cv2.resize(depth, (crop_bgr.shape[1], crop_bgr.shape[0]))
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-6:
            return np.zeros(depth.shape, dtype=np.float32)
        return (depth - d_min) / (d_max - d_min)
    except Exception:
        logger.exception("Depth estimation failed; continuing without depth map")
        return None
