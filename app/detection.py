"""
Generic clothing/object detection: any registered YOLOv11-seg model
(primary) with optional SAM2 mask refinement.

Multi-model support: `detect_objects(image_bgr, model_key)` works for any
key registered in config.MODEL_REGISTRY (curtain, kurta, shirt, saree, ...).
Model loading is delegated to ModelManager, so adding a new clothing type
requires zero changes to this file -- just a new ModelSpec entry in
config.py.

`detect_curtains(image_bgr)` is kept as a thin backward-compatible wrapper
around `detect_objects(image_bgr, "curtain")` so existing callers (services,
tests, scripts) do not need to change.

Handles two model types transparently, for any model_key:
  - segmentation model (`yolo segment train`): uses result.masks directly.
  - detection-only model (`yolo detect train`, box-only dataset): no
    result.masks available, so a mask is synthesized per-box via
    spec.box_to_mask_method (GrabCut by default).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from . import config
from .config import ModelSpec
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

_sam2_predictor = None


def _get_sam2_predictor():
    global _sam2_predictor
    if _sam2_predictor is None and config.SAM2_ENABLED:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        logger.info("Loading SAM2 refinement model from %s", config.SAM2_CHECKPOINT_PATH)
        sam2_model = build_sam2(config.SAM2_CONFIG_PATH, config.SAM2_CHECKPOINT_PATH)
        _sam2_predictor = SAM2ImagePredictor(sam2_model)
    return _sam2_predictor


@dataclass
class RawDetection:
    bbox: List[int]           # [x1, y1, x2, y2]
    mask: np.ndarray          # uint8 HxW, same size as input image, 0/255
    confidence: float
    class_name: str
    model_key: str = ""       # which registered model produced this (e.g. "curtain", "kurta")


def detect_objects(image_bgr: np.ndarray, model_key: str) -> List[RawDetection]:
    """Run the registered YOLO model for `model_key`, keep only its
    configured classes, optionally refine with SAM2.

    Raises model_manager.UnknownModelError if model_key isn't registered
    in config.MODEL_REGISTRY.
    """
    spec = ModelManager.get_spec(model_key)
    model = ModelManager.get_model(model_key)
    h, w = image_bgr.shape[:2]

    predict_kwargs = {
        "source": image_bgr,
        "conf": spec.conf_threshold,
        "iou": spec.iou_threshold,
        "verbose": False,
    }
    max_dim = max(h, w)
    if max_dim > 1280:
        predict_kwargs["imgsz"] = 1280
    results = model.predict(**predict_kwargs)
    if not results:
        return []

    result = results[0]
    detections: List[RawDetection] = []

    has_seg_masks = result.masks is not None
    if not has_seg_masks:
        logger.info(
            "Model '%s' produced no segmentation masks (likely trained with `yolo detect`, "
            "not `yolo segment`) -- falling back to box-to-mask via '%s'.",
            model_key, spec.box_to_mask_method,
        )

    names = result.names  # class_id -> name
    boxes = result.boxes
    if boxes is None:
        return []

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        if cls_name.lower() not in spec.class_names:
            continue  # exclude classes outside this model's registered set

        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()

        if has_seg_masks:
            mask_small = result.masks.data[i].cpu().numpy()  # model-resolution mask
            mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_full = (mask_full > 0.5).astype(np.uint8) * 255
        else:
            mask_full = _box_to_mask(image_bgr, xyxy, spec)

        mask_full = _smooth_mask(mask_full)  # fixes jagged/staircase edges from either source

        if cv2.countNonZero(mask_full) < spec.min_mask_area_px:
            continue

        detections.append(RawDetection(
            bbox=xyxy, mask=mask_full, confidence=conf,
            class_name=cls_name, model_key=model_key,
        ))

    if config.SAM2_ENABLED and detections:
        detections = _refine_with_sam2(image_bgr, detections)

    return detections


def detect_curtains(image_bgr: np.ndarray) -> List[RawDetection]:
    """Backward-compatible wrapper. Existing callers keep working unchanged;
    equivalent to detect_objects(image_bgr, "curtain")."""
    return detect_objects(image_bgr, "curtain")


def detect_kurtas(image_bgr: np.ndarray) -> List[RawDetection]:
    """Convenience wrapper, equivalent to detect_objects(image_bgr, "kurta")."""
    return detect_objects(image_bgr, "kurta")

def detect_sofas(image_bgr: np.ndarray) -> List[RawDetection]:
    """Convenience wrapper, equivalent to detect_objects(image_bgr, "sofa")."""
    return detect_objects(image_bgr, "sofa")

def _smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological close (fill small gaps/notches) + open (remove small
    spurs) + Gaussian blur/rethreshold (round off pixel-staircase edges).
    Applied to every mask -- segmentation-derived or box-fallback-derived,
    for any model -- since both can otherwise produce jagged boundaries."""
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (config.MASK_MORPH_CLOSE_KSIZE, config.MASK_MORPH_CLOSE_KSIZE))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (config.MASK_MORPH_OPEN_KSIZE, config.MASK_MORPH_OPEN_KSIZE))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)

    blur_k = config.MASK_SMOOTH_BLUR_KSIZE
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1
    m = cv2.GaussianBlur(m, (blur_k, blur_k), 0)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def _box_to_mask(image_bgr: np.ndarray, bbox: List[int], spec: ModelSpec) -> np.ndarray:
    """Synthesize a mask from a bounding box when no segmentation output is
    available. See spec.box_to_mask_method."""
    if spec.box_to_mask_method == "rect":
        return _rect_mask(image_bgr.shape[:2], bbox)
    return _grabcut_mask(image_bgr, bbox, spec)


def _rect_mask(hw, bbox: List[int]) -> np.ndarray:
    h, w = hw
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _grabcut_mask(image_bgr: np.ndarray, bbox: List[int], spec: ModelSpec) -> np.ndarray:
    """Refine a box into a tighter mask via OpenCV GrabCut, seeded by the box
    as the foreground rectangle. Falls back to a filled rectangle if GrabCut
    fails (e.g. degenerate/too-small box)."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 - x1 < 4 or y2 - y1 < 4:
        logger.warning("GrabCut skipped: degenerate box %s -- using rect fallback", bbox)
        return _rect_mask((h, w), bbox)

    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)

    try:
        cv2.grabCut(image_bgr, gc_mask, rect, bgd_model, fgd_model,
                    spec.grabcut_iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        logger.exception("GrabCut raised an error, falling back to rectangle mask")
        return _rect_mask((h, w), bbox)

    binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    box_area = (x2 - x1) * (y2 - y1)
    fg_area = cv2.countNonZero(binary)
    coverage = fg_area / max(box_area, 1)
    logger.info("GrabCut result for box %s: coverage=%.2f (%d/%d px)", bbox, coverage, fg_area, box_area)

    if fg_area < spec.min_mask_area_px:
        logger.warning(
            "GrabCut collapsed to near-empty (%.2f%% of box) -- falling back to rect mask. "
            "This usually means low contrast between subject and background in this photo.",
            coverage * 100,
        )
        return _rect_mask((h, w), bbox)

    if coverage > 0.97:
        logger.warning(
            "GrabCut returned %.2f%% of the box unchanged -- it likely couldn't separate the "
            "subject from background (low contrast) and this is effectively a rectangle, not a real silhouette.",
            coverage * 100,
        )

    return binary


def _refine_with_sam2(image_bgr: np.ndarray, detections: List[RawDetection]) -> List[RawDetection]:
    """Use YOLO boxes as prompts into SAM2 for tighter, cleaner masks.
    Model-agnostic: works the same regardless of which model_key produced
    the boxes."""
    predictor = _get_sam2_predictor()
    if predictor is None:
        return detections

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    refined: List[RawDetection] = []
    for det in detections:
        box_arr = np.array(det.bbox)
        masks, scores, _ = predictor.predict(box=box_arr, multimask_output=False)
        if masks is None or len(masks) == 0:
            refined.append(det)
            continue
        refined_mask = (masks[0].astype(np.uint8)) * 255
        min_area = ModelManager.get_spec(det.model_key).min_mask_area_px if det.model_key else config.MIN_CURTAIN_MASK_AREA_PX
        if cv2.countNonZero(refined_mask) < min_area:
            refined.append(det)
            continue
        refined.append(RawDetection(
            bbox=det.bbox, mask=refined_mask,
            confidence=det.confidence, class_name=det.class_name,
            model_key=det.model_key,
        ))
    return refined
