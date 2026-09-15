"""
Diagnostic: run this directly to see EXACTLY what a given registered model
outputs, before detection.py applies any filtering.

Usage:
    python diagnose_model.py --model curtain path/to/room_image.jpg
    python diagnose_model.py --model kurta path/to/person_image.jpg
"""
import argparse
import sys

from ultralytics import YOLO

from app import config
from app.model_manager import ModelManager, UnknownModelError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--model", default=config.DEFAULT_MODEL_KEY,
                         help=f"Model key to diagnose. Available: {ModelManager.available_models()}")
    args = parser.parse_args()

    try:
        spec = ModelManager.get_spec(args.model)
    except UnknownModelError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    model = YOLO(spec.weights_path)

    print("=" * 60)
    print(f"MODEL: {args.model} ({spec.display_name})  weights={spec.weights_path}")
    print("MODEL CLASS NAMES (result.names):")
    print(model.names)
    print("=" * 60)

    # run with a very low confidence so nothing gets silently dropped
    results = model.predict(source=args.image_path, conf=0.05, iou=0.5, verbose=False)
    result = results[0]

    print(f"\nHas masks? {result.masks is not None}")
    print(f"Number of detections: {len(result.boxes) if result.boxes is not None else 0}\n")

    if result.boxes is not None:
        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            cls_name = result.names.get(cls_id, str(cls_id))
            conf = float(result.boxes.conf[i].item())
            xyxy = result.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            print(f"  [{i}] class='{cls_name}' (id={cls_id})  conf={conf:.3f}  bbox={xyxy}")

    print("\n" + "=" * 60)
    print(f"config.py currently filters '{args.model}' to class_names:")
    print(spec.class_names)
    print(f"CONF_THRESHOLD in use: {spec.conf_threshold}")
    print(f"MIN_MASK_AREA_PX in use: {spec.min_mask_area_px}")
    print(f"render_enabled: {spec.render_enabled}"
          + ("  (full fabric-swap render pipeline available)" if spec.render_enabled
             else "  (detection-only -- /api/render-curtain will return 409 for this model)"))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
