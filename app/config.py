# config.py
"""
Central configuration for the Virtual Curtain Try-On service.
"""
import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# MySQL
# ---------------------------------------------------------------------------
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "curtain_app")

MYSQL_USER_ENC = urllib.parse.quote_plus(MYSQL_USER)
MYSQL_PASSWORD_ENC = urllib.parse.quote_plus(MYSQL_PASSWORD)
MYSQL_DB_ENC = urllib.parse.quote_plus(MYSQL_DB)

ASYNC_MYSQL_DRIVER = os.getenv("ASYNC_MYSQL_DRIVER", "aiomysql")
DATABASE_URL = f"mysql+{ASYNC_MYSQL_DRIVER}://{MYSQL_USER_ENC}:{MYSQL_PASSWORD_ENC}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB_ENC}"
DATABASE_SYNC_URL = f"mysql+pymysql://{MYSQL_USER_ENC}:{MYSQL_PASSWORD_ENC}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB_ENC}"

# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------
GOOGLE_CLIENT_ID = os.getenv(
    "GOOGLE_CLIENT_ID",
    "321411570807-8cfo2tgu2v2mqb2rf9st10c5u6vq0ia4.apps.googleusercontent.com",
)

# ---------------------------------------------------------------------------
# AWS S3
# ---------------------------------------------------------------------------
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "aws.tds")

# Folder structure inside the bucket: ai_curtain/fabric/... and ai_curtain/curtain/...
S3_FABRIC_PREFIX = os.getenv("S3_FABRIC_PREFIX", "ai_curtain/fabric")
S3_ROOM_PREFIX = os.getenv("S3_ROOM_PREFIX", "ai_curtain/curtain")

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ROOM_CACHE_TTL_SECONDS = int(os.getenv("ROOM_CACHE_TTL_SECONDS", "600"))
ROOM_CACHE_MAX_ENTRIES = int(os.getenv("ROOM_CACHE_MAX_ENTRIES", "1000"))
ROOM_CACHE_SWEEP_INTERVAL_SECONDS = int(os.getenv("ROOM_CACHE_SWEEP_INTERVAL_SECONDS", "60"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FABRIC_IMAGE_DIR = os.path.join(BASE_DIR, "storage", "fabrics")
ROOM_IMAGE_DIR = os.path.join(BASE_DIR, "storage", "rooms")

# Rendering / texture defaults
DEFAULT_MODEL_KEY = "curtain"
DEPTH_ENABLED = os.getenv("DEPTH_ENABLED", "false").lower() == "true"
DEPTH_MODEL_NAME = os.getenv("DEPTH_MODEL_NAME", "")
FABRIC_TARGET_PPI = int(os.getenv("FABRIC_TARGET_PPI", "150"))
FABRIC_TARGET_REPEATS_ACROSS_WIDTH = int(os.getenv("FABRIC_TARGET_REPEATS_ACROSS_WIDTH", "4"))
FABRIC_AUTO_TILE_WIDTH_FRACTION = float(os.getenv("FABRIC_AUTO_TILE_WIDTH_FRACTION", "0.2"))
FABRIC_AUTO_TILE_MIN_PX = int(os.getenv("FABRIC_AUTO_TILE_MIN_PX", "32"))
FABRIC_AUTO_TILE_MAX_PX = int(os.getenv("FABRIC_AUTO_TILE_MAX_PX", "512"))

MASK_FEATHER_PX = int(os.getenv("MASK_FEATHER_PX", "9"))
MASK_MORPH_CLOSE_KSIZE = int(os.getenv("MASK_MORPH_CLOSE_KSIZE", "15"))
MASK_MORPH_OPEN_KSIZE = int(os.getenv("MASK_MORPH_OPEN_KSIZE", "7"))
MASK_SMOOTH_BLUR_KSIZE = int(os.getenv("MASK_SMOOTH_BLUR_KSIZE", "21"))
MIN_CURTAIN_MASK_AREA_PX = int(os.getenv("MIN_CURTAIN_MASK_AREA_PX", "200"))

ROW_BOUNDS_SMOOTH_KSIZE = int(os.getenv("ROW_BOUNDS_SMOOTH_KSIZE", "31"))
SHADING_BASE_BLUR_KSIZE = int(os.getenv("SHADING_BASE_BLUR_KSIZE", "51"))
SHADING_RATIO_EPS = float(os.getenv("SHADING_RATIO_EPS", "1e-3"))
SHADING_CONTRAST_GAIN = float(os.getenv("SHADING_CONTRAST_GAIN", "1.4"))
SHADING_RATIO_MIN = float(os.getenv("SHADING_RATIO_MIN", "0.5"))
SHADING_RATIO_MAX = float(os.getenv("SHADING_RATIO_MAX", "2.0"))
DARK_FABRIC_ADDITIVE_GAIN = float(os.getenv("DARK_FABRIC_ADDITIVE_GAIN", "0.4"))
DARK_FABRIC_BLEND_THRESHOLD_L = float(os.getenv("DARK_FABRIC_BLEND_THRESHOLD_L", "0.15"))

FOLD_HORIZONTAL_SMOOTH_KSIZE = int(os.getenv("FOLD_HORIZONTAL_SMOOTH_KSIZE", "31"))
FOLD_VERTICAL_SMOOTH_KSIZE = int(os.getenv("FOLD_VERTICAL_SMOOTH_KSIZE", "7"))
FOLD_GRADIENT_SOBEL_KSIZE = int(os.getenv("FOLD_GRADIENT_SOBEL_KSIZE", "3"))
FOLD_DISPLACEMENT_STRENGTH_PX = float(os.getenv("FOLD_DISPLACEMENT_STRENGTH_PX", "18.0"))

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass
class ModelSpec:
    key: str
    display_name: str
    weights_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    class_names: tuple[str, ...] = ("curtain",)
    min_mask_area_px: int = 200
    render_enabled: bool = True
    box_to_mask_method: str = "grabcut"
    grabcut_iterations: int = 5


MODEL_REGISTRY = {
    "curtain": ModelSpec(
        key="curtain",
        display_name="Curtain",
        weights_path="models/curtain_yolov11_seg.pt",
        class_names=("curtain",),
    )
}

# Optional SAM2 support flags
SAM2_ENABLED = os.getenv("SAM2_ENABLED", "false").lower() == "true"
SAM2_CONFIG_PATH = os.getenv("SAM2_CONFIG_PATH", "")
SAM2_CHECKPOINT_PATH = os.getenv("SAM2_CHECKPOINT_PATH", "")
