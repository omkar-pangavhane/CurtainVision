# s3_service.py
"""
All AWS S3 upload/delete/download logic lives here. No other module should
call boto3 directly -- this keeps credentials, bucket name, and key-naming
conventions centralized in one place.

Folder layout in the bucket:
    ai_curtain/fabric/{uuid}.{ext}
    ai_curtain/curtain/{uuid}.{ext}
"""
from __future__ import annotations

import base64
import binascii
import logging
import re
import uuid
from typing import Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from . import config

logger = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(r"^data:image/(?P<ext>[a-zA-Z0-9.+-]+);base64,(?P<data>.+)$", re.DOTALL)

# Normalize the handful of extensions we accept.
_EXT_ALIASES = {"jpg": "jpg", "jpeg": "jpg", "png": "png", "webp": "webp"}


class InvalidBase64ImageError(ValueError):
    """Raised when the supplied string isn't a valid/decodable base64 image."""


class S3UploadError(RuntimeError):
    """Raised when an S3 operation (upload/delete/download) fails."""


def _get_client():
    return boto3.client(
        "s3",
        region_name=config.AWS_REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY_ID or None,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY or None,
    )

def decode_base64_image(base64_str: str, default_ext: str = "jpg") -> Tuple[bytes, str]:
    """Validate + decode a base64 image string.

    Accepts either a raw base64 string or a data URI
    (e.g. "data:image/png;base64,...."). Returns (raw_bytes, extension)
    where extension is one of "jpg", "png", "webp" (no leading dot).
    Raises InvalidBase64ImageError if the string is empty, malformed, or
    doesn't decode to any bytes.
    """
    if not base64_str or not isinstance(base64_str, str):
        raise InvalidBase64ImageError("Image data is empty or not a string")

    ext = default_ext
    payload = base64_str.strip()

    match = _DATA_URI_RE.match(payload)
    if match:
        raw_ext = match.group("ext").lower()
        ext = _EXT_ALIASES.get(raw_ext, default_ext)
        payload = match.group("data")

    # Strip whitespace/newlines that sometimes sneak into copy-pasted base64.
    payload = re.sub(r"\s+", "", payload)

    # Pad if the client stripped trailing '=' padding.
    padding = len(payload) % 4
    if padding:
        payload += "=" * (4 - padding)

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidBase64ImageError(f"Could not decode base64 image data: {exc}") from exc

    if len(image_bytes) == 0:
        raise InvalidBase64ImageError("Decoded image is empty")

    return image_bytes, ext


def _build_key(name: str, prefix: str, ext: str) -> str:
    filename = f"{name}.{ext}"
    return f"{prefix.rstrip('/')}/{filename}"


def _build_url(key: str) -> str:
    return f"https://s3.{config.AWS_REGION}.amazonaws.com/{config.S3_BUCKET_NAME}/{key}"


def upload_base64_image(base64_str: str, name: str, prefix: str) -> Tuple[str, str]:
    """Decode + upload a base64 image to S3 under the given prefix.

    Returns ( s3_object_key). Raises InvalidBase64ImageError for bad
    input and S3UploadError if the PUT to S3 fails.
    """
    image_bytes, ext = decode_base64_image(base64_str)
    key = _build_key(name, prefix, ext)
    content_type = f"image/{'jpeg' if ext == 'jpg' else ext}"

    client = _get_client()
    try:
        client.put_object(
            Bucket=config.S3_BUCKET_NAME,
            Key=key,
            Body=image_bytes,
            ContentType=content_type,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.exception("S3 upload failed for key %s", key)
        raise S3UploadError(f"Failed to upload image to S3: {exc}") from exc

    return _build_url(key), key


def delete_object(s3_key: str) -> None:
    """Delete an object from S3."""
    if not s3_key:
        return
    client = _get_client()
    try:
        client.delete_object(Bucket=config.S3_BUCKET_NAME, Key=s3_key)
    except (BotoCoreError, ClientError) as exc:
        logger.exception("S3 delete failed for key %s", s3_key)
        raise S3UploadError(f"Failed to delete image from S3: {exc}") from exc


def download_object(s3_key: str) -> bytes:
    """Download raw object bytes. Useful if a future feature (e.g. the
    detection/render pipeline) needs pixel data for a catalogued image."""
    client = _get_client()
    try:
        resp = client.get_object(Bucket=config.S3_BUCKET_NAME, Key=s3_key)
        return resp["Body"].read()
    except (BotoCoreError, ClientError) as exc:
        logger.exception("S3 download failed for key %s", s3_key)
        raise S3UploadError(f"Failed to download image from S3: {exc}") from exc
