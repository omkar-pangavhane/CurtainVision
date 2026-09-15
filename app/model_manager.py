"""
ModelManager: lazily loads and caches YOLO models by registry key.

This is the single place in the project responsible for turning a
"model key" (e.g. "curtain", "kurta") into a loaded ultralytics.YOLO
instance. Nothing else should call `YOLO(...)` directly -- that keeps
model loading centralized, cacheable, and easy to extend (e.g. swap in
a different backend for one model type later) without touching
detection.py or the API layer.

Usage:
    from . import model_manager as mm

    model = mm.ModelManager.get_model("kurta")
    spec = mm.ModelManager.get_spec("kurta")
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List

from . import config
from .config import ModelSpec

logger = logging.getLogger(__name__)


class UnknownModelError(KeyError):
    """Raised when a model key isn't in config.MODEL_REGISTRY."""


class ModelManager:
    """Process-wide cache of loaded YOLO models, keyed by registry key.

    Thread-safe lazy loading: two concurrent requests for the same
    not-yet-loaded model will not both trigger a load; the second
    waits for the first's result.
    """

    _models: Dict[str, "object"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_spec(cls, model_key: str) -> ModelSpec:
        spec = config.MODEL_REGISTRY.get(model_key)
        if spec is None:
            raise UnknownModelError(
                f"Unknown model key '{model_key}'. Available: {cls.available_models()}"
            )
        return spec

    @classmethod
    def get_model(cls, model_key: str):
        """Return the loaded ultralytics.YOLO instance for this key, loading
        it on first use. Safe to call from request handlers repeatedly --
        subsequent calls are a dict lookup, not a disk load."""
        if model_key in cls._models:
            return cls._models[model_key]

        with cls._lock:
            # Re-check inside the lock in case another thread loaded it
            # while we were waiting.
            if model_key in cls._models:
                return cls._models[model_key]

            spec = cls.get_spec(model_key)
            try:
                from ultralytics import YOLO  # lazy import: keep module import light for tests
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "ultralytics is not installed. Install backend dependencies with 'pip install -r requirements.txt'"
                ) from exc

            logger.info("Loading '%s' model (%s) from %s", spec.key, spec.display_name, spec.weights_path)
            model = YOLO(spec.weights_path)
            cls._models[model_key] = model
            return model

    @classmethod
    def available_models(cls) -> List[str]:
        """Registry keys, e.g. ['curtain', 'kurta']. Useful for API validation
        and for listing valid values in error messages / docs."""
        return list(config.MODEL_REGISTRY.keys())

    @classmethod
    def is_loaded(cls, model_key: str) -> bool:
        return model_key in cls._models

    @classmethod
    def preload(cls, model_keys: List[str] | None = None) -> None:
        """Optionally call at app startup to load models eagerly instead of
        on first request (avoids the first user of each model eating the
        load-time latency). Defaults to preloading everything registered."""
        for key in (model_keys or cls.available_models()):
            cls.get_model(key)

    @classmethod
    def unload(cls, model_key: str) -> None:
        """Free a loaded model's memory (e.g. to swap weights during a
        hot-reload / retrain without restarting the service)."""
        with cls._lock:
            cls._models.pop(model_key, None)
