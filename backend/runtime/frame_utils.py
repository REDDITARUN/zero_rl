"""Helpers for frame encoding and JSON-safe payloads."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image


def encode_frame(frame: Any) -> str:
    """Convert numpy-compatible frame into base64 PNG."""

    arr = np.asarray(frame, dtype=np.uint8)
    image = Image.fromarray(arr)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def to_jsonable(value: Any) -> Any:
    """Recursively convert runtime values into JSON-serializable structures."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
