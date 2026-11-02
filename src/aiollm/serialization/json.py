"""JSON codec used by default, backed by ``orjson``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import orjson

type JsonDumps = Callable[[Any], bytes]
type JsonLoads = Callable[[bytes | str], Any]


def dumps(data: Any, /) -> bytes:
    """Serialize ``data`` to JSON.

    Args:
        data: A JSON-compatible value.

    Returns:
        UTF-8 encoded JSON.
    """
    return orjson.dumps(data)


def loads(data: bytes | str, /) -> Any:
    """Parse JSON.

    Args:
        data: JSON text or UTF-8 encoded bytes.

    Returns:
        The decoded value.

    Raises:
        orjson.JSONDecodeError: If ``data`` is not valid JSON. It is a
            subclass of ``ValueError``.
    """
    return orjson.loads(data)
