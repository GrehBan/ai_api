"""Parsing and escaping of request path templates."""

from __future__ import annotations

import string
from typing import Any, Final
from urllib.parse import quote

_DOT_SEGMENTS: Final[frozenset[str]] = frozenset({".", ".."})


def parse_path_params(template: str) -> frozenset[str]:
    """Extract the placeholder names of a request path template.

    Args:
        template: A path such as ``"/models/{model_id}"``. Literal braces
            are written as ``{{`` and ``}}``.

    Returns:
        The placeholder names.

    Raises:
        ValueError: If the template is malformed or a placeholder is not
            a plain name (``{}``, ``{a.b}``, ``{a[0]}``, ``{a:>5}`` and
            ``{a!r}`` are rejected).
    """
    names: set[str] = set()

    for _, name, spec, conversion in string.Formatter().parse(template):
        if name is None:
            continue

        if not name.isidentifier() or spec or conversion:
            raise ValueError(
                f"Path placeholder {{{name}}} must be a plain field name"
            )

        names.add(name)

    return frozenset(names)


def encode_path_value(name: str, value: Any, safe: str, raw: Any) -> str:
    """Convert a path parameter to an escaped path segment.

    Args:
        name: The parameter name, for error messages.
        value: The JSON-mode dump of the field value.
        safe: Characters left unescaped, e.g. ``"/"``.
        raw: The field value itself, for error messages.

    Returns:
        The percent-encoded value.

    Raises:
        ValueError: If the value is ``None``, dumps to ``null`` (such as
            NaN), is empty, or is a ``.``/``..`` segment, which would
            change the request path.
    """
    if value is None:
        if raw is not None:
            raise ValueError(
                f"Path parameter {name!r} has an unsafe value: {raw!r}"
            )

        raise ValueError(f"Path parameter {name!r} is not set")

    text = str(value).lower() if isinstance(value, bool) else str(value)
    segments = text.split("/") if "/" in safe else [text]

    if any(not segment or segment in _DOT_SEGMENTS for segment in segments):
        raise ValueError(
            f"Path parameter {name!r} has an unsafe value: {text!r}"
        )

    # The escaping is what keeps "../admin" from changing the path.
    return quote(text, safe=safe)
