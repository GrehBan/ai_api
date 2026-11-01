"""Parsing and escaping of request path templates."""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import quote

from ai_api.serialization.values import scalar_text

if TYPE_CHECKING:
    from ai_api.serialization.protocols import IValueFormatter

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


def encode_path_value(
    name: str,
    value: Any,
    safe: str,
    formatter: IValueFormatter,
) -> str:
    """Convert a path parameter to an escaped path segment.

    Args:
        name: The parameter name, for error messages.
        value: The Python-mode dump of the field value.
        safe: Characters left unescaped, e.g. ``"/"``.
        formatter: Writes the dates, times and durations.

    Returns:
        The percent-encoded value, written by the rule of ``scalar_text``.

    Raises:
        TypeError: If the value is an object or a collection, which has
            no single path segment.
        ValueError: If the value is ``None``, NaN or infinite, rejected by
            the formatter, empty, or a ``.``/``..`` segment, which would
            change the request path.
    """
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        raise TypeError(
            f"Path parameter {name!r} must be a scalar, not {value!r}"
        )

    if value is None:
        raise ValueError(f"Path parameter {name!r} is not set")

    text = scalar_text(value, formatter, label=f"Path parameter {name!r}")
    segments = text.split("/") if "/" in safe else [text]

    if any(not segment or segment in _DOT_SEGMENTS for segment in segments):
        raise ValueError(
            f"Path parameter {name!r} has an unsafe value: {text!r}"
        )

    # The escaping is what keeps "../admin" from changing the path.
    return quote(text, safe=safe)
