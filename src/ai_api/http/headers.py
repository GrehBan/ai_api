"""Header names and values used by the library, and header parsing."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Final

from multidict import CIMultiDict, istr

from ai_api.__meta__ import __version__
from ai_api.http.protocols import (
    HeadersDict,
    HeadersMapping,
    HeadersType,
    IHeaders,
)

USER_AGENT: Final[istr] = istr("User-Agent")
SERVER_SOFTWARE: Final[str] = (
    f"Python/{sys.version_info[0]}.{sys.version_info[1]} ai_api/{__version__}"
)
"""Value of the ``User-Agent`` header sent by the library."""
CONTENT_TYPE: Final[istr] = istr("Content-Type")
AUTHORIZATION: Final[istr] = istr("Authorization")
LOCATION: Final[istr] = istr("Location")
RETRY_AFTER: Final[istr] = istr("Retry-After")
REQUEST_ID: Final[istr] = istr("Request-Id")
X_REQUEST_ID: Final[istr] = istr("X-Request-Id")
X_API_KEY: Final[istr] = istr("X-Api-Key")
ANTHROPIC_VERSION: Final[istr] = istr("Anthropic-Version")

JSON_CONTENT_TYPE: Final[str] = "application/json"
"""Media type of JSON request bodies."""
EVENT_STREAM_CONTENT_TYPE: Final[str] = "text/event-stream"
"""Media type of server-sent event streams."""


def get_header(
    headers: Mapping[str, str],
    name: str,
) -> str | None:
    """Look up a header case-insensitively.

    Args:
        headers: Response headers, in any mapping type.
        name: Header name to look up.

    Returns:
        The first matching header value, or ``None`` if it is absent.
    """
    name = name.lower()

    for key, value in headers.items():
        if key.lower() == name:
            return value

    return None


def _now() -> datetime:
    """Read the clock, for the date form of ``Retry-After``.

    The one place the library reads the time, so tests fix it here alone.

    Returns:
        The current time in UTC.
    """
    return datetime.now(UTC)


def parse_retry_after(
    headers: Mapping[str, str],
) -> float | None:
    """Read the ``Retry-After`` header as a number of seconds.

    Only the two forms of RFC 9110 (section 10.2.3) are read: a whole
    number of seconds in ASCII digits and an HTTP date. A date in the
    past yields ``0.0``. Anything else (a fraction, a sign, ``1e3``,
    non-ASCII digits) is invalid and counts as no header: a negative
    number, for one, would otherwise mean an immediate retry.

    Args:
        headers: Response headers.

    Returns:
        Seconds to wait, or ``None`` if the header is absent or invalid.
    """
    value = get_header(headers, RETRY_AFTER)

    if value is None:
        return None

    value = value.strip()

    if value.isascii() and value.isdigit():
        return float(value)

    try:
        date = parsedate_to_datetime(value)
    except (ValueError, TypeError, IndexError):
        return None

    if date.tzinfo is None:
        date = date.replace(tzinfo=UTC)

    return max(0.0, (date - _now()).total_seconds())


def _as_mapping(headers: HeadersType) -> HeadersMapping:
    """Return the headers of a mapping or a collection as a mapping.

    Args:
        headers: A mapping, or an ``IHeaders`` collection.

    Returns:
        The mapping itself, or the collection's stored headers.
    """
    if isinstance(headers, IHeaders):
        return headers.headers

    return headers


class Headers:
    """Case-insensitive collection of default request headers.

    Attributes:
        headers: The stored headers.
    """

    def __init__(self, headers: HeadersType | None = None) -> None:
        """Create the collection.

        Args:
            headers: Initial headers, as a mapping or another collection,
                which is copied.
        """
        self.headers = CIMultiDict(_as_mapping(headers or {}))

    def add(self, headers: HeadersType) -> None:
        """Add or replace headers.

        Args:
            headers: Headers to merge in, as a mapping or a collection;
                existing names are overwritten.
        """
        self.headers.update(_as_mapping(headers))

    def copy(self) -> HeadersDict:
        """Return an independent copy of the stored headers.

        Returns:
            A new case-insensitive mapping.
        """
        return self.headers.copy()

    def build(self, extra: HeadersType | None = None) -> HeadersDict:
        """Return the stored headers merged with ``extra``.

        The collection itself is not modified.

        Args:
            extra: Headers that take precedence over the stored ones, as
                a mapping or a collection; ``None`` for none.

        Returns:
            A new case-insensitive mapping.
        """
        headers = self.copy()

        if extra is not None:
            headers.update(_as_mapping(extra))

        return headers
