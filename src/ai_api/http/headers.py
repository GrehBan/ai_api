"""Header names and values used by the library, and header parsing."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Final

from multidict import CIMultiDict, istr

from ai_api.__meta__ import __version__
from ai_api.http.protocols import HeadersDict, HeadersMapping

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


def parse_retry_after(
    headers: Mapping[str, str],
) -> float | None:
    """Read the ``Retry-After`` header as a number of seconds.

    Both forms allowed by RFC 9110 are supported: a number of seconds and
    an HTTP date. Dates in the past yield ``0.0``.

    Args:
        headers: Response headers.

    Returns:
        Seconds to wait, or ``None`` if the header is absent or invalid.
    """
    value = get_header(headers, RETRY_AFTER)

    if value is None:
        return None

    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
        except (ValueError, TypeError, IndexError):
            return None

        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)

        seconds = (date - datetime.now(UTC)).total_seconds()

    if not math.isfinite(seconds):
        return None

    return max(0.0, seconds)


class Headers:
    """Case-insensitive collection of default request headers.

    Attributes:
        headers: The stored headers.
    """

    def __init__(self, headers: HeadersMapping | None = None) -> None:
        """Create the collection.

        Args:
            headers: Initial headers.
        """
        if headers is None:
            headers = {}

        self.headers = CIMultiDict(headers)

    def add(self, headers: HeadersMapping) -> None:
        """Add or replace headers.

        Args:
            headers: Headers to merge in; existing names are overwritten.
        """
        self.headers.update(headers)

    def copy(self) -> HeadersDict:
        """Return an independent copy of the stored headers.

        Returns:
            A new case-insensitive mapping.
        """
        return self.headers.copy()

    def build(self, extra: HeadersMapping) -> HeadersDict:
        """Return the stored headers merged with ``extra``.

        The collection itself is not modified.

        Args:
            extra: Headers that take precedence over the stored ones.

        Returns:
            A new case-insensitive mapping.
        """
        headers = self.headers.copy()
        headers.update(extra)
        return headers
