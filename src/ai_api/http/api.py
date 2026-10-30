"""Default ``IAPI`` implementation."""

from __future__ import annotations

import re
from typing import Final

from yarl import URL

from ai_api.http.protocols import Query

_VERSION: Final = re.compile(r"[A-Za-z0-9._~-]*")
"""Unreserved URL characters: a version is joined without encoding."""


def _validate_version(value: str) -> str:
    """Check that a version is one safe path segment.

    Args:
        value: The version, e.g. ``"v1"``; empty for none.

    Returns:
        The version.

    Raises:
        ValueError: If it contains ``/`` or other reserved characters, or
            is ``.`` or ``..``, which would change the path.
    """
    if not _VERSION.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "version must be one path segment of letters, digits, "
            f"'.', '_', '~' and '-': {value!r}"
        )

    return value


def _validate_base_url(value: str | URL) -> URL:
    """Parse and check the root URL of an API.

    Args:
        value: The URL as a string or ``URL``.

    Returns:
        The parsed URL.

    Raises:
        ValueError: If the URL is not an absolute HTTP(S) URL or has a
            query or fragment.
    """
    url = URL(value)

    if url.scheme not in {"http", "https"} or not url.host:
        raise ValueError("base_url must be an absolute HTTP(S) URL")

    if url.query_string or url.fragment:
        raise ValueError("base_url cannot contain query or fragment")

    return url


class BaseAPI:
    """Builds request URLs from a root URL and a version segment.

    Attributes:
        base_url: Root URL of the API; it may end with a path prefix.
        version: API version segment, e.g. ``"v1"``; empty for none.

    Example:
        >>> api = BaseAPI("https://api.example.com", "v1")
        >>> api.build_url("/models/m%2F1", {"limit": "5"})
        'https://api.example.com/v1/models/m%2F1?limit=5'
    """

    def __init__(self, base_url: str | URL, version: str) -> None:
        """Create the API description.

        Args:
            base_url: Root URL, e.g. ``"https://api.example.com"``.
            version: API version segment, e.g. ``"v1"``.

        Raises:
            ValueError: If ``base_url`` is not an absolute HTTP(S) URL or
                has a query or fragment, or ``version`` is not one path
                segment.
        """
        self.base_url = _validate_base_url(base_url)
        self.version = _validate_version(version)

    def build_url(
        self,
        method: str,
        query: Query | None = None,
    ) -> str:
        """Build the absolute URL for a request.

        Args:
            method: Request path of the method, already percent-encoded;
                it is joined without encoding it again.
            query: Query parameters, not encoded yet; a list value
                becomes a repeated key.

        Returns:
            The absolute URL to request.
        """
        url = self.base_url.joinpath(
            self.version,
            method.lstrip("/"),
            encoded=True,
        )
        return str(url.with_query(query or {}))
