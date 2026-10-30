"""Interfaces of the HTTP primitives: API location, auth and headers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from multidict import CIMultiDict
from yarl import URL

type Query = Mapping[str, str | list[str]]
"""Query parameters; a list is sent as a repeated key."""


class IAPI(Protocol):
    """Builds request URLs for one API.

    Implemented once per provider; the session asks it for the full URL
    of every request.

    Attributes:
        base_url: Root URL of the API, e.g. ``https://api.example.com``.
        version: API version segment, e.g. ``"v1"``.
    """

    base_url: URL
    version: str

    def build_url(
        self,
        method: str,
        query: Query | None = None,
    ) -> str:
        """Build the absolute URL for a request.

        Args:
            method: Request path of the method, e.g. ``"/models/m%2F1"``.
                It is already percent-encoded and must not be encoded
                again: join it with ``URL.joinpath(..., encoded=True)``,
                not with yarl's ``/`` operator, which would turn ``%2F``
                into ``%252F``.
            query: Query parameters to append, already converted to
                strings, not encoded yet; a list value becomes a repeated
                key (``ids=a&ids=b``).

        Returns:
            The absolute URL to request.
        """
        ...


class IAuth(Protocol):
    """Produces the authentication header of every request."""

    @property
    def header_name(self) -> str:
        """Name of the header, such as ``Authorization`` or ``x-api-key``."""
        ...

    def encode(self) -> str:
        """Return the header value.

        Returns:
            The full header value, including the scheme if any.
        """
        ...


type HeadersDict = CIMultiDict[Any]
type HeadersMapping = Mapping[str, Any]


@runtime_checkable
class IHeaders(Protocol):
    """Default headers the session sends with every request."""

    @property
    def headers(self) -> HeadersDict:
        """The stored headers."""
        ...

    def add(self, headers: HeadersType) -> None:
        """Add or replace headers.

        Args:
            headers: Headers to merge in, as a mapping or a collection.
        """
        ...

    def copy(self) -> HeadersDict:
        """Return an independent copy of the stored headers.

        Returns:
            A new case-insensitive mapping.
        """
        ...

    def build(self, extra: HeadersType | None = None) -> HeadersDict:
        """Return the stored headers merged with ``extra``.

        Args:
            extra: Headers that take precedence over the stored ones;
                ``None`` for none.

        Returns:
            A new case-insensitive mapping.
        """
        ...


type HeadersType = HeadersMapping | IHeaders
"""Headers given as a mapping or as an ``IHeaders`` collection."""
