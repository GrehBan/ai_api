"""Interface and types for default request headers."""

from collections.abc import Mapping
from typing import Any, Protocol

from multidict import CIMultiDict

type HeadersDict = CIMultiDict[Any]
type HeadersMapping = Mapping[str, Any]


class IHeaders(Protocol):
    """Default headers the session sends with every request."""

    @property
    def headers(self) -> HeadersDict:
        """The stored headers."""
        ...

    def add(self, headers: HeadersMapping) -> None:
        """Add or replace headers.

        Args:
            headers: Headers to merge in.
        """
        ...

    def copy(self) -> HeadersDict:
        """Return an independent copy of the stored headers.

        Returns:
            A new case-insensitive mapping.
        """
        ...

    def build(self, extra: HeadersMapping) -> HeadersDict:
        """Return the stored headers merged with ``extra``.

        Args:
            extra: Headers that take precedence over the stored ones.

        Returns:
            A new case-insensitive mapping.
        """
        ...
