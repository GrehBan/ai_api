"""Default headers sent with every request."""

from multidict import CIMultiDict

from ai_api.client.headers.protocols import HeadersDict, HeadersMapping


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
