"""Interface for request authentication."""

from typing import Protocol


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
