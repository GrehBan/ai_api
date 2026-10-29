"""Interface for request authentication."""

from typing import Protocol


class IAuth(Protocol):
    """Produces the value of the ``Authorization`` header."""

    def encode(self) -> str:
        """Return the ``Authorization`` header value.

        Returns:
            The full header value, including the scheme.
        """
        ...
