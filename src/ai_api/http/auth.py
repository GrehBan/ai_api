"""Request authentication: bearer token and API key."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_api.http.headers import AUTHORIZATION


@dataclass(frozen=True, slots=True)
class BearerAuth:
    """Sends the token as ``Authorization: Bearer <token>``.

    Attributes:
        token: The API key or access token. Excluded from ``repr`` so it
            does not leak into logs.
    """

    token: str = field(repr=False)

    def __post_init__(self) -> None:
        """Validate the token.

        Raises:
            ValueError: If the token is empty or contains whitespace.
        """
        if not self.token:
            raise ValueError("Bearer token cannot be empty")

        if any(char.isspace() for char in self.token):
            raise ValueError("Bearer token cannot contain whitespace")

    @property
    def header_name(self) -> str:
        """Name of the header: ``Authorization``."""
        return AUTHORIZATION

    def encode(self) -> str:
        """Return the ``Authorization`` header value.

        Returns:
            ``"Bearer <token>"``.
        """
        return f"Bearer {self.token}"


@dataclass(frozen=True, slots=True)
class APIKeyAuth:
    """Sends the key as is, in the header named ``header_name``.

    Example:
        >>> auth = APIKeyAuth("sk-ant-...")  # Anthropic: x-api-key
        >>> auth.header_name, auth.encode()
        ('x-api-key', 'sk-ant-...')

    Attributes:
        key: The API key. Excluded from ``repr`` so it does not leak into
            logs.
        header_name: Name of the header that carries the key.
    """

    key: str = field(repr=False)
    header_name: str = "x-api-key"

    def __post_init__(self) -> None:
        """Validate the key and the header name.

        Raises:
            ValueError: If the key or the header name is empty or
                contains whitespace.
        """
        for name, value in (
            ("key", self.key),
            ("header name", self.header_name),
        ):
            if not value:
                raise ValueError(f"API {name} cannot be empty")

            if any(char.isspace() for char in value):
                raise ValueError(f"API {name} cannot contain whitespace")

    def encode(self) -> str:
        """Return the header value.

        Returns:
            The key itself.
        """
        return self.key
