"""Bearer token authentication."""

from dataclasses import dataclass, field


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

    def encode(self) -> str:
        """Return the ``Authorization`` header value.

        Returns:
            ``"Bearer <token>"``.
        """
        return f"Bearer {self.token}"
