"""API key authentication in a provider-specific header."""

from dataclasses import dataclass, field


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
