"""Fully read HTTP responses."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BufferedResponse:
    """An HTTP response whose body has been read completely.

    Transport-neutral: the session produces it from whatever HTTP client
    it uses, and parsers consume it.

    Attributes:
        status: HTTP status code.
        headers: Response headers.
        body: The raw response body.
        charset: Charset from the ``Content-Type`` header, if any.
    """

    status: int
    headers: Mapping[str, str]
    body: bytes
    charset: str | None = None
