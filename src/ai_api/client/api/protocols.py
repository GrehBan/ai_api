"""Interface describing where an API lives."""

from collections.abc import Mapping
from typing import Protocol

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
