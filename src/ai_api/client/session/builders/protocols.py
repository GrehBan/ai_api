"""Interface for building transport-specific request bodies."""

from collections.abc import Mapping
from typing import Any, Protocol

from ai_api.methods.enums import BodyFormat


class IBodyBuilder(Protocol):
    """Turns a method's body fields into HTTP client arguments."""

    def build(
        self,
        body: Mapping[str, Any],
        body_format: BodyFormat,
    ) -> Mapping[str, Any]:
        """Encode the body.

        Args:
            body: Body fields of the method.
            body_format: How the body must be encoded.

        Returns:
            Keyword arguments for the HTTP client's request call.
        """
        ...
