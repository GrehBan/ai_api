"""Interface for building transport-specific request bodies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from ai_api.enums import BodyFormat

if TYPE_CHECKING:
    from ai_api.serialization.protocols import IValueFormatter


class IBodyBuilder(Protocol):
    """Turns a method's body fields into HTTP client arguments."""

    def build(
        self,
        body: Mapping[str, Any],
        body_format: BodyFormat,
        formatter: IValueFormatter,
    ) -> Mapping[str, Any]:
        """Encode the body.

        Args:
            body: Body fields of the method.
            body_format: How the body must be encoded.
            formatter: The method's ``value_formatter``, which writes the
                dates, times and durations.

        Returns:
            Keyword arguments for the HTTP client's request call.
        """
        ...
