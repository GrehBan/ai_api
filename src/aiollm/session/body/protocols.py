"""Interface for building transport-specific request bodies."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Protocol

from aiollm.enums import BodyFormat

if TYPE_CHECKING:
    from aiollm.serialization.protocols import IValueFormatter


class IBodyBuilder(Protocol):
    """Turns a method's body fields into HTTP client arguments."""

    def build(
        self,
        body: Mapping[str, Any],
        body_format: BodyFormat,
        formatter: IValueFormatter,
        *,
        uploads: AsyncExitStack,
    ) -> Mapping[str, Any]:
        """Encode the body.

        Args:
            body: Body fields of the method.
            body_format: How the body must be encoded.
            formatter: The method's ``value_formatter``, which writes the
                dates, times and durations.
            uploads: The stack of the attempt: whatever the body reads
                from (an upload source) is registered on it, and the
                session closes it when the attempt ends, so a retry can
                read the source again.

        Returns:
            Keyword arguments for the HTTP client's request call.
        """
        ...
