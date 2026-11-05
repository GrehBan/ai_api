"""Interfaces of the data layer."""

from __future__ import annotations

from typing import Protocol


class IEventMatcher(Protocol):
    """Recognizes items of a stream, such as its terminal or error events.

    A matcher sees validated models, decoded JSON objects and
    ``DONE_MARKER``, so it should answer ``False`` for anything it does
    not understand instead of raising.
    """

    def matches(self, item: object) -> bool:
        """Check one item.

        Args:
            item: A model, a decoded JSON value or ``DONE_MARKER``.

        Returns:
            Whether the item is one the matcher looks for.
        """
        ...
