"""Interfaces of request value encoding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import datetime


class IValueFormatter(Protocol):
    """Turns dates, times and durations into the text an API expects.

    A method declares one in ``__value_formatter__``; every part of the
    request (body, form, query and path) goes through it, so a value is
    written the same way wherever it is sent.
    """

    def format_datetime(self, value: datetime.datetime, /) -> str:
        """Write a moment in time.

        Args:
            value: The datetime to write.

        Returns:
            Its text.
        """
        ...

    def format_date(self, value: datetime.date, /) -> str:
        """Write a calendar date.

        Args:
            value: The date to write.

        Returns:
            Its text.
        """
        ...

    def format_time(self, value: datetime.time, /) -> str:
        """Write a time of day.

        Args:
            value: The time to write.

        Returns:
            Its text.
        """
        ...

    def format_timedelta(self, value: datetime.timedelta, /) -> str:
        """Write a duration.

        Args:
            value: The duration to write.

        Returns:
            Its text.
        """
        ...
