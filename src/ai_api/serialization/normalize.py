"""Conversion of request values into wire-ready values."""

from __future__ import annotations

import datetime
import io
from collections.abc import Mapping
from enum import Enum
from typing import Any, Final

from ai_api.types.base import BaseAPIType
from ai_api.types.files import InputFile

MAX_DEPTH: Final = 256
"""Deepest nesting of containers accepted in a request body."""


class BodyNormalizer:
    """Prepares request body values for a transport-specific builder.

    Nested models are dumped, enums are unwrapped and datetimes become
    Unix timestamps. ``None`` fields are dropped from the body and from
    nested models, but kept as ``null`` inside lists and dicts. Files are
    passed through unchanged for the body builder to handle.
    """

    def normalize_mapping(
        self,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Normalize the fields of a body, dropping ``None`` results.

        Args:
            body: Body fields of a request.

        Returns:
            A new dict with normalized values.

        Raises:
            ValueError: If a value cannot be normalized.
        """
        return {
            key: normalized
            for key, value in body.items()
            if (normalized := self.normalize(value)) is not None
        }

    def normalize(self, value: Any) -> Any:
        """Normalize a single value, recursing into containers.

        Args:
            value: Any value of a request body.

        Returns:
            The normalized value; ``None`` for ``None``.

        Raises:
            ValueError: If a stream is closed or unreadable, a datetime is
                naive, or a container contains itself or is nested deeper
                than ``MAX_DEPTH``.
        """
        return self._normalize(value, set())

    def _normalize(self, value: Any, parents: set[int]) -> Any:
        """Normalize a value inside the containers being normalized.

        Args:
            value: The value to normalize.
            parents: Ids of the enclosing containers, to detect cycles.

        Returns:
            The normalized value.

        Raises:
            ValueError: If a stream is closed or unreadable, a datetime is
                naive, or a container contains itself or is nested deeper
                than ``MAX_DEPTH``.
        """
        if value is None:
            return None

        if isinstance(value, InputFile):
            return value

        if isinstance(value, io.IOBase):
            if value.closed or not value.readable():
                raise ValueError("Invalid file stream")

            return value

        if isinstance(value, BaseAPIType):
            return self._normalize(
                value.model_dump(
                    mode="python",
                    exclude_none=True,
                    warnings=False,
                ),
                parents,
            )

        if isinstance(value, Enum):
            return self._normalize(value.value, parents)

        if isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                raise ValueError("Datetime must be timezone-aware")

            return str(round(value.timestamp()))

        if isinstance(value, datetime.timedelta):
            now = datetime.datetime.now(datetime.UTC)
            return str(round((now + value).timestamp()))

        if isinstance(value, (Mapping, list, tuple)):
            if id(value) in parents:
                raise ValueError("Request body contains a circular reference")

            # Deeper data would hit Python's recursion limit.
            if len(parents) >= MAX_DEPTH:
                raise ValueError(
                    f"Request body is nested deeper than {MAX_DEPTH} levels"
                )

            parents.add(id(value))

            try:
                if isinstance(value, Mapping):
                    return {
                        key: self._normalize(item, parents)
                        for key, item in value.items()
                    }

                return [self._normalize(item, parents) for item in value]
            finally:
                parents.discard(id(value))

        return value
