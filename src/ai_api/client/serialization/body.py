"""Conversion of request values into wire-ready values."""

import datetime
import io
from collections.abc import Mapping
from enum import Enum
from typing import Any

from ai_api.types.base import BaseAPIType
from ai_api.types.files import InputFile


class BodyNormalizer:
    """Prepares request body values for a transport-specific builder.

    Nested models are dumped, enums are unwrapped, datetimes become Unix
    timestamps and ``None`` values are dropped. Files are passed through
    unchanged for the body builder to handle.
    """

    def normalize_mapping(
        self,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Normalize every value of a mapping, dropping ``None`` results.

        Args:
            body: Body fields of a request.

        Returns:
            A new dict with normalized values.
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
            The normalized value, or ``None`` if it should be omitted.

        Raises:
            ValueError: If a stream is closed or unreadable, or a datetime
                is naive.
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
            return self.normalize(
                value.model_dump(
                    mode="python",
                    warnings=False,
                )
            )

        if isinstance(value, Enum):
            return self.normalize(value.value)

        if isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                raise ValueError("Datetime must be timezone-aware")

            return str(round(value.timestamp()))

        if isinstance(value, datetime.timedelta):
            now = datetime.datetime.now(datetime.UTC)
            return str(round((now + value).timestamp()))

        if isinstance(value, Mapping):
            return self.normalize_mapping(value)

        if isinstance(value, (list, tuple)):
            return [
                normalized
                for item in value
                if (normalized := self.normalize(item)) is not None
            ]

        return value
