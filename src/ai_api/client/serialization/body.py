import datetime
import io
from collections.abc import Mapping
from enum import Enum
from typing import Any

from ai_api.types.base import BaseAPIType


class BodyNormalizer:
    def normalize_mapping(
        self,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            key: normalized
            for key, value in body.items()
            if (normalized := self.normalize(value)) is not None
        }

    def normalize(self, value: Any) -> Any:
        if value is None:
            return None

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