from collections.abc import Mapping
from typing import Any, Protocol

from ai_api.methods.enums import BodyFormat


class IBodyBuilder(Protocol):
    def build(
        self,
        body: Mapping[str, Any],
        body_format: BodyFormat,
    ) -> Mapping[str, Any]: ...