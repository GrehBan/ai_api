from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BufferedResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    charset: str | None = None