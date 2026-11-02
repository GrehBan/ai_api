"""Parsers that turn responses into typed results."""

from __future__ import annotations

from aiollm.response.parse.buffered import (
    BufferedResponseParser,
)
from aiollm.response.parse.protocols import (
    IResponseParser,
    IStreamParser,
)
from aiollm.response.parse.sse import (
    ServerSentEvent,
)
from aiollm.response.parse.stream import (
    StreamResponseParser,
)

__all__ = [
    "BufferedResponseParser",
    "IResponseParser",
    "IStreamParser",
    "ServerSentEvent",
    "StreamResponseParser",
]
