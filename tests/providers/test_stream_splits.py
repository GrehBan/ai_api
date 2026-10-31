"""T2: documented provider streams survive any split into chunks.

Each documented stream of the contract snapshot is sent through
``generation.stream`` as two chunks, cut at every byte position: inside
a line, between ``\\r`` and ``\\n``, and inside a multi-byte UTF-8
character (the text deltas get non-ASCII text for that). Every split
must give the same shared events as the whole body.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ai_api.providers.anthropic.adapters import AnthropicGenerationAdapter
from ai_api.providers.openai.adapters import OpenAIGenerationAdapter
from ai_api.resources.generation import GenerationResource
from ai_api.types.requests.generation import GenerationRequest
from ai_api.types.responses.events import TextDeltaEvent
from tests.conftest import FakeStreamResponse, FakeTransport
from tests.providers.test_wire_contracts import load

REQUEST = GenerationRequest(model="model-1", input="Hi")

SSE = {"Content-Type": "text/event-stream"}

MULTIBYTE = " é€😀"
"""Appended to every text delta: two-, three- and four-byte characters."""

STREAMS = [
    pytest.param(
        "anthropic/messages",
        "a05fa2",
        AnthropicGenerationAdapter(),
        id="anthropic",
    ),
    pytest.param(
        "openai/responses", "244618", OpenAIGenerationAdapter(), id="openai"
    ),
]


def documented_stream(name: str, example: str) -> list[dict[str, Any]]:
    """Take one whole documented stream from the snapshot.

    Args:
        name: Snapshot folder.
        example: Prefix of the example's id.

    Returns:
        Its events, the text deltas extended with ``MULTIBYTE``.
    """
    events = [
        item["body"]
        for item in load(name, "events.json")
        if item["id"].startswith(example)
    ]
    for event in events:
        if event["type"] == "response.output_text.delta":
            event["delta"] += MULTIBYTE
        delta = event.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            delta["text"] += MULTIBYTE
    return events


def encode(events: list[dict[str, Any]], newline: str) -> bytes:
    """Write events as an SSE body.

    Args:
        events: The events.
        newline: Line ending of every line.

    Returns:
        The body, UTF-8 encoded.
    """
    return "".join(
        f"event: {event['type']}{newline}"
        f"data: {json.dumps(event, ensure_ascii=False)}{newline}{newline}"
        for event in events
    ).encode()


async def stream(adapter: Any, chunks: list[bytes]) -> list[Any]:
    """Run chunks through ``generation.stream``.

    Args:
        adapter: The provider's generation adapter.
        chunks: The body.

    Returns:
        The shared events, dumped for comparison.
    """
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(chunks, headers=SSE)]
    )
    generation = GenerationResource(transport, adapter)

    async with generation.stream(REQUEST) as events:
        return [event async for event in events]


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize(("name", "example", "adapter"), STREAMS)
async def test_every_split_gives_the_same_events(
    name: str,
    example: str,
    adapter: Any,
    newline: str,
) -> None:
    body = encode(documented_stream(name, example), newline)
    whole = await stream(adapter, [body])

    texts = [e.text for e in whole if isinstance(e, TextDeltaEvent)]
    assert texts
    assert all(text.endswith(MULTIBYTE) for text in texts)
    assert whole[-1].type == "completed"

    for position in range(1, len(body)):
        split = await stream(adapter, [body[:position], body[position:]])

        assert split == whole, position
