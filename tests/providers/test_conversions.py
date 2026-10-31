"""Conversions of the generation adapters can be swapped.

Each adapter takes its input, usage and stop conversions as keyword
arguments; a swapped one is used for the request, the response and the
stream alike, and gets the provider's own types.
"""

from __future__ import annotations

from typing import Any

from ai_api.adapters import BaseConversion, IConversion, Stop
from ai_api.enums import IncompleteReason
from ai_api.providers.anthropic.adapters import AnthropicGenerationAdapter
from ai_api.providers.anthropic.adapters.conversions import (
    INPUT_CONVERSION as ANTHROPIC_INPUT,
)
from ai_api.providers.anthropic.adapters.conversions import (
    STOP_CONVERSION as ANTHROPIC_STOP,
)
from ai_api.providers.anthropic.adapters.conversions import (
    USAGE_CONVERSION as ANTHROPIC_USAGE,
)
from ai_api.providers.anthropic.methods.base import MessageParam
from ai_api.providers.anthropic.types import Message
from ai_api.providers.anthropic.types.events import MessageDelta
from ai_api.providers.anthropic.types.messages import Usage
from ai_api.providers.openai.adapters import OpenAIGenerationAdapter
from ai_api.providers.openai.adapters.conversions import (
    INPUT_CONVERSION as OPENAI_INPUT,
)
from ai_api.providers.openai.adapters.conversions import (
    STOP_CONVERSION as OPENAI_STOP,
)
from ai_api.providers.openai.adapters.conversions import (
    USAGE_CONVERSION as OPENAI_USAGE,
)
from ai_api.providers.openai.types import Response
from ai_api.providers.openai.types.responses import ResponseUsage
from ai_api.types.requests import Turn
from ai_api.types.responses import CompletedEvent, TokenUsage
from tests.providers import test_anthropic_generation as anthropic
from tests.providers import test_openai_generation as openai

USAGE = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3)

STOP = Stop(IncompleteReason.OTHER, "swapped", "because")


class Recording:
    """Swapped conversions that record what they were given."""

    def __init__(self) -> None:
        """Start with nothing seen."""
        self.seen: list[object] = []

    def usage(self, source: object, /) -> TokenUsage:
        """Record the source and give fixed counts."""
        self.seen.append(source)
        return USAGE

    def stop(self, source: object, /) -> Stop:
        """Record the source and give a fixed stop."""
        self.seen.append(source)
        return STOP


class OpenAIInput(
    BaseConversion[str | list[Turn], str | list[dict[str, Any]]],
):
    """Sends a fixed text instead of the request's input."""

    def __call__(
        self, source: str | list[Turn], /
    ) -> str | list[dict[str, Any]]:
        """Ignore the input."""
        return "swapped"


class AnthropicInput(BaseConversion[str | list[Turn], list[MessageParam]]):
    """Sends a fixed turn instead of the request's input."""

    def __call__(self, source: str | list[Turn], /) -> list[MessageParam]:
        """Ignore the input."""
        return [{"role": "user", "content": "swapped"}]


def test_default_conversions_are_conversions() -> None:
    defaults: list[IConversion[Any, Any]] = [
        OPENAI_INPUT,
        OPENAI_USAGE,
        OPENAI_STOP,
        ANTHROPIC_INPUT,
        ANTHROPIC_USAGE,
        ANTHROPIC_STOP,
    ]

    assert all(isinstance(each, BaseConversion) for each in defaults)
    assert OpenAIGenerationAdapter().usage_conversion is OPENAI_USAGE
    assert AnthropicGenerationAdapter().stop_conversion is ANTHROPIC_STOP


def check_completed(events: list[Any]) -> None:
    """Check that the completion went through the swapped conversions."""
    completed = events[-1]
    assert isinstance(completed, CompletedEvent)
    assert completed.usage == USAGE
    assert completed.incomplete_reason is STOP.incomplete_reason
    assert completed.raw_incomplete_reason == STOP.raw_incomplete_reason
    assert completed.refusal == STOP.refusal


async def test_openai_uses_swapped_conversions() -> None:
    recording = Recording()
    adapter = OpenAIGenerationAdapter(
        input_conversion=OpenAIInput(),
        usage_conversion=recording.usage,
        stop_conversion=recording.stop,
    )

    method = adapter.to_method(openai.REQUEST)
    response = adapter.to_response(openai.response(), openai.REQUEST)
    events = [
        event
        async for event in adapter.to_events(
            openai.events(openai.STREAM), openai.REQUEST
        )
    ]

    assert method.input == "swapped"
    assert response.usage == USAGE
    assert response.incomplete_reason is STOP.incomplete_reason
    assert response.raw_incomplete_reason == STOP.raw_incomplete_reason
    assert response.refusal == STOP.refusal
    check_completed(events)
    # Buffered, then streamed: the stop, then the usage of each.
    assert [type(source) for source in recording.seen] == [
        Response,
        ResponseUsage,
        Response,
        ResponseUsage,
    ]


async def test_anthropic_uses_swapped_conversions() -> None:
    recording = Recording()
    adapter = AnthropicGenerationAdapter(
        input_conversion=AnthropicInput(),
        usage_conversion=recording.usage,
        stop_conversion=recording.stop,
    )

    method = adapter.to_method(anthropic.REQUEST)
    response = adapter.to_response(anthropic.message(), anthropic.REQUEST)
    events = [
        event
        async for event in adapter.to_events(
            anthropic.events(anthropic.STREAM), anthropic.REQUEST
        )
    ]

    assert method.get_body_map()["messages"] == [
        {"role": "user", "content": "swapped"}
    ]
    assert response.usage == USAGE
    assert response.refusal == STOP.refusal
    check_completed(events)
    # A stream gives the same types as a message: a real Usage, and the
    # MessageDelta collected from the message_delta events.
    assert [type(source) for source in recording.seen] == [
        Message,
        Usage,
        MessageDelta,
        Usage,
    ]
