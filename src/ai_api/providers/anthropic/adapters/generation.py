"""Adapter of shared generation to Anthropic's Messages API."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Final

from ai_api.adapters.base import BaseCompositeAdapter
from ai_api.enums import Provider
from ai_api.providers.anthropic.adapters.conversions import (
    StreamState,
    incomplete_reason,
    message_usage,
)
from ai_api.providers.anthropic.methods.messages import (
    CreateMessage,
    StreamMessage,
)
from ai_api.providers.anthropic.types.events import (
    ContentBlockDeltaEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    TextDelta,
)
from ai_api.providers.anthropic.types.messages import Message
from ai_api.types.requests.generation import GenerationRequest
from ai_api.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    TextDeltaEvent,
)
from ai_api.types.responses.generation import GenerationResponse

DEFAULT_MAX_TOKENS: Final = 1024
"""``max_tokens`` sent when the request has no ``max_output_tokens``."""

METADATA_KEYS: Final = frozenset({"user_id"})
"""The only metadata keys Anthropic accepts."""


class AnthropicGenerationAdapter(
    BaseCompositeAdapter[
        GenerationRequest,
        Message,
        GenerationResponse[Message],
        MessageStreamEvent,
        GenerationEvent[MessageStreamEvent],
    ],
):
    """Generates text with ``POST /messages``.

    ``input`` becomes one ``user`` turn and ``instructions`` the
    ``system`` prompt. Anthropic requires ``max_tokens``, so a request
    without ``max_output_tokens`` gets ``default_max_tokens``. Anthropic
    takes only ``user_id`` as metadata; any other key raises
    ``ValueError`` instead of being dropped.

    Attributes:
        default_max_tokens: ``max_tokens`` of a request without
            ``max_output_tokens``.

    Example:
        >>> generation = GenerationResource(
        ...     session, AnthropicGenerationAdapter(default_max_tokens=4096)
        ... )
    """

    def __init__(
        self, *, default_max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        """Store the default token limit.

        Args:
            default_max_tokens: ``max_tokens`` of a request without
                ``max_output_tokens``; positive.

        Raises:
            ValueError: If ``default_max_tokens`` is not positive.
        """
        if default_max_tokens < 1:
            message = (
                "default_max_tokens must be positive, "
                f"got {default_max_tokens}"
            )
            raise ValueError(message)
        self.default_max_tokens = default_max_tokens

    def to_method(self, request: GenerationRequest) -> CreateMessage:
        """Build the method that creates a message at once.

        Args:
            request: The shared request.

        Returns:
            ``CreateMessage`` for the request.

        Raises:
            ValueError: If the metadata has keys other than ``user_id``.
        """
        return CreateMessage(**self._fields(request))

    def to_response(
        self,
        raw: Message,
        request: GenerationRequest,
    ) -> GenerationResponse[Message]:
        """Convert a finished message.

        Args:
            raw: Anthropic's message.
            request: The request it answers.

        Returns:
            The generated text with its token counts; a message stopped
            early, by ``max_tokens`` for example, keeps its text and gets
            ``incomplete_reason``.

        Raises:
            APIResponseError: The message has no stop reason.
        """
        return GenerationResponse[Message](
            id=raw.id,
            provider=Provider.ANTHROPIC,
            model=raw.model,
            text=raw.text,
            usage=message_usage(raw.usage),
            incomplete_reason=incomplete_reason(raw.stop_reason),
            raw=raw,
        )

    def to_stream_method(self, request: GenerationRequest) -> StreamMessage:
        """Build the method that streams a message.

        Args:
            request: The shared request.

        Returns:
            ``StreamMessage`` for the request.

        Raises:
            ValueError: If the metadata has keys other than ``user_id``.
        """
        return StreamMessage(**self._fields(request))

    async def to_events(
        self,
        items: AsyncIterator[MessageStreamEvent],
        request: GenerationRequest,
    ) -> AsyncGenerator[GenerationEvent[MessageStreamEvent]]:
        """Convert the events of a message stream.

        Text deltas become ``TextDeltaEvent``; ``message_stop`` becomes
        ``CompletedEvent`` with the counts and stop reason collected from
        ``message_start`` and ``message_delta``; every other event is an
        ``OtherEvent``.

        Args:
            items: The events Anthropic sent.
            request: The request the stream answers.

        Yields:
            One shared event per Anthropic event.

        Raises:
            APIResponseError: ``message_stop`` came without a stop
                reason. The stream's end is checked by ``StreamMessage``
                and by the use case.
        """
        state = StreamState()
        async for item in items:
            match item:
                case ContentBlockDeltaEvent(delta=TextDelta() as delta):
                    yield TextDeltaEvent[MessageStreamEvent](
                        text=delta.text, raw=item
                    )
                    continue
                case MessageStartEvent():
                    state.start(item)
                case MessageDeltaEvent():
                    state.update(item)
                case MessageStopEvent():
                    yield CompletedEvent[MessageStreamEvent](
                        usage=state.usage(),
                        incomplete_reason=incomplete_reason(state.stop_reason),
                        raw=item,
                    )
                    continue
                case _:
                    pass
            yield OtherEvent[MessageStreamEvent](raw=item)

    def _fields(self, request: GenerationRequest) -> dict[str, Any]:
        """Take the method fields of a request.

        Args:
            request: The shared request.

        Returns:
            The fields that both methods take.

        Raises:
            ValueError: If the metadata has keys other than ``user_id``.
        """
        fields: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.input}],
            "max_tokens": request.max_output_tokens or self.default_max_tokens,
            "system": request.instructions,
        }
        if request.metadata is not None:
            unknown = sorted(set(request.metadata) - METADATA_KEYS)
            if unknown:
                message = (
                    "Anthropic accepts only 'user_id' as metadata; "
                    f"unsupported keys: {', '.join(unknown)}"
                )
                raise ValueError(message)
            fields["metadata"] = request.metadata
        return fields
