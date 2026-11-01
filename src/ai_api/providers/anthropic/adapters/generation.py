"""Adapter of shared generation to Anthropic's Messages API."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Final

from ai_api.adapters.base import BaseCompositeAdapter
from ai_api.adapters.validation import check_last_turn, request_errors
from ai_api.exceptions import RequestValidationError
from ai_api.providers.anthropic.adapters.conversions import (
    INPUT_CONVERSION,
    STOP_CONVERSION,
    USAGE_CONVERSION,
    StreamState,
)
from ai_api.providers.anthropic.api import PROVIDER
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

if TYPE_CHECKING:
    from ai_api.adapters.protocols import IConversion
    from ai_api.adapters.stop import Stop
    from ai_api.providers.anthropic.methods.base import MessageParam
    from ai_api.providers.anthropic.types.events import MessageDelta
    from ai_api.providers.anthropic.types.messages import Usage
    from ai_api.types.requests.generation import Turn
    from ai_api.types.responses.generation import TokenUsage

DEFAULT_MAX_TOKENS: Final = 1024
"""``max_tokens`` sent when the request has no ``max_output_tokens``."""

METADATA_KEYS: Final = frozenset({"user_id"})
"""The only metadata keys Anthropic accepts."""

SHARED_FIELDS: Final[Mapping[str, str]] = {
    "messages": "input",
    "max_tokens": "max_output_tokens",
    "system": "instructions",
}
"""Shared request field of each method field with another name."""


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

    A text ``input`` becomes one ``user`` turn, a list of turns the
    ``messages`` as they are, and ``instructions`` the ``system``
    prompt. Anthropic requires ``max_tokens``, so a request without
    ``max_output_tokens`` gets ``default_max_tokens``. A request that
    Anthropic would reject raises ``RequestValidationError`` before
    anything is sent: Anthropic takes only ``user_id`` as metadata, so
    any other key raises instead of being dropped, and a conversation
    must end with a ``user`` turn.

    The conversions of the input, the token counts and the stop are
    keyword arguments, so any of them can be swapped without a subclass;
    a stream goes through the same ones as a message.

    Attributes:
        default_max_tokens: ``max_tokens`` of a request without
            ``max_output_tokens``.
        input_conversion: Turns the shared ``input`` into ``messages``.
        usage_conversion: Turns Anthropic's ``Usage`` into shared counts.
        stop_conversion: Tells how a message, or a stream, stopped.

    Example:
        >>> generation = GenerationResource(
        ...     session, AnthropicGenerationAdapter(default_max_tokens=4096)
        ... )
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        input_conversion: IConversion[
            str | list[Turn], list[MessageParam]
        ] = INPUT_CONVERSION,
        usage_conversion: IConversion[Usage, TokenUsage] = USAGE_CONVERSION,
        stop_conversion: IConversion[
            Message | MessageDelta, Stop
        ] = STOP_CONVERSION,
    ) -> None:
        """Store the default token limit and the conversions.

        Args:
            default_max_tokens: ``max_tokens`` of a request without
                ``max_output_tokens``; positive.
            input_conversion: Turns the shared ``input`` into
                ``messages``.
            usage_conversion: Turns Anthropic's ``Usage`` into shared
                counts.
            stop_conversion: Tells how a message, or a stream, stopped.

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
        self.input_conversion = input_conversion
        self.usage_conversion = usage_conversion
        self.stop_conversion = stop_conversion

    def to_method(self, request: GenerationRequest) -> CreateMessage:
        """Build the method that creates a message at once.

        Args:
            request: The shared request.

        Returns:
            ``CreateMessage`` for the request.

        Raises:
            RequestValidationError: Anthropic cannot take the request.
        """
        fields = self._fields(request)
        with request_errors(self.provider, SHARED_FIELDS):
            return CreateMessage(**fields)

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
            ``incomplete_reason``, and a refusal gets ``refusal``.

        Raises:
            APIResponseError: The message has no stop reason.
        """
        stop = self.stop_conversion(raw)
        return GenerationResponse[Message](
            id=raw.id,
            provider=self.provider,
            model=raw.model,
            text=raw.text,
            usage=self.usage_conversion(raw.usage),
            incomplete_reason=stop.incomplete_reason,
            raw_incomplete_reason=stop.raw_incomplete_reason,
            refusal=stop.refusal,
            raw=raw,
        )

    def to_stream_method(self, request: GenerationRequest) -> StreamMessage:
        """Build the method that streams a message.

        Args:
            request: The shared request.

        Returns:
            ``StreamMessage`` for the request.

        Raises:
            RequestValidationError: Anthropic cannot take the request.
        """
        fields = self._fields(request)
        with request_errors(self.provider, SHARED_FIELDS):
            return StreamMessage(**fields)

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
                reason or without ``message_start``. The stream's end is
                checked by ``StreamMessage`` and by the use case.
        """
        state = StreamState()
        async for item in items:
            match item:
                case ContentBlockDeltaEvent(delta=TextDelta() as delta):
                    state.add_text(delta.text)
                    yield TextDeltaEvent[MessageStreamEvent](
                        text=delta.text, raw=item
                    )
                    continue
                case MessageStartEvent():
                    state.start(item)
                case MessageDeltaEvent():
                    state.update(item)
                case MessageStopEvent():
                    stop = self.stop_conversion(state.delta)
                    message = state.message()
                    yield CompletedEvent[MessageStreamEvent](
                        id=message.id,
                        provider=self.provider,
                        model=message.model,
                        text=state.text(),
                        usage=self.usage_conversion(state.usage()),
                        incomplete_reason=stop.incomplete_reason,
                        raw_incomplete_reason=stop.raw_incomplete_reason,
                        refusal=stop.refusal,
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
            RequestValidationError: The conversation does not end with a
                ``user`` turn, or the metadata has keys other than
                ``user_id``.
        """
        check_last_turn(request, self.provider)
        fields: dict[str, Any] = {
            "model": request.model,
            "messages": self.input_conversion(request.input),
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
                raise RequestValidationError(
                    message, provider=self.provider, field="metadata"
                )
            fields["metadata"] = request.metadata
        return fields
