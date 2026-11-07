"""Shared response of text generation."""

from __future__ import annotations

from pydantic import MISSING, Field

from aiollm.enums import IncompleteReason, Provider
from aiollm.types.base import BaseAPIType
from aiollm.types.requests.generation import (
    NativeOutput,
    ToolCall,
    ToolCallTurn,
    Turn,
)
from aiollm.types.responses.base import BaseResponse


class TokenUsage(BaseAPIType):
    """Token counts reported by the provider.

    The cache and reasoning counts are parts of the input and output
    counts, not added to them. ``None`` means the provider did not
    report the count.

    Attributes:
        input_tokens: Tokens in the input, the cached ones included.
        output_tokens: Tokens generated, reasoning included.
        total_tokens: Both together.
        cache_read_tokens: Input tokens read from the prompt cache.
        cache_write_tokens: Input tokens written to the prompt cache.
        reasoning_tokens: Output tokens spent on reasoning (thinking).
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None


class GenerationResponse[RawT](BaseResponse):
    """Generated text with the provider's own response.

    ``RawT`` is the provider's response type. It has no bound: an
    unparametrized ``GenerationResponse`` keeps ``raw`` as it is given,
    while a bound such as ``BaseAPIType`` would turn a dict into an
    empty model, because unknown fields are ignored.

    A refusal is an answer, as in the providers' SDKs, not an error: it
    sets ``incomplete_reason`` to ``IncompleteReason.REFUSAL``.

    Attributes:
        id: Response identifier from the provider, if any.
        provider: Provider that generated the response.
        model: Model that generated it.
        text: Generated text, if any.
        usage: Token counts, if reported.
        incomplete_reason: Why the provider stopped before the end, in
            terms every provider has; ``None`` if the generation
            finished. ``text`` then holds what was generated until the
            stop.
        raw_incomplete_reason: The provider's own reason, such as
            OpenAI's ``"content_filter"``, if it gave one.
        refusal: Why the model refused, if it did and the provider says
            why.
        reasoning: A summary of the model's reasoning, when the
            provider sent one (OpenAI's ``reasoning.summary``,
            Anthropic's ``thinking.display`` of ``summarized``, both
            set through ``provider_options``); ``None`` otherwise.
        tool_calls: The calls of the request's tools, in the order the
            model made them; empty without calls. A response with calls
            waits for their results; when ``incomplete_reason`` is set,
            the last call may be cut off, so run none of them.
        native: The response's output as its provider's next request
            takes it; ``to_turn()`` puts it in a turn of calls.
        raw: The provider's response.
    """

    id: str | None = None
    provider: Provider
    model: str

    text: str | None = None
    usage: TokenUsage | None = None
    incomplete_reason: IncompleteReason | None = None
    raw_incomplete_reason: str | None = None
    refusal: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    native: NativeOutput | None = None

    raw: RawT

    def to_turn(self) -> Turn | ToolCallTurn:
        """Turn the response into the next item of the conversation.

        Returns:
            A ``ToolCallTurn`` with the calls, the text and ``native``
            when the model called tools, which the tools' results follow;
            else an ``assistant`` turn with ``text``.

        Raises:
            ValueError: The response has neither calls nor text, such as
                a refusal without text; or it has calls but stopped early
                (``incomplete_reason``), so the last one may be cut off.

        Example:
            >>> request = GenerationRequest(
            ...     model=model,
            ...     input=[*turns, response.to_turn(), Turn.user("More")],
            ... )
        """
        return continuation(
            "the response" if self.id is None else f"response {self.id}",
            text=self.text,
            tool_calls=self.tool_calls,
            incomplete_reason=self.incomplete_reason,
            native=self.native,
        )


def continuation(
    name: str,
    *,
    text: str | None,
    tool_calls: list[ToolCall],
    incomplete_reason: IncompleteReason | None,
    native: NativeOutput | None,
) -> Turn | ToolCallTurn:
    """Build the turn that continues a conversation after a generation.

    The one rule of ``GenerationResponse.to_turn()`` and
    ``CompletedEvent.to_turn()``.

    Args:
        name: How the errors name the generation.
        text: Its text, if any.
        tool_calls: Its calls.
        incomplete_reason: Why it stopped early, if it did.
        native: Its output as its provider's next request takes it.

    Returns:
        A ``ToolCallTurn`` when there are calls, else an ``assistant``
        turn.

    Raises:
        ValueError: There are neither calls nor text, or there are calls
            but the generation stopped early.
    """
    if tool_calls:
        if incomplete_reason is not None:
            message = (
                f"{name} stopped early ({incomplete_reason.value}), so its "
                "last tool call may be cut off; retry it with a higher "
                "limit instead"
            )
            raise ValueError(message)
        return ToolCallTurn(
            text=MISSING if text is None else text,
            tool_calls=tool_calls,
            native=MISSING if native is None else native,
        )
    if text is None:
        message = f"{name} has no text to continue the conversation with"
        raise ValueError(message)
    return Turn.assistant(text)
