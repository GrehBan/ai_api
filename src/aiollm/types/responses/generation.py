"""Shared response of text generation."""

from __future__ import annotations

import json

from pydantic import ConfigDict, Field

from aiollm.enums import IncompleteReason, Provider
from aiollm.types.base import BaseAPIType
from aiollm.types.requests.generation import JSONValue, Turn
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


class ToolCall(BaseAPIType):
    """A call of one of the request's tools.

    Immutable. The arguments are the JSON text the model wrote, as
    OpenAI sends them and as both stream them; Anthropic's object is
    written as JSON. Without ``strict`` the text may be invalid JSON, so
    it is parsed only on ``parse_arguments()``.

    Attributes:
        id: The call's id, which its result refers to (OpenAI's
            ``call_id``, Anthropic's ``tool_use.id``).
        name: The tool called.
        arguments: The arguments, a JSON object as text.

    Example:
        >>> for call in response.tool_calls:
        ...     result = run(call.name, call.parse_arguments())
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: str

    def parse_arguments(self) -> dict[str, JSONValue]:
        """Parse the arguments.

        Returns:
            The arguments as a JSON object.

        Raises:
            ValueError: The arguments are not a JSON object.
        """
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError as error:
            message = f"arguments of tool call {self.id} are not JSON"
            raise ValueError(message) from error
        if not isinstance(value, dict):
            message = f"arguments of tool call {self.id} are not an object"
            raise ValueError(message)
        return value


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

    raw: RawT

    def to_turn(self) -> Turn:
        """Turn the generated text into a turn of the conversation.

        Returns:
            An ``assistant`` turn with ``text``.

        Raises:
            ValueError: The response has no text, such as a refusal
                without one.

        Example:
            >>> request = GenerationRequest(
            ...     model=model,
            ...     input=[*turns, response.to_turn(), Turn.user("More")],
            ... )
        """
        if self.text is None:
            name = "the response" if self.id is None else f"response {self.id}"
            message = f"{name} has no text to continue the conversation with"
            raise ValueError(message)
        return Turn.assistant(self.text)
