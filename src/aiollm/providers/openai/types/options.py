"""OpenAI's own options of a generation call."""

from __future__ import annotations

from typing import Literal, TypedDict

from aiollm.providers.openai.methods.params.responses import (
    CreateResponseParamsTemperature,
    CreateResponseParamsTopP,
    IncludeEnum,
    ReasoningModeEnum,
)
from aiollm.providers.openai.methods.responses import Metadata


class OpenAIReasoningOptions(TypedDict, total=False):
    """The parts of ``reasoning`` that the shared request does not set.

    ``effort`` is the shared ``GenerationRequest.effort``, so it is not
    here: the adapter merges these keys with it into one ``reasoning``.

    Attributes:
        summary: Ask for a summary of the reasoning, which comes back as
            ``GenerationResponse.reasoning`` and ``ReasoningDeltaEvent``.
        context: Which earlier reasoning the model sees.
        mode: The reasoning mode, such as ``standard`` or ``pro``.
    """

    summary: Literal["auto", "concise", "detailed"] | None
    context: Literal["auto", "current_turn", "all_turns"] | None
    mode: ReasoningModeEnum


class OpenAIGenerationOptions(TypedDict, total=False):
    """Fields of ``POST /responses`` that the shared request does not hold.

    Given to ``client.generation.create()`` or ``stream()`` as
    ``provider_options``; a key left out is not sent. Each key is a field
    of ``CreateResponse`` with its type, and the method checks the value.

    Attributes:
        metadata: Up to 16 key-value pairs stored with the response.
        reasoning: Reasoning settings besides the shared ``effort``,
            such as ``{"summary": "auto"}``.
        store: Whether OpenAI keeps the response; ``None`` sends ``null``.
        previous_response_id: The response this one continues.
        temperature: Sampling temperature, 0 to 2.
        top_p: Nucleus sampling mass, 0 to 1.
        include: Extra data to return, such as
            ``"reasoning.encrypted_content"``, which a stateless tool loop
            (``store: False``) needs to pass reasoning back.
        max_tool_calls: Most built-in tool calls in one response.

    Example:
        >>> await client.generation.create(
        ...     model="gpt-5", input="Hi", provider_options={"store": False}
        ... )
    """

    metadata: Metadata | None
    reasoning: OpenAIReasoningOptions
    store: bool | None
    previous_response_id: str | None
    temperature: CreateResponseParamsTemperature | None
    top_p: CreateResponseParamsTopP | None
    include: list[IncludeEnum] | None
    max_tool_calls: int | None
