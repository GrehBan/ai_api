"""OpenAI's own options of a generation call."""

from __future__ import annotations

from typing import TypedDict

from aiollm.providers.openai.methods.base import Metadata


class OpenAIGenerationOptions(TypedDict, total=False):
    """Fields of ``POST /responses`` that the shared request does not hold.

    Given to ``client.generation.create()`` or ``stream()`` as
    ``provider_options``; a key left out is not sent. Each key is a field
    of ``CreateResponse`` with its type, and the method checks the value.

    Attributes:
        metadata: Up to 16 key-value pairs stored with the response.
        store: Whether OpenAI keeps the response; ``None`` sends ``null``.
        previous_response_id: The response this one continues.
        temperature: Sampling temperature, 0 to 2.
        top_p: Nucleus sampling mass, 0 to 1.

    Example:
        >>> await client.generation.create(
        ...     model="gpt-5", input="Hi", provider_options={"store": False}
        ... )
    """

    metadata: Metadata | None
    store: bool | None
    previous_response_id: str | None
    temperature: float | None
    top_p: float | None
