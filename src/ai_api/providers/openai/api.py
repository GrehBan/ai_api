"""URLs of the OpenAI API."""

from __future__ import annotations

from typing import Final

from ai_api.enums import Provider
from ai_api.http.api import BaseAPI

PROVIDER: Final = Provider.OPENAI
"""The provider of this package; its client, resolver and adapter use it."""


class OpenAIAPI(BaseAPI):
    """Builds OpenAI request URLs."""


OPENAI_API: Final[OpenAIAPI] = OpenAIAPI(
    base_url="https://api.openai.com/",
    version="v1",
)
"""The OpenAI API, version 1."""
