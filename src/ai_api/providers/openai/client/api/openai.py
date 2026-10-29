"""URLs of the OpenAI API."""

from typing import Final

from ai_api.client.api.base import BaseAPI


class OpenAIAPI(BaseAPI):
    """Builds OpenAI request URLs."""


OPENAI_API: Final[OpenAIAPI] = OpenAIAPI(
    base_url="https://api.openai.com/",
    version="v1",
)
"""The OpenAI API, version 1."""
