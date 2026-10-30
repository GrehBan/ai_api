"""URLs and version of the Anthropic API."""

from __future__ import annotations

from typing import Final

from ai_api.http.api import BaseAPI


class AnthropicAPI(BaseAPI):
    """Builds Anthropic request URLs."""


ANTHROPIC_API: Final[AnthropicAPI] = AnthropicAPI(
    base_url="https://api.anthropic.com/",
    version="v1",
)
"""The Anthropic API, version 1."""

API_VERSION: Final = "2026-10-21"
"""The ``anthropic-version`` that the library's Anthropic types follow.

Anthropic requires the header on every request; it fixes the request and
response format, so it belongs to the methods written against it.
"""
