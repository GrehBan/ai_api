"""Client of the Anthropic API."""

from __future__ import annotations

from typing import Final, Unpack

from ai_api.client.base import BaseProviderClient
from ai_api.http.auth import BearerAuth
from ai_api.http.headers import AUTHORIZATION, X_API_KEY
from ai_api.providers.anthropic.adapters.generation import (
    DEFAULT_MAX_TOKENS,
    AnthropicGenerationAdapter,
)
from ai_api.providers.anthropic.api import ANTHROPIC_API, PROVIDER
from ai_api.providers.anthropic.resolver import AnthropicErrorResolver
from ai_api.providers.anthropic.types.events import MessageStreamEvent
from ai_api.providers.anthropic.types.messages import Message
from ai_api.resources.generation import GenerationResource
from ai_api.session.aiohttp import AiohttpSession, AiohttpSessionOptions
from ai_api.session.factory import SessionFactory
from ai_api.session.protocols import ISession

session_factory: Final = SessionFactory(
    AiohttpSession,
    auth_factory=BearerAuth,
    api=ANTHROPIC_API,
    error_resolver=AnthropicErrorResolver(),
)
"""Builds the ``AiohttpSession`` of an ``AnthropicClient`` from a token."""

ANTHROPIC_API_KEY_ENV: Final = "ANTHROPIC_API_KEY"
"""Environment variable with the API key, read without a token or session."""


class AnthropicClient(BaseProviderClient):
    """Client of the Anthropic API.

    Anthropic takes the key in ``Authorization`` or ``x-api-key``; a
    given session that sends it elsewhere, or uses the generic
    ``BaseErrorResolver``, gets a warning from ``warn_about_session``.

    Attributes:
        generation: Shared text generation over the Messages API, with
            ``max_tokens`` of ``default_max_tokens`` for a request
            without ``max_output_tokens``.

    Example:
        >>> async with AnthropicClient("sk-ant-...") as client:
        ...     response = await client.generation.create(
        ...         GenerationRequest(model="claude-opus-5-5", input="Hi")
        ...     )
    """

    provider = PROVIDER
    session_factory = session_factory
    api_key_env = ANTHROPIC_API_KEY_ENV
    auth_headers = frozenset({AUTHORIZATION, X_API_KEY})
    adapter_types = (AnthropicGenerationAdapter,)

    generation: GenerationResource[Message, MessageStreamEvent]

    def __init__(
        self,
        token: str | None = None,
        *,
        session: ISession | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        **session_kwargs: Unpack[AiohttpSessionOptions],
    ) -> None:
        """Create the client from a token or a ready session.

        Args:
            token: API key; an ``AiohttpSession`` with Anthropic's URL,
                error resolver and ``BearerAuth`` is created from it.
                Without a token and a session, ``ANTHROPIC_API_KEY`` is
                used.
            session: A ready session, for custom settings; it is checked
                with ``warn_about_session``.
            default_max_tokens: ``max_tokens`` that ``generation`` sends
                for a request without ``max_output_tokens``, since
                Anthropic requires it.
            **session_kwargs: Further arguments of the
                ``AiohttpSession`` built from the token, such as
                ``timeout`` or ``max_body_size``.

        Raises:
            ValueError: If both ``token`` and ``session`` are given, or
                neither is and ``ANTHROPIC_API_KEY`` is unset or empty, or
                ``session_kwargs`` are given with a ``session``, or
                ``default_max_tokens`` is not positive.
        """
        # Built first: a bad default_max_tokens fails before a session
        # (and its connection pool) exists.
        adapter = self.adapter(
            AnthropicGenerationAdapter, default_max_tokens=default_max_tokens
        )
        super().__init__(token, session=session, **session_kwargs)
        self.generation = GenerationResource(self.session, adapter)
