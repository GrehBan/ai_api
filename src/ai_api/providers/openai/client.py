"""Client of the OpenAI API."""

from __future__ import annotations

from typing import Final, Unpack

from ai_api.client.base import BaseProviderClient
from ai_api.client.factory import SessionFactory
from ai_api.http.auth import BearerAuth
from ai_api.http.headers import AUTHORIZATION
from ai_api.providers.openai.adapters.generation import (
    OpenAIGenerationAdapter,
)
from ai_api.providers.openai.api import OPENAI_API, PROVIDER
from ai_api.providers.openai.resolver import (
    OpenAIErrorResolver,
)
from ai_api.providers.openai.types.events import ResponseStreamEvent
from ai_api.providers.openai.types.responses import Response
from ai_api.resources.generation import GenerationResource
from ai_api.session.aiohttp import AiohttpSession, AiohttpSessionOptions
from ai_api.session.protocols import ISession

session_factory: Final = SessionFactory(
    AiohttpSession,
    auth_factory=BearerAuth,
    api=OPENAI_API,
    error_resolver=OpenAIErrorResolver(),
)
"""Builds the ``AiohttpSession`` of an ``OpenAIClient`` from a token."""

OPENAI_API_KEY_ENV: Final = "OPENAI_API_KEY"
"""Environment variable with the API key, read without a token or session."""


class OpenAIClient(BaseProviderClient[OpenAIGenerationAdapter]):
    """Client of the OpenAI API.

    A given session that does not fit OpenAI (the generic
    ``BaseErrorResolver``, a key outside ``Authorization``) gets a
    warning from ``warn_about_session``.

    Attributes:
        generation: Shared text generation over the Responses API.

    Example:
        >>> async with OpenAIClient("sk-...") as client:
        ...     response = await client.generation.create(
        ...         GenerationRequest(model="gpt-5", input="Hi")
        ...     )
    """

    provider = PROVIDER
    session_factory = session_factory
    api_key_env = OPENAI_API_KEY_ENV
    auth_headers = frozenset({AUTHORIZATION})
    adapter_type = OpenAIGenerationAdapter

    generation: GenerationResource[Response, ResponseStreamEvent]

    def __init__(
        self,
        token: str | None = None,
        *,
        session: ISession | None = None,
        **session_kwargs: Unpack[AiohttpSessionOptions],
    ) -> None:
        """Create the client from a token or a ready session.

        Args:
            token: API key; an ``AiohttpSession`` with OpenAI's URL, error
                resolver and ``BearerAuth`` is created from it. Without a
                token and a session, ``OPENAI_API_KEY`` is used.
            session: A ready session, for custom settings; it is checked
                with ``warn_about_session``.
            **session_kwargs: Further arguments of the
                ``AiohttpSession`` built from the token, such as
                ``timeout`` or ``max_body_size``.

        Raises:
            ValueError: If both ``token`` and ``session`` are given, or
                neither is and ``OPENAI_API_KEY`` is unset or empty, or
                ``session_kwargs`` are given with a ``session``.
        """
        super().__init__(token, session=session, **session_kwargs)
        self.generation = GenerationResource(self.session, self.adapter_type())
