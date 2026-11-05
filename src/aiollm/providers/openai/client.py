"""Client of the OpenAI API."""

from __future__ import annotations

from typing import Final, Unpack

from aiollm.client.base import BaseProviderClient
from aiollm.client.binding import bind_resource
from aiollm.http.auth import BearerAuth
from aiollm.http.headers import AUTHORIZATION
from aiollm.providers.openai.adapters.generation import (
    OpenAIGenerationAdapter,
)
from aiollm.providers.openai.api import OPENAI_API, PROVIDER
from aiollm.providers.openai.resolver import (
    OpenAIErrorResolver,
)
from aiollm.providers.openai.types.events import ResponseStreamEvent
from aiollm.providers.openai.types.options import OpenAIGenerationOptions
from aiollm.providers.openai.types.responses import Response
from aiollm.resources.generation import GenerationResource
from aiollm.session.aiohttp import AiohttpSession, AiohttpSessionOptions
from aiollm.session.factory import SessionFactory
from aiollm.session.protocols import ISession

session_factory: Final = SessionFactory(
    AiohttpSession,
    auth_factory=BearerAuth,
    api=OPENAI_API,
    error_resolver=OpenAIErrorResolver(),
)
"""Builds the ``AiohttpSession`` of an ``OpenAIClient`` from a token."""

OPENAI_API_KEY_ENV: Final = "OPENAI_API_KEY"
"""Environment variable with the API key, read without a token or session."""


class OpenAIClient(BaseProviderClient):
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

    generation = bind_resource(
        GenerationResource[
            Response, ResponseStreamEvent, OpenAIGenerationOptions
        ],
        OpenAIGenerationAdapter,
    )

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
