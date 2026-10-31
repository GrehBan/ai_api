"""Client of the OpenAI API."""

from __future__ import annotations

import warnings
from typing import Final, Unpack

from ai_api.client.base import BaseClient
from ai_api.client.factory import SessionFactory
from ai_api.client.validators import resolve_token
from ai_api.http.auth import BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.headers import AUTHORIZATION
from ai_api.providers.openai.adapters.generation import (
    OpenAIGenerationAdapter,
)
from ai_api.providers.openai.api import OPENAI_API
from ai_api.providers.openai.resolver import (
    OpenAIErrorResolver,
)
from ai_api.providers.openai.types.events import ResponseStreamEvent
from ai_api.providers.openai.types.responses import Response
from ai_api.resources.generation import GenerationResource
from ai_api.session.aiohttp import AiohttpSession, AiohttpSessionOptions
from ai_api.session.base import BaseSession
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


def warn_about_session(session: ISession) -> None:
    """Warn about a session that does not fit OpenAI.

    Only ``BaseSession`` subclasses are checked; other sessions are
    trusted as they are.

    Args:
        session: The session given to ``OpenAIClient``.

    Warns:
        UserWarning: If the session uses the generic ``BaseErrorResolver``
            or does not authenticate with ``Authorization``.
    """
    if not isinstance(session, BaseSession):
        return

    # stacklevel 3: the warning points at the OpenAIClient(...) call.
    if type(session.error_resolver) is BaseErrorResolver:
        warnings.warn(
            "OpenAIClient is using the generic BaseErrorResolver. "
            "Consider using OpenAIErrorResolver for provider-specific "
            "error handling.",
            UserWarning,
            stacklevel=3,
        )

    if session.auth.header_name.lower() != AUTHORIZATION.lower():
        warnings.warn(
            "OpenAI requires Authorization: Bearer <token>.",
            UserWarning,
            stacklevel=3,
        )


class OpenAIClient(BaseClient[ISession]):
    """Client of the OpenAI API.

    Attributes:
        generation: Shared text generation over the Responses API.

    Example:
        >>> async with OpenAIClient("sk-...") as client:
        ...     response = await client.generation.create(
        ...         GenerationRequest(model="gpt-5", input="Hi")
        ...     )
    """

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
        token = resolve_token(token, session, OPENAI_API_KEY_ENV)

        if session is None:
            assert token is not None  # noqa: S101 - checked above
            session = session_factory(token, **session_kwargs)
        else:
            if session_kwargs:
                names = ", ".join(sorted(session_kwargs))
                message = (
                    "session_kwargs configure the session built from a "
                    f"token; pass them to the given session instead: {names}"
                )
                raise ValueError(message)
            warn_about_session(session)

        super().__init__(session)
        self.generation: GenerationResource[Response, ResponseStreamEvent] = (
            GenerationResource(session, OpenAIGenerationAdapter())
        )
