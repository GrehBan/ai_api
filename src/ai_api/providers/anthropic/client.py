"""Client of the Anthropic API."""

from __future__ import annotations

import warnings
from typing import Final

from ai_api.client.base import BaseClient
from ai_api.client.factory import SessionFactory
from ai_api.client.validators import resolve_token
from ai_api.http.auth import BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.headers import AUTHORIZATION, X_API_KEY
from ai_api.providers.anthropic.adapters.generation import (
    DEFAULT_MAX_TOKENS,
    AnthropicGenerationAdapter,
)
from ai_api.providers.anthropic.api import ANTHROPIC_API
from ai_api.providers.anthropic.resolver import AnthropicErrorResolver
from ai_api.providers.anthropic.types.events import MessageStreamEvent
from ai_api.providers.anthropic.types.messages import Message
from ai_api.resources.generation import GenerationResource
from ai_api.session.aiohttp import AiohttpSession
from ai_api.session.base import BaseSession
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

_AUTH_HEADERS: Final = frozenset({AUTHORIZATION.lower(), X_API_KEY.lower()})
"""Headers Anthropic accepts the key in: ``Authorization`` or ``x-api-key``."""


def warn_about_session(session: ISession) -> None:
    """Warn about a session that does not fit Anthropic.

    Only ``BaseSession`` subclasses are checked; other sessions are
    trusted as they are.

    Args:
        session: The session given to ``AnthropicClient``.

    Warns:
        UserWarning: If the session uses the generic ``BaseErrorResolver``
            or authenticates with neither ``Authorization`` nor
            ``x-api-key``.
    """
    if not isinstance(session, BaseSession):
        return

    # stacklevel 3: the warning points at the AnthropicClient(...) call.
    if type(session.error_resolver) is BaseErrorResolver:
        warnings.warn(
            "AnthropicClient is using the generic BaseErrorResolver. "
            "Consider using AnthropicErrorResolver for provider-specific "
            "error handling.",
            UserWarning,
            stacklevel=3,
        )

    if session.auth.header_name.lower() not in _AUTH_HEADERS:
        warnings.warn(
            "Anthropic requires Authorization: Bearer <token> or x-api-key.",
            UserWarning,
            stacklevel=3,
        )


class AnthropicClient(BaseClient[ISession]):
    """Client of the Anthropic API.

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

    def __init__(
        self,
        token: str | None = None,
        *,
        session: ISession | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
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

        Raises:
            ValueError: If both ``token`` and ``session`` are given, or
                neither is and ``ANTHROPIC_API_KEY`` is unset or empty, or
                ``default_max_tokens`` is not positive.
        """
        adapter = AnthropicGenerationAdapter(
            default_max_tokens=default_max_tokens
        )
        token = resolve_token(token, session, ANTHROPIC_API_KEY_ENV)

        if session is None:
            assert token is not None  # noqa: S101 - checked above
            session = session_factory(token)
        else:
            warn_about_session(session)

        super().__init__(session)
        self.generation: GenerationResource[Message, MessageStreamEvent] = (
            GenerationResource(session, adapter)
        )
