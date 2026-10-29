"""Client of the OpenAI API."""

import warnings
from typing import Final

from ai_api.client.auth.bearer import BearerAuth
from ai_api.client.base import BaseClient
from ai_api.client.exceptions.error_resolver.base import BaseErrorResolver
from ai_api.client.factory.factory import SessionFactory
from ai_api.client.headers.names import AUTHORIZATION
from ai_api.client.session.aiohttp import AiohttpSession
from ai_api.client.session.base import BaseSession
from ai_api.client.session.protocols import ISession
from ai_api.client.validators import check_session_creds
from ai_api.providers.openai.client.api.openai import OPENAI_API
from ai_api.providers.openai.client.exceptions.error_resolver.openai import (
    OpenAIErrorResolver,
)

session_factory: Final = SessionFactory(
    AiohttpSession,
    auth_factory=BearerAuth,
    api=OPENAI_API,
    error_resolver=OpenAIErrorResolver(),
)
"""Builds the ``AiohttpSession`` of an ``OpenAIClient`` from a token."""


def validate_session(session: ISession) -> None:
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

    Example:
        >>> async with OpenAIClient("sk-...") as client:
        ...     models = await client.send_request(ListModels())
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        session: ISession | None = None,
    ) -> None:
        """Create the client from a token or a ready session.

        Args:
            token: API key; an ``AiohttpSession`` with OpenAI's URL, error
                resolver and ``BearerAuth`` is created from it.
            session: A ready session, for custom settings; it is checked
                with ``validate_session``.

        Raises:
            ValueError: If both or neither of ``token`` and ``session``
                are given.
        """
        check_session_creds(token, session)

        if session is None:
            assert token is not None  # noqa: S101 - checked above
            session = session_factory(token)
        else:
            validate_session(session)

        super().__init__(session)
