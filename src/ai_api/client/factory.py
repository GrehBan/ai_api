"""Factory that builds sessions for one provider."""

from __future__ import annotations

from collections.abc import Callable
from typing import Concatenate

from ai_api.http.exceptions.protocols import IErrorResolver
from ai_api.http.protocols import IAPI, IAuth
from ai_api.session.protocols import ISession


class SessionFactory[SessionT: ISession, AuthT: IAuth, **P]:
    """Builds sessions with a provider's API and error resolver.

    The remaining session arguments (``timeout``, ``max_body_size``, ...)
    are taken from ``session_type`` and checked by type checkers.

    Attributes:
        session_type: Session class, or a callable that builds one from
            the API, the auth and the error resolver.
        auth_factory: Turns a token string into an ``IAuth``.
        api: The provider's ``IAPI``.
        error_resolver: The provider's error resolver.

    Example:
        >>> factory = SessionFactory(
        ...     AiohttpSession,
        ...     auth_factory=BearerAuth,
        ...     api=OPENAI_API,
        ...     error_resolver=OpenAIErrorResolver(),
        ... )
        >>> session = factory("sk-...", max_body_size=10_000_000)
    """

    def __init__(
        self,
        session_type: Callable[
            Concatenate[IAPI, AuthT, IErrorResolver, P],
            SessionT,
        ],
        *,
        auth_factory: Callable[[str], AuthT],
        api: IAPI,
        error_resolver: IErrorResolver,
    ) -> None:
        """Store the provider's configuration.

        Args:
            session_type: Session class, or a callable that builds one
                from the API, the auth and the error resolver.
            auth_factory: Turns a token string into an ``IAuth``.
            api: The provider's ``IAPI``.
            error_resolver: The provider's error resolver.
        """
        self.session_type = session_type
        self.auth_factory = auth_factory
        self.api = api
        self.error_resolver = error_resolver

    def __call__(
        self,
        auth: AuthT | str,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> SessionT:
        """Build a session.

        Args:
            auth: An ``IAuth``, or a token passed to ``auth_factory``.
            *args: Further positional arguments of ``session_type``.
            **kwargs: Further keyword arguments of ``session_type``.

        Returns:
            The new session.
        """
        if isinstance(auth, str):
            auth = self.auth_factory(auth)

        return self.session_type(
            self.api,
            auth,
            self.error_resolver,
            *args,
            **kwargs,
        )
