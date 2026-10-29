"""Interface of a session factory."""

from typing import Protocol

from ai_api.client.auth.protocols import IAuth
from ai_api.client.session.protocols import ISession


class ISessionFactory[SessionT: ISession, AuthT: IAuth, **P](Protocol):
    """Builds sessions for one provider."""

    def __call__(
        self,
        auth: AuthT | str,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> SessionT:
        """Build a session.

        Args:
            auth: An ``IAuth``, or a token.
            *args: Further positional arguments of the session.
            **kwargs: Further keyword arguments of the session.

        Returns:
            The new session.
        """
        ...
