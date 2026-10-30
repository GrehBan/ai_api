"""Base classes of resources and of everything that holds a session."""

from __future__ import annotations

from ai_api.session.protocols import ISession


class BaseSessionAccess[SessionT: ISession = ISession]:
    """Holds the session that sends the requests.

    ``SessionT`` keeps the session's own type, as in ``BaseClient``, so a
    subclass built on ``AiohttpSession`` reaches its settings without a
    cast.
    """

    def __init__(self, session: SessionT) -> None:
        """Keep the session.

        Args:
            session: The session that sends the requests.
        """
        self._session = session

    @property
    def session(self) -> SessionT:
        """The session that sends the requests."""
        return self._session


class BaseResource[SessionT: ISession = ISession](
    BaseSessionAccess[SessionT],
):
    """Base class for a group of use cases of one API area.

    Example:
        >>> class Responses(BaseResource):
        ...     def __init__(self, session: ISession) -> None:
        ...         super().__init__(session)
        ...         self.generate = Generate(session)
    """
