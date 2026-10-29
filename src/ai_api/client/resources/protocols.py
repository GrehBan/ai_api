"""Interfaces of objects that send requests through a session."""

from typing import Protocol

from ai_api.client.session.protocols import ISession


class ISessionAccess(Protocol):
    """Gives access to the session that sends the requests."""

    @property
    def session(self) -> ISession:
        """The session that sends the requests."""
        ...


class IResource(ISessionAccess, Protocol):
    """A group of use cases of one API area, such as responses."""
