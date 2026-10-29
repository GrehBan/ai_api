"""Checks of the arguments shared by provider clients."""

from ai_api.client.session.protocols import ISession


def check_session_creds(token: str | None, session: ISession | None) -> None:
    """Ensure exactly one of a token and a session is given.

    Args:
        token: The API token, if any.
        session: A ready session, if any.

    Raises:
        ValueError: If both or neither are given.
    """
    if session is not None and token is not None:
        raise ValueError("Provide either a token or a session, not both")

    if session is None and token is None:
        raise ValueError("Either a token or a session must be provided")
