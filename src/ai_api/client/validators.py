"""Checks of the arguments shared by provider clients."""

from __future__ import annotations

import os

from ai_api.session.protocols import ISession


def resolve_token(
    token: str | None,
    session: ISession | None,
    env: str,
) -> str | None:
    """Pick the token of a client and check it against the session.

    Exactly one of a token and a session must be given. Without either,
    the token is read from the environment variable ``env``, so an
    explicit argument always wins over the environment and a session is
    never combined with a token from it; an empty variable counts as
    unset.

    Args:
        token: The API token, if any.
        session: A ready session, if any.
        env: Name of the environment variable with the token, e.g.
            ``OPENAI_API_KEY``.

    Returns:
        The token to build a session from, or ``None`` when a session
        is given.

    Raises:
        ValueError: If both a token and a session are given, or neither
            is and the variable is unset or empty.
    """
    if session is not None:
        if token is not None:
            raise ValueError("Provide either a token or a session, not both")

        return None

    if token is None:
        token = os.environ.get(env) or None

    if token is None:
        raise ValueError(f"Provide a token or a session, or set {env}")

    return token
