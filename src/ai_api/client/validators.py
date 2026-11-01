"""Checks of the arguments shared by provider clients."""

from __future__ import annotations

import os

from ai_api.enums import Provider
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


def providers_of(
    sources: tuple[type, ...],
    kind: str,
) -> dict[str, Provider]:
    """Read the provider each class of a provider client declares.

    Args:
        sources: Classes of one kind, such as the adapter classes.
        kind: What they are, for the names and error messages.

    Returns:
        The provider of each class, keyed by ``"<kind> <class name>"``.

    Raises:
        TypeError: If a class declares no ``Provider``, or two classes
            share a name: one of them would be dropped from the check.
    """
    providers: dict[str, Provider] = {}

    for source in sources:
        name = f"{kind} {source.__name__}"

        if name in providers:
            message = f"Two {kind} classes are named {source.__name__}"
            raise TypeError(message)

        providers[name] = provider_of(source, kind)

    return providers


def provider_of(source: object, name: str) -> Provider:
    """Read the provider a part of a provider package declares.

    Args:
        source: An error resolver, or an adapter class.
        name: What the source is, for the error message.

    Returns:
        Its ``provider``.

    Raises:
        TypeError: If it declares no ``Provider``.
    """
    provider = getattr(source, "provider", None)

    if not isinstance(provider, Provider):
        message = f"The {name} {source!r} declares no provider"
        raise TypeError(message)

    return provider


def check_same_provider(**sources: Provider) -> None:
    """Check that every part of a provider client names one provider.

    A client, the error resolver of its sessions and its adapter each
    declare their provider; a mismatch means one of them belongs to
    another provider, which would mislabel every error and response.

    Args:
        **sources: The provider of each part, by the part's name.

    Raises:
        TypeError: If they differ; the message lists every source.
    """
    if len(set(sources.values())) > 1:
        listed = ", ".join(
            f"{name}={provider.value}" for name, provider in sources.items()
        )
        message = f"The parts of a provider client disagree: {listed}"
        raise TypeError(message)
