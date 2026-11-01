"""``SessionFactory``: a provider's session from a token."""

from __future__ import annotations

from typing import Any

from ai_api.http.auth import BearerAuth
from ai_api.providers.openai.api import OPENAI_API
from ai_api.providers.openai.resolver import OpenAIErrorResolver
from ai_api.session.factory import SessionFactory
from tests.conftest import FakeTransport


def test_factory_turns_a_token_into_auth_and_forwards_arguments() -> None:
    calls: list[tuple[Any, ...]] = []

    def build(
        api: Any,
        auth: BearerAuth,
        resolver: Any,
        *,
        limit: int = 1,
    ) -> FakeTransport:
        calls.append((api, auth, resolver, limit))
        return FakeTransport()

    resolver = OpenAIErrorResolver()
    factory = SessionFactory(
        build,
        auth_factory=BearerAuth,
        api=OPENAI_API,
        error_resolver=resolver,
    )

    factory("token", limit=5)
    factory(BearerAuth("ready"))

    assert calls == [
        (OPENAI_API, BearerAuth("token"), resolver, 5),
        (OPENAI_API, BearerAuth("ready"), resolver, 1),
    ]
