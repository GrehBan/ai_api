"""Block E: default headers and bearer authentication."""

from __future__ import annotations

import dataclasses
import sys

import pytest

from ai_api.__meta__ import __version__
from ai_api.http.auth import APIKeyAuth, BearerAuth
from ai_api.http.headers import (
    AUTHORIZATION,
    CONTENT_TYPE,
    SERVER_SOFTWARE,
    USER_AGENT,
    Headers,
)
from tests.conftest import TEST_TOKEN


def test_headers_default_to_empty() -> None:
    assert dict(Headers().headers) == {}


def test_headers_initial_values_are_case_insensitive() -> None:
    headers = Headers({"X-Trace": "1"})

    assert headers.headers["x-trace"] == "1"


def test_add_overwrites_case_insensitively() -> None:
    headers = Headers({"X-Trace": "1", "Keep": "k"})

    headers.add({"x-trace": "2", "New": "n"})

    assert headers.headers.getall("X-TRACE") == ["2"]
    assert headers.headers["keep"] == "k"
    assert headers.headers["new"] == "n"


def test_build_gives_extra_priority_without_mutating() -> None:
    headers = Headers({"User-Agent": "mine", "X-Trace": "1"})

    built = headers.build({"user-agent": "lib", "Authorization": "Bearer t"})

    assert built.getall("User-Agent") == ["lib"]
    assert built["x-trace"] == "1"
    assert built["authorization"] == "Bearer t"
    assert dict(headers.headers) == {"User-Agent": "mine", "X-Trace": "1"}


def test_copy_is_independent() -> None:
    headers = Headers({"X-Trace": "1"})

    copied = headers.copy()
    copied["X-Trace"] = "2"
    headers.add({"Other": "o"})

    assert headers.headers["X-Trace"] == "1"
    assert "Other" not in copied


def test_header_names_are_case_insensitive() -> None:
    assert USER_AGENT == "User-Agent"
    assert AUTHORIZATION == "Authorization"
    assert CONTENT_TYPE == "Content-Type"


def test_server_software_contains_version() -> None:
    python = f"Python/{sys.version_info[0]}.{sys.version_info[1]}"

    assert f"ai_api/{__version__}" in SERVER_SOFTWARE
    assert SERVER_SOFTWARE.startswith(python)


def test_bearer_encode() -> None:
    assert BearerAuth(TEST_TOKEN).encode() == f"Bearer {TEST_TOKEN}"


def test_bearer_rejects_empty_token() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        BearerAuth("")


@pytest.mark.parametrize(
    "token",
    ["a b", " token", "token\n", "to\tken"],
    ids=["space", "leading-space", "newline", "tab"],
)
def test_bearer_rejects_whitespace(token: str) -> None:
    with pytest.raises(ValueError, match="cannot contain whitespace"):
        BearerAuth(token)


def test_bearer_token_not_in_repr() -> None:
    auth = BearerAuth("sk-secret-123")

    assert "sk-secret-123" not in repr(auth)
    assert "sk-secret-123" not in str(auth)


def test_bearer_is_frozen() -> None:
    auth = BearerAuth(TEST_TOKEN)

    with pytest.raises(dataclasses.FrozenInstanceError):
        auth.token = "other"  # type: ignore[misc]


def test_bearer_header_name() -> None:
    assert BearerAuth(TEST_TOKEN).header_name == AUTHORIZATION


def test_api_key_defaults_to_x_api_key() -> None:
    auth = APIKeyAuth("sk-secret-123")

    assert (auth.header_name, auth.encode()) == ("x-api-key", "sk-secret-123")
    assert "sk-secret-123" not in repr(auth)


def test_api_key_custom_header() -> None:
    assert APIKeyAuth("k", header_name="api-key").header_name == "api-key"


@pytest.mark.parametrize(
    ("key", "header_name", "message"),
    [
        ("", "x-api-key", "key cannot be empty"),
        ("a b", "x-api-key", "key cannot contain whitespace"),
        ("k", "", "header name cannot be empty"),
        ("k", "x api", "header name cannot contain whitespace"),
    ],
)
def test_api_key_rejects_invalid_values(
    key: str,
    header_name: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        APIKeyAuth(key, header_name=header_name)
