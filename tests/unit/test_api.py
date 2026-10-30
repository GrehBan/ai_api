"""The default ``IAPI`` implementation."""

from __future__ import annotations

import pytest
from yarl import URL

from ai_api.http.api import BaseAPI


@pytest.mark.parametrize(
    "base_url",
    ["https://api.example.com", URL("https://api.example.com")],
    ids=["str", "url"],
)
def test_base_url_is_parsed(base_url: str | URL) -> None:
    api = BaseAPI(base_url, "v1")

    assert api.base_url == URL("https://api.example.com")
    assert api.version == "v1"


@pytest.mark.parametrize(
    ("base_url", "message"),
    [
        ("ftp://api.example.com", "absolute HTTP"),
        ("/v1", "absolute HTTP"),
        ("https://", "absolute HTTP"),
        ("https://api.example.com?key=1", "query or fragment"),
        ("https://api.example.com#top", "query or fragment"),
    ],
    ids=["scheme", "relative", "no-host", "query", "fragment"],
)
def test_invalid_base_url_is_rejected(base_url: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BaseAPI(base_url, "v1")


@pytest.mark.parametrize(
    ("base_url", "version", "expected"),
    [
        ("https://a.test", "v1", "https://a.test/v1/models/m%2F1"),
        ("https://a.test/api/", "v1", "https://a.test/api/v1/models/m%2F1"),
        ("https://a.test", "", "https://a.test/models/m%2F1"),
    ],
    ids=["root", "path-prefix", "no-version"],
)
def test_build_url_keeps_the_encoded_path(
    base_url: str,
    version: str,
    expected: str,
) -> None:
    # Joining with yarl's "/" would turn %2F into %252F.
    assert BaseAPI(base_url, version).build_url("/models/m%2F1") == expected


def test_build_url_encodes_query() -> None:
    url = BaseAPI("https://a.test", "v1").build_url("/s", {"q": "a b&c"})

    assert url == "https://a.test/v1/s?q=a+b%26c"


def test_build_url_repeats_list_keys() -> None:
    url = BaseAPI("https://a.test", "v1").build_url(
        "/s",
        {"ids": ["a", "b c"], "empty": [], "q": "x"},
    )

    assert url == "https://a.test/v1/s?ids=a&ids=b+c&q=x"


@pytest.mark.parametrize("version", ["v1", "2024-06-01", "v1beta", ""])
def test_valid_version_is_kept(version: str) -> None:
    assert BaseAPI("https://a.test", version).version == version


@pytest.mark.parametrize(
    "version",
    ["../admin", "..", ".", "v1/admin", "v1?x=1", "v1#x", "v 1", "%2e%2e"],
)
def test_unsafe_version_is_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="one path segment"):
        BaseAPI("https://a.test", version)
