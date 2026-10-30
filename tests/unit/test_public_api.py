"""Block P: the names re-exported by the packages."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

PACKAGES = [
    "ai_api",
    "ai_api.adapters",
    "ai_api.client",
    "ai_api.http",
    "ai_api.http.exceptions",
    "ai_api.methods",
    "ai_api.providers.anthropic",
    "ai_api.providers.anthropic.adapters",
    "ai_api.providers.anthropic.methods",
    "ai_api.providers.anthropic.types",
    "ai_api.providers.openai",
    "ai_api.providers.openai.adapters",
    "ai_api.providers.openai.methods",
    "ai_api.providers.openai.types",
    "ai_api.resources",
    "ai_api.response",
    "ai_api.response.parse",
    "ai_api.serialization",
    "ai_api.session",
    "ai_api.session.body",
    "ai_api.types",
    "ai_api.types.requests",
    "ai_api.types.responses",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_all_names_exist_and_are_sorted(name: str) -> None:
    package = importlib.import_module(name)
    exported: list[str] = package.__all__

    assert exported == sorted(exported)
    assert all(hasattr(package, item) for item in exported)


def test_root_reexports_the_error_tree() -> None:
    import ai_api  # noqa: PLC0415 - checked against its subpackage
    from ai_api.http import exceptions  # noqa: PLC0415

    errors = set(exceptions.__all__) - {
        "BaseErrorResolver",
        "HTTP_ERRORS",
        "IErrorResolver",
    }

    assert errors <= set(ai_api.__all__)


def test_importing_the_root_does_not_import_a_transport() -> None:
    code = "import sys, ai_api; print('aiohttp' in sys.modules)"

    result = subprocess.run(  # noqa: S603 - fixed command
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
