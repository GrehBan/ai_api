"""Block P: the names re-exported by the packages."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

PACKAGES = [
    "aiollm",
    "aiollm.adapters",
    "aiollm.client",
    "aiollm.http",
    "aiollm.http.exceptions",
    "aiollm.methods",
    "aiollm.providers.anthropic",
    "aiollm.providers.anthropic.adapters",
    "aiollm.providers.anthropic.methods",
    "aiollm.providers.anthropic.types",
    "aiollm.providers.openai",
    "aiollm.providers.openai.adapters",
    "aiollm.providers.openai.methods",
    "aiollm.providers.openai.types",
    "aiollm.resources",
    "aiollm.response",
    "aiollm.response.parse",
    "aiollm.serialization",
    "aiollm.session",
    "aiollm.session.body",
    "aiollm.session.retry",
    "aiollm.types",
    "aiollm.types.requests",
    "aiollm.types.responses",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_all_names_exist_and_are_sorted(name: str) -> None:
    package = importlib.import_module(name)
    exported: list[str] = package.__all__

    assert exported == sorted(exported)
    assert all(hasattr(package, item) for item in exported)


def test_root_reexports_the_error_tree() -> None:
    import aiollm  # noqa: PLC0415 - checked against its subpackage
    from aiollm.http import exceptions  # noqa: PLC0415

    errors = set(exceptions.__all__) - {
        "BaseErrorResolver",
        "HTTP_ERRORS",
        "IErrorResolver",
    }

    assert errors <= set(aiollm.__all__)


def test_importing_the_root_does_not_import_a_transport() -> None:
    code = "import sys, aiollm; print('aiohttp' in sys.modules)"

    result = subprocess.run(  # noqa: S603 - fixed command
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
