"""Packages import only from the layers below them (CLAUDE.md).

The order is core → data and HTTP primitives → methods → response →
session → client → adapters → resources → providers. The exceptions are
the ones ``CLAUDE.md`` names: ``methods/base.py`` imports the default
parsers, and protocols that refer to each other import under
``TYPE_CHECKING``. Transport modules are never imported from an
``__init__.py``, except a provider package's.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

import aiollm

ROOT = Path(aiollm.__file__).parent

LAYERS = {
    "__meta__": 0,
    "enums": 0,
    "exceptions": 0,
    "types": 1,
    "serialization": 1,
    "http": 1,
    "methods": 2,
    "response": 3,
    "session": 4,
    "client": 5,
    "adapters": 6,
    "resources": 7,
    "providers": 8,
}
"""The layer of each top-level package or module of ``aiollm``."""

ALLOWED = {("methods/base.py", "response.parse")}
"""Upward imports ``CLAUDE.md`` allows: the module and the target."""

TRANSPORTS = {"aiohttp"}
"""Names of transport modules (``<layer>/<transport>.py``)."""


@dataclass(frozen=True)
class Import:
    """One ``from aiollm... import`` of a module."""

    module: str
    target: str
    line: int
    deferred: bool


def imports(module: str, source: str) -> list[Import]:
    """Find the ``aiollm`` imports of a module's source."""
    tree = ast.parse(source)
    deferred: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(
            node.test
        ):
            deferred.update(
                sub.lineno
                for sub in ast.walk(node)
                if isinstance(sub, ast.ImportFrom)
            )

    return [
        Import(
            module=module,
            target=node.module.removeprefix("aiollm."),
            line=node.lineno,
            deferred=node.lineno in deferred,
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("aiollm.")
    ]


def layer(name: str) -> int:
    """The layer of a module path or a dotted target."""
    return LAYERS[name.replace("/", ".").split(".")[0]]


def violations(found: list[Import]) -> list[str]:
    """Upward imports that no rule allows."""
    wrong = []

    for item in found:
        if layer(item.target) <= layer(item.module):
            continue
        if any(
            item.module == module and item.target.startswith(target)
            for module, target in ALLOWED
        ):
            continue
        if (
            item.deferred
            and item.module.endswith("protocols.py")
            and item.target.endswith("protocols")
        ):
            continue
        wrong.append(f"{item.module}:{item.line} imports {item.target}")

    return wrong


def transport_imports(found: list[Import]) -> list[str]:
    """Transport modules imported from an ``__init__.py``."""
    return [
        f"{item.module}:{item.line} imports {item.target}"
        for item in found
        if item.module.endswith("__init__.py")
        and not item.module.startswith("providers/")
        and item.target.split(".")[-1] in TRANSPORTS
    ]


def library_imports() -> list[Import]:
    """The ``aiollm`` imports of every module of the library."""
    found = []

    for path in sorted(ROOT.rglob("*.py")):
        module = path.relative_to(ROOT).as_posix()
        if module == "__init__.py":
            continue
        found += imports(module, path.read_text(encoding="utf-8"))

    return found


def test_every_package_has_a_layer() -> None:
    names = {
        path.stem if path.is_file() else path.name
        for path in ROOT.iterdir()
        if path.name != "__init__.py"
        and (path.suffix == ".py" or (path / "__init__.py").exists())
    }

    assert names == set(LAYERS)


def test_no_module_imports_from_a_higher_layer() -> None:
    assert violations(library_imports()) == []


def test_no_init_imports_a_transport() -> None:
    assert transport_imports(library_imports()) == []


CASES = {
    "upward": (
        "client/base.py",
        ["from aiollm.resources.binding import X"],
        ["client/base.py:1 imports resources.binding"],
    ),
    "upward-outside-the-allowed-module": (
        "methods/fields.py",
        ["from aiollm.response.parse.sse import X"],
        ["methods/fields.py:1 imports response.parse.sse"],
    ),
    "deferred-but-no-protocol": (
        "session/base.py",
        ["if TYPE_CHECKING:", "    from aiollm.client.base import X"],
        ["session/base.py:2 imports client.base"],
    ),
    "allowed": (
        "methods/base.py",
        ["from aiollm.response.parse.sse import X"],
        [],
    ),
    "protocols-under-type-checking": (
        "methods/protocols.py",
        [
            "if TYPE_CHECKING:",
            "    from aiollm.response.parse.protocols import X",
        ],
        [],
    ),
    "downward": (
        "resources/base.py",
        ["from aiollm.session.base import X"],
        [],
    ),
}
"""Sources the check is tried on, as lines, and what it must report."""


@pytest.mark.parametrize("case", CASES)
def test_the_check_tells_what_is_allowed(case: str) -> None:
    # The check itself is tested, so a passing run means no violation.
    module, lines, found = CASES[case]

    assert violations(imports(module, "\n".join(lines))) == found


def test_the_check_finds_a_transport_in_an_init() -> None:
    source = "from aiollm.session.aiohttp import X"
    found = imports("session/__init__.py", source)
    provider = imports("providers/openai/__init__.py", source)

    assert transport_imports(found) == [
        "session/__init__.py:1 imports session.aiohttp"
    ]
    assert transport_imports(provider) == []
