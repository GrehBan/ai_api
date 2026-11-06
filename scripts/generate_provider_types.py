"""Regenerate the provider types and request parameters with aiollm-codegen.

Run from the aiollm checkout, with aiollm_codegen next to it:

    uv run python scripts/generate_provider_types.py [--codegen DIR]

OpenAI's types come from its official OpenAPI file (``OPENAI_OPENAPI``,
by default the copy in aiollm_codegen's ``examples/``), restricted to
``createResponse``; Anthropic's come from the docs-lite contract snapshot
in ``tests/providers/contracts/anthropic/messages``. For each provider the
script replaces ``types/<family>.py`` and ``methods/params/`` with the
generated modules, keeps the hand-written ``types/options.py`` and the
method classes in ``methods/``, and writes ``types/__init__.py`` from the
generated one with the options added. Nothing else is touched.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ROOT / "src" / "aiollm" / "providers"
CONTRACTS = ROOT / "tests" / "providers" / "contracts"
ANTHROPIC_VERSION = "2026-10-21"
HAND_WRITTEN_TYPES = {"options.py"}
"""Modules of ``types/`` the script never replaces."""


def codegen(codegen_dir: Path, *args: str, output: Path) -> None:
    """Run aiollm-codegen from its own project."""
    command = ["uv", "run", "aiollm-codegen", *args, "-o", str(output)]
    subprocess.run(command, cwd=codegen_dir, check=True)  # noqa: S603


def exported(module: Path) -> tuple[list[str], list[str]]:
    """Read the import lines and ``__all__`` of a generated ``__init__``."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    lines = module.read_text(encoding="utf-8").splitlines()
    imports: list[str] = []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imports.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            names = list(ast.literal_eval(node.value))
    return imports, names


def options_names(provider: Path) -> list[str]:
    """Public names of the hand-written ``types/options.py``."""
    tree = ast.parse((provider / "types" / "options.py").read_text("utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]


def install(name: str, generated: Path) -> None:
    """Replace one provider's generated modules."""
    provider = PROVIDERS / name
    types_dir = provider / "types"
    params_dir = provider / "methods" / "params"

    for module in types_dir.glob("*.py"):
        if module.name not in HAND_WRITTEN_TYPES:
            module.unlink()
    shutil.rmtree(params_dir, ignore_errors=True)
    params_dir.mkdir(parents=True)

    for module in (generated / "types").glob("*.py"):
        if module.name != "__init__.py":
            shutil.copy2(module, types_dir / module.name)
    for module in (generated / "methods" / "params").glob("*.py"):
        shutil.copy2(module, params_dir / module.name)

    imports, names = exported(generated / "types" / "__init__.py")
    options = options_names(provider)
    title = "OpenAI" if name == "openai" else "Anthropic"
    body = [
        f'"""{title}\'s own types and the options of a generation call.',
        "",
        "Every module but ``options`` is written by aiollm-codegen through",
        "``scripts/generate_provider_types.py``; do not edit them by hand.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        *imports,
        f"from aiollm.providers.{name}.types.options import (",
        *[f"    {option}," for option in sorted(options)],
        ")",
        "",
        "__all__ = [",
        *[f'    "{item}",' for item in sorted({*names, *options})],
        "]",
        "",
    ]
    (types_dir / "__init__.py").write_text(
        "\n".join(body), encoding="utf-8", newline="\n"
    )


def main() -> None:
    """Generate both providers and install them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--codegen", type=Path, default=ROOT.parent / "aiollm_codegen"
    )
    args = parser.parse_args()
    codegen_dir: Path = args.codegen.resolve()
    openapi = Path(
        os.environ.get(
            "OPENAI_OPENAPI", codegen_dir / "examples" / "openai.openapi.yaml"
        )
    ).resolve()

    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        codegen(
            codegen_dir,
            str(openapi),
            "--package",
            "--operation",
            "createResponse",
            output=out / "openai",
        )
        codegen(
            codegen_dir,
            str(CONTRACTS / "anthropic" / "messages"),
            "--package",
            "--contract",
            "/messages",
            "--header",
            f"anthropic-version={ANTHROPIC_VERSION}",
            output=out / "anthropic",
        )
        install("openai", out / "openai")
        install("anthropic", out / "anthropic")

    inits = [
        str(PROVIDERS / name / "types" / "__init__.py")
        for name in ("openai", "anthropic")
    ]
    ruff = ["uv", "run", "ruff"]
    for command in (
        [*ruff, "check", "--select", "I", "--fix", "-q", *inits],
        [*ruff, "format", "-q", "src/aiollm/providers"],
    ):
        subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603
    print("provider types regenerated", file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    main()
