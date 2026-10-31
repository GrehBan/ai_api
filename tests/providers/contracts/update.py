"""Rebuild the wire-contract snapshots from the offline docs dump.

The dump is the ``docs-lite`` folder of ``GrehBan/docs-lite-mcp`` (the
data behind the ``unified-api-docs`` MCP server). For every endpoint in
``ENDPOINTS`` this script copies the request, response and stream-event
schemas, and keeps the documentation examples that the upstream schema
accepts: whole response bodies, and single stream events taken out of
the SSE examples. An example that looks like a response or a known
event but fails the upstream schema is not copied; it is listed under
``rejected`` in ``manifest.json`` with the first error, so nothing is
dropped silently.

Usage::

    uv run python -m tests.providers.contracts.update [DOCS_LITE_DIR]

Without an argument the dump is found through ``DOCS_LITE_DIR``, or
next to ``DOCS_SUITE_MCP_DIR`` (the variable ``.mcp.json`` uses).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from jsonschema import Draft202012Validator

CONTRACTS: Final = Path(__file__).parent
"""Folder of the snapshots."""

MANIFEST: Final = CONTRACTS / "manifest.json"
"""Provenance of every snapshot file."""


@dataclass(frozen=True)
class Endpoint:
    """One endpoint whose contract is snapshotted.

    Attributes:
        provider: Provider id in the dump.
        name: Folder name of the snapshot.
        operation: Operation id in the dump's ``_meta/enrichment.json``.
        stream: Stream contract id in ``_meta/stream-contracts.json``.
        kind: Field and value that mark a JSON example as a response
            body of this endpoint, such as ``("object", "response")``.
    """

    provider: str
    name: str
    operation: str
    stream: str
    kind: tuple[str, str]


REDACTED: Final = frozenset({"fallback_credit_token"})
"""Fields whose example values are replaced by ``REDACTION``.

They are opaque tokens (the documented one is base64 of
``Anthropic/Claude``) that secret scanners flag as credentials; the
library does not read them.
"""

REDACTION: Final = "<redacted>"
"""Value written in place of a ``REDACTED`` field."""


def redact(value: Any) -> Any:
    """Replace the values of ``REDACTED`` fields at any depth.

    Args:
        value: A decoded example.

    Returns:
        A copy with every ``REDACTED`` field set to ``REDACTION``.
    """
    if isinstance(value, dict):
        return {
            key: REDACTION if key in REDACTED else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


ENDPOINTS: Final = (
    Endpoint(
        provider="openai",
        name="responses",
        operation="openai:post:688886f44f64cc",
        stream="openai.responses",
        kind=("object", "response"),
    ),
    Endpoint(
        provider="anthropic",
        name="messages",
        operation="anthropic:post:c64fb925aa6487",
        stream="anthropic.messages",
        kind=("type", "message"),
    ),
)
"""Endpoints the library implements."""


def type_tags(schema: Any) -> set[str]:
    """Collect every ``type`` value a schema declares.

    Args:
        schema: A JSON schema, or any node of one.

    Returns:
        The ``const`` and ``enum`` values of every ``type`` property.
    """
    tags: set[str] = set()
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            kind = properties.get("type")
            if isinstance(kind, dict):
                if isinstance(kind.get("const"), str):
                    tags.add(kind["const"])
                tags.update(
                    value
                    for value in kind.get("enum", [])
                    if isinstance(value, str)
                )
        for value in schema.values():
            tags |= type_tags(value)
    elif isinstance(schema, list):
        for value in schema:
            tags |= type_tags(value)
    return tags


def event_tags(schema: dict[str, Any]) -> set[str]:
    """Collect the ``type`` of every top-level event of a stream schema.

    Args:
        schema: A stream schema whose ``anyOf`` (or ``oneOf``, as
            pydantic writes a tagged union) refers to one definition per
            event.

    Returns:
        The ``type`` values of those definitions, without the types of
        nested values.
    """
    tags: set[str] = set()
    for option in schema.get("anyOf") or schema["oneOf"]:
        name = option["$ref"].removeprefix("#/$defs/")
        kind = schema["$defs"][name]["properties"]["type"]
        tags |= type_tags({"properties": {"type": kind}})
    return tags


def sse_data(text: str) -> Iterator[str]:
    """Yield the ``data`` of every event of an SSE example.

    Args:
        text: The example, as the dump stores it.

    Yields:
        The joined ``data`` lines of each event.
    """
    lines: list[str] = []
    for line in [*text.splitlines(), ""]:
        if not line:
            if lines:
                yield "\n".join(lines)
            lines = []
        elif line.startswith("data:"):
            lines.append(line.removeprefix("data:").removeprefix(" "))


def first_error(validator: Draft202012Validator, value: Any) -> str | None:
    """Return the first schema error of a value.

    Args:
        validator: Validator of the upstream schema.
        value: The value to check.

    Returns:
        The error message cut to 200 characters, or ``None`` if the
        value is valid.
    """
    error = next(iter(validator.iter_errors(value)), None)
    return None if error is None else error.message[:200]


def read_json(path: Path) -> Any:
    """Read a JSON file.

    Args:
        path: The file.

    Returns:
        The decoded value.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write a JSON file the way the snapshots are stored.

    Args:
        path: The file; its folder is created.
        value: The value to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def canonical(value: Any) -> str:
    """Return a key for finding duplicate examples.

    Args:
        value: A decoded example.

    Returns:
        Its JSON with sorted keys.
    """
    return json.dumps(value, sort_keys=True)


def source_of(entry: dict[str, Any]) -> dict[str, Any]:
    """Describe where a schema came from.

    Args:
        entry: The schema entry of the dump's operation index.

    Returns:
        Its file, hash and provenance.
    """
    return {
        "file": entry["file"],
        "sha256": entry.get("sha256"),
        "provenance": entry.get("provenance"),
    }


def snapshot(
    docs: Path,
    endpoint: Endpoint,
    fixtures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the snapshot of one endpoint.

    Args:
        docs: The ``docs-lite`` folder.
        endpoint: The endpoint.
        fixtures: The dump's example index.

    Returns:
        The endpoint's manifest entry.
    """
    operation = read_json(docs / "_meta" / "enrichment.json")["operations"][
        endpoint.operation
    ]
    stream = read_json(docs / "_meta" / "stream-contracts.json")["contracts"][
        endpoint.stream
    ]
    request_entry = operation["requests"][0]
    response_entry = operation["responses"][0]
    schemas = {
        "request": read_json(docs / request_entry["file"]),
        "response": read_json(docs / response_entry["file"]),
        "stream": read_json(docs / stream["schema"]["file"]),
    }
    target = CONTRACTS / endpoint.provider / endpoint.name
    for name, schema in schemas.items():
        write_json(target / f"{name}.schema.json", schema)

    responses = Draft202012Validator(schemas["response"])
    events = Draft202012Validator(schemas["stream"])
    tags = event_tags(schemas["stream"])
    field, value = endpoint.kind
    kept: dict[str, list[dict[str, Any]]] = {"responses": [], "events": []}
    seen: set[str] = set()
    rejected: list[dict[str, Any]] = []

    for fixture in fixtures:
        if fixture["provider"] != endpoint.provider:
            continue
        text = (docs / fixture["file"]).read_text(encoding="utf-8")
        if fixture["kind"] == "json":
            candidates = [("responses", text)]
        else:
            candidates = [("events", data) for data in sse_data(text)]
        for group, raw in candidates:
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(body, dict):
                continue
            if group == "responses" and body.get(field) != value:
                continue
            if group == "events" and body.get("type") not in tags:
                continue
            validator = responses if group == "responses" else events
            error = first_error(validator, body)
            if error is not None:
                rejected.append(
                    {"id": fixture["id"], "group": group, "error": error}
                )
                continue
            key = canonical(body)
            if key in seen:
                continue
            seen.add(key)
            kept[group].append(
                {
                    "id": fixture["id"],
                    "source": fixture["source"],
                    "api_version": fixture.get("api_version"),
                    "body": redact(body),
                }
            )

    for group, items in kept.items():
        write_json(target / f"{group}.json", items)

    return {
        "operation": endpoint.operation,
        "stream_contract": endpoint.stream,
        "request": source_of(request_entry),
        "response": source_of(response_entry),
        "stream": {
            "file": stream["schema"]["file"],
            "provenance": stream.get("provenance"),
        },
        "responses": len(kept["responses"]),
        "events": len(kept["events"]),
        "rejected": rejected,
    }


def find_docs(argv: list[str]) -> Path:
    """Find the ``docs-lite`` folder.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        The folder.

    Raises:
        SystemExit: If no folder is given or found.
    """
    if argv:
        docs = Path(argv[0])
    elif "DOCS_LITE_DIR" in os.environ:
        docs = Path(os.environ["DOCS_LITE_DIR"])
    elif "DOCS_SUITE_MCP_DIR" in os.environ:
        docs = Path(os.environ["DOCS_SUITE_MCP_DIR"]).parent / "docs-lite"
    else:
        message = "pass the docs-lite folder or set DOCS_LITE_DIR"
        raise SystemExit(message)
    if not (docs / "_meta" / "enrichment.json").is_file():
        message = f"{docs} is not a docs-lite folder"
        raise SystemExit(message)
    return docs


def revision(docs: Path) -> str | None:
    """Return the git commit of the dump, if it is a checkout.

    Args:
        docs: The ``docs-lite`` folder.

    Returns:
        The commit hash, or ``None`` outside a git checkout.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "-C", str(docs), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main(argv: list[str]) -> None:
    """Rebuild every snapshot and the manifest.

    Args:
        argv: Command-line arguments without the program name.
    """
    docs = find_docs(argv)
    fixtures = read_json(docs / "_meta" / "fixtures.json")["fixtures"]
    endpoints = {
        f"{endpoint.provider}/{endpoint.name}": snapshot(
            docs, endpoint, fixtures
        )
        for endpoint in ENDPOINTS
    }
    write_json(
        MANIFEST,
        {
            "docs_revision": revision(docs),
            "sdk_versions": read_json(docs / "_meta" / "enrichment.json")[
                "sdk_versions"
            ],
            "endpoints": endpoints,
        },
    )
    for name, entry in endpoints.items():
        sys.stdout.write(
            f"{name}: {entry['responses']} responses, "
            f"{entry['events']} events, "
            f"{len(entry['rejected'])} rejected\n"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
