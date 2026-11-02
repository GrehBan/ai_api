---
name: ai-api-endpoint
description: Add or change an endpoint of an existing aiollm provider (OpenAI, Anthropic) - read the upstream schema from the unified-api-docs MCP, write method fields by the MISSING/None rule, response types with __upstream_schema__, the wire-contract snapshot and Case, adapters and shared-layer mapping, docs and tests. Use it in the aiollm repository whenever the owner asks for a new provider endpoint or resource (models, embeddings, files, tools, structured output...), a new request field on an existing method, or to follow an upstream schema change.
---

# aiollm endpoint

Every provider type and method in aiollm mirrors the provider's own schema,
and tests hold it there. An endpoint is done when the snapshot, the types, the
method fields and the shared layer all agree with upstream - not when a
request happens to work.

## 1. Read upstream, not memory

Use the `unified-api-docs` MCP (load its tools with ToolSearch): `find_endpoints`,
`get_endpoint`, `get_endpoint_schemas`, `get_stream_contract`, `list_fixtures`,
`compare_providers`. Note the operation id, the request and response `$defs`,
which fields are required, which are nullable, and the stream events. If the
MCP lacks something, say so and ask; do not guess a schema from an example.

Decide with the owner (`change-proposal`) when the endpoint touches the shared
layer: a shared type holds only what every provider has; the rest stays in the
provider's types and is reached through `raw`.

## 2. Snapshot and contract tests

- `tests/providers/contracts/update.py`: add an `Endpoint(provider, name,
  operation, stream, kind)` to `ENDPOINTS`, then refresh with
  `uv run python -m tests.providers.contracts.update [DOCS_LITE_DIR]`. Review
  the snapshot diff.
- `tests/providers/test_wire_contracts.py`: add a `Case` (methods, a request
  that sets every field, response model, event union, examples, adapter,
  model modules). The tests then check the request against the closed schema,
  every method field (`test_method_fields_match_the_request_schema`), and
  every model field against its `$def`.

## 3. Types

In `providers/<name>/types/`: models derive from `BaseAPITypeExtra`; every
model sets `__upstream_schema__` to the `$def` it mirrors (or `()`); `Literal`
for constants that name the object (`type`, `object`, `role`), `str` for
enumerations that grow (`status`, `stop_reason`, codes); response fields keep
`= None` defaults; unknown items go to the provider's `BaseUnknown` types.

## 4. Methods

In `providers/<name>/methods/`, following the upstream request schema
(`CLAUDE.md` "Optional fields"):

- required upstream → no default;
- optional → `= MISSING` with `MISSING` in the type;
- nullable upstream → `None` in the type; never `= None`;
- nested request data → `TypedDict` (`NotRequired` for optional keys);
- wire names via `validation_alias`/`serialization_alias`, never `alias`;
- `__retryable__`, `__stream_terminal__`, error events as the provider's
  other methods do; Anthropic methods send `VERSION_HEADERS`.

## 5. Adapters and resources

Adapters are pure converters (no I/O) with `provider = PROVIDER`; they pass
`MISSING` through unchanged and test it with `is MISSING`. A request the
provider cannot take raises `RequestValidationError` before sending. Shared
fields get a row in `docs/unification.md` (checked by
`tests/docs/test_unification.py`).

## 6. Tests, docs, release

Tests in `tests/providers/` (and the generation contract if the shared layer
changes), docs in `docs/guide/providers.md` (and generation/unification),
public names re-exported with sorted `__all__`, then `definition-of-done` and
`ai-api-release`. Report in Russian.
