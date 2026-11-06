---
name: ai-api-endpoint
description: Add or change an endpoint of an existing aiollm provider (OpenAI, Anthropic) - read the upstream schema from the unified-api-docs MCP, generate types and request parameters with scripts/generate_provider_types.py (aiollm-codegen), hand-write the method classes, the wire-contract snapshot and Case, adapters and shared-layer mapping, docs and tests. Use it in the aiollm repository whenever the owner asks for a new provider endpoint or resource (models, embeddings, files, tools, structured output...), a new request field on an existing method, or to follow an upstream schema change.
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
  operation, stream, kind)` to `ENDPOINTS`, with `openapi=OpenAPIOperation(...)`
  naming the request, response and stream components where the provider
  publishes an official OpenAPI file (OpenAI), then refresh with
  `OPENAI_OPENAPI=<openapi.yaml> uv run python -m tests.providers.contracts.update [DOCS_LITE_DIR]`.
  Review the snapshot diff and `manifest.json` (its `rejected` examples).
- `tests/providers/test_wire_contracts.py`: add a `Case` (methods, a request
  that sets every field the adapter and the options fill, response model,
  event union, examples, adapter, options, type and parameter modules). The
  tests then check the request against the closed schema, every method field
  (`test_method_fields_match_the_request_schema`), and every model field
  against its `$def`, request parameters both ways.

## 3. Types and request parameters

They are generated, never written by hand: `scripts/generate_provider_types.py`
runs aiollm-codegen (`../aiollm_codegen`) for each provider, OpenAI from its
official OpenAPI file restricted to the endpoints' operations, Anthropic from
the docs-lite snapshot, and replaces `providers/<name>/types/<family>.py` and
`methods/params/`. A new endpoint adds its operation to the script. The
generated models already follow the rules the tests check:
`__upstream_schema__` on every model, `Literal` for constants and closed
request enumerations, `str` for response enumerations that grow, `MISSING`
for optional request fields and `None` only where upstream allows `null`,
unknown variants as `UnknownVariant`, request models with `extra="forbid"`.
A defect in them is fixed in aiollm-codegen (its own project), then
regenerated. `types/options.py` stays hand-written: its `TypedDict`s mirror
the method fields (`test_provider_options.py`), so dict literals type-check.

## 4. Methods

In `providers/<name>/methods/<endpoint>.py`, hand-written over the generated
parameter mixins: `__request_path__` and the other dunders,
`__retryable__`, `__stream_terminal__`, error events and parsers as the
provider's other methods do (Anthropic methods send `VERSION_HEADERS`), and
the limits the schema leaves out (OpenAI's `metadata`, Anthropic's
`MessageLimits`) as narrowed fields or validators, so the generated
parameters are never edited.

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
