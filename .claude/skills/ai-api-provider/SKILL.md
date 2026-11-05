---
name: ai-api-provider
description: Add a new AI provider package to aiollm (e.g. OpenRouter, Gemini) - api.py with PROVIDER, resolver.py with the error table, client.py with SessionFactory and BaseProviderClient, the first endpoints, and the shared generation/client contract cases. Use it in the aiollm repository whenever the owner asks to support a new provider or vendor, or asks what adding one would take. For a new endpoint of an existing provider use ai-api-endpoint instead.
---

# aiollm provider

A provider is a flat package built from the same parts as `providers/openai/`
and `providers/anthropic/`; the steps are in `docs/guide/extending.md` and
the conventions in `CLAUDE.md`. Read both first - this skill is the order of
work and the checks that make a provider complete.

## 0. Agree first

Adding a provider extends the closed `Provider` enum and the public surface:
propose it (`change-proposal`) with what upstream offers (from the
`unified-api-docs` MCP: `suite_info`, `get_capability_matrix`,
`compare_providers`), the auth scheme, error format and stream format, and
which shared resources it will support. Code starts on the owner's go.

## 1. Package skeleton (`src/aiollm/providers/<name>/`)

1. `api.py`: `BaseAPI(base_url, version)` and `PROVIDER: Final =
   Provider.<NAME>`; add the member to `Provider` in `enums.py`.
2. `resolver.py`: `BaseErrorResolver` subclass with `provider = PROVIDER`,
   `error_statuses` (in-band code/type → status), `get_error_code` or
   `classify_status` for codes such as quota vs rate limit; in-band failures
   are never transient.
3. `client.py`: module-level `session_factory = SessionFactory(AiohttpSession,
   auth_factory=..., api=..., error_resolver=...)`, and a `BaseProviderClient`
   subclass with `provider`, `session_factory`, `api_key_env`, `auth_headers`
   and each resource as `name = bind_resource(Resource[...], Adapter)`
   (`adapter_factory=` for an adapter with options, read from the client);
   `__init__` types `**session_kwargs`, keeps the adapter options and calls
   `super().__init__()`, which builds adapters before the session. The
   generation adapter has `with_options` with the provider's options
   `TypedDict` (`types/options.py`), and the client binds
   `GenerationResource[Raw, Event, Options]`.
4. `__init__.py` re-exports the client (the one `__init__` allowed to import a
   transport), sorted `__all__`.

## 2. First endpoints

Use `ai-api-endpoint` for each: types, methods, snapshot, `Case`, adapters.

## 3. Shared contracts

- `tests/providers/test_generation_contract.py`: add a `Case` - the same
  scenarios run against every provider's `client.generation`.
- `tests/providers/test_provider_clients.py`: add a `ClientCase`.
- `test_<name>_client.py` keeps only what is the provider's own.
- `docs/unification.md`: a column's worth of mapping for every shared field
  and error case.

## 4. Docs, public API, release

`docs/guide/providers.md` section, README feature list, `CLAUDE.md` if an
invariant changes, `tests/unit/test_public_api.py` passing, then
`definition-of-done` and `ai-api-release` (a new provider is a minor bump).
Report in Russian.
