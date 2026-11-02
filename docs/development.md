# Development

## Project layout

```text
src/ai_api/
├── enums.py            Provider, BodyFormat, ResponseFormat, Role,
│                       IncompleteReason, GenerationEventType
├── exceptions.py       AIAPIError (the root of all library errors),
│                       RequestValidationError, SessionMismatchWarning
├── types/              BaseAPIType, files, matchers, BaseUnknown; requests/ and responses/
│                       hold the shared GenerationRequest/Response, events
├── serialization/      JSON codec, body value normalization
├── http/               IAPI and BaseAPI, BearerAuth, APIKeyAuth, headers
│   └── exceptions/     status, transport and response errors, resolver
├── methods/            BaseAPIMethod, BaseStreamMethod, path templates
├── response/           BufferedResponse, Stream, transform_stream, map_stream;
│                       parse/ parsers
├── session/            BaseSession, AiohttpSession, SessionFactory, retries,
│                       body/ builders
├── client/             BaseClient, BaseProviderClient
├── adapters/           adapters between shared types and provider methods
├── resources/          GenerationResource, use cases and AdapterUseCase orchestrators
└── providers/
    ├── anthropic/      AnthropicClient, ANTHROPIC_API, API_VERSION,
    │                   AnthropicErrorResolver, methods/, types/ and
    │                   adapters/ of the Messages API
    └── openai/         OpenAIClient, OPENAI_API, OpenAIErrorResolver,
                        methods/, types/ and adapters/ of the Responses API
```

The names most code needs are re-exported from `ai_api` (`from ai_api import BaseAPIMethod, RateLimitError`), and each layer package re-exports its own, e.g. `from ai_api.http import Query`. Transport modules such as `ai_api.session.aiohttp` and provider packages such as `ai_api.providers.openai` are imported on their own, so `import ai_api` does not import aiohttp.

## Commands

```bash
uv sync --all-groups                 # runtime, dev and test dependencies
uv run ruff check src tests          # lint
uv run ruff format src tests         # format (CI: --check)
uv run mypy src                      # strict type checking
uv run pyright                       # strict type checking of src and tests
uv run pytest --cov                  # tests with branch coverage
uv run pytest --mypy -m mypy tests/typing   # type inference of the public API
uv run pre-commit install            # git hooks
uv run mkdocs serve                  # this site, at http://127.0.0.1:8000
uv run mkdocs build --strict         # build it; any warning fails
```

The tests are in `tests/`: `unit/`, `session/` (`BaseSession` and `SessionFactory` on a fake transport), `integration/` (aiohttp against a local server and raw socket servers), `providers/` (every provider against shared scenarios and the upstream schemas), `typing/` (static checks of the public API) and `docs/` (the examples, and `unification.md` against the shared types). Shared doubles and fixtures are in `tests/conftest.py`. Coverage must stay at 98 % or more, for lines and branches. CI runs the same commands, with pytest on Ubuntu and Windows, and builds this site; a push to `master` publishes it to GitHub Pages.

## Examples are tested

Every ```` ```python ```` block of the README and of these pages is run by `tests/docs/test_examples.py` against a local server. A block is found by the title on its opening line, such as ```` ```python title="query.py" ````, so it can move between pages. A new example needs a `title` and a test that runs it; a block without a title, two blocks with one title, or a title no test runs fails `test_every_example_is_tested`. Output and shell snippets use ```` ```text ```` and ```` ```bash ````, which are not run.

The code uses Google-style docstrings, which ruff enforces, and PEP 695 generics. Every module starts with `from __future__ import annotations`. pyright is configured in `pyproject.toml` too; the few rules it disables are explained there.
