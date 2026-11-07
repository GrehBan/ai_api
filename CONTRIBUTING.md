# Contributing

Thank you for helping with aiollm. The library is pre-alpha: the public API can still change, and every change is welcome, from a typo to a new provider. For anything larger than a fix, open an issue first and describe the change, so that we agree on the design before you spend time on code.

The guide below walks through one change from start to finish. The rules it refers to are written down in [`CLAUDE.md`](https://github.com/GrehBan/aiollm/blob/master/CLAUDE.md), the project's profile: commands, architecture invariants and conventions. It is written for AI assistants, and it is the most complete description of the code base for people too.

## Setting up

You need Python 3.13 or newer and git. Fork the repository on GitHub, clone your fork and create a branch for your change:

```bash
git clone https://github.com/<you>/aiollm.git
cd aiollm
git switch -c my-change
```

Install the package with all its development dependencies (dev tools, tests and docs).

Using traditional approach (pip 25.1 or newer, which installs dependency groups):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e . --group dev --group test --group docs
```

Or with [uv](https://docs.astral.sh/uv/), which the project uses and whose `uv.lock` pins every version:

```bash
uv sync --all-groups
```

Then install the git hooks once. They run the formatter, the linter and both type checkers on every commit:

```bash
uv run pre-commit install          # or: pre-commit install
```

## Project layout

```text
src/aiollm/
├── enums.py            Provider, BodyFormat, ResponseFormat, Role,
│                       IncompleteReason, GenerationEventType
├── exceptions.py       AIAPIError (the root of all library errors),
│                       RequestValidationError, SessionMismatchWarning
├── types/              BaseAPIType, files, matchers, BaseUnknown; requests/ and
│                       responses/ hold the shared GenerationRequest/Response, events
├── serialization/      JSON codec, body value normalization, value formatters
├── http/               IAPI and BaseAPI, BearerAuth, APIKeyAuth, headers
│   └── exceptions/     status, transport and response errors, resolver
├── methods/            BaseAPIMethod, BaseStreamMethod, field checks, path templates
├── response/           BufferedResponse, Stream, transform_stream, map_stream;
│                       parse/ parsers
├── session/            BaseSession, AiohttpSession, SessionFactory;
│                       retry/ RetryPolicy, body/ builders
├── client/             BaseClient, BaseProviderClient
├── adapters/           adapters between shared types and provider methods
├── resources/          GenerationResource, use cases and AdapterUseCase orchestrators
└── providers/
    ├── anthropic/      AnthropicClient and the Messages API: methods/, types/, adapters/
    └── openai/         OpenAIClient and the Responses API: methods/, types/, adapters/
```

The packages are layers, listed from the bottom up; a module imports only from the layers below it. Transport code lives only in `*/aiohttp.py` modules, so `import aiollm` does not import aiohttp. The tests mirror this: `tests/unit/`, `tests/session/`, `tests/integration/` (aiohttp against a local server), `tests/providers/` (every provider against shared scenarios and the upstream schemas), `tests/typing/` (static checks of the public API) and `tests/docs/` (the examples of these pages).

## Making changes in code

At this point you can make any changes in the code that you want: fixes, new features or experiments. A few rules keep the code base consistent, and the checks below enforce most of them:

- **Every module, class and function has a Google-style docstring.** Ruff checks it. A docstring is user documentation: the [reference](https://grehban.github.io/aiollm/reference/) is generated from it.
- **Modern Python, the oldest supported version.** Every module starts with `from __future__ import annotations`; use PEP 695 generics (`class Stream[T]`) and nothing newer than Python 3.13 allows.
- **Public names are exported on purpose.** A name users need goes into its package's `__init__.py` with a sorted `__all__`; everything else stays internal.
- **Fail early.** A mistake in a method class raises `TypeError` when the class is defined, a bad value raises before anything is sent, and nothing is dropped silently.
- **Generated code is regenerated, not edited.** A provider's `types/<family>.py` and `methods/params/` come from [aiollm-codegen](https://github.com/GrehBan/aiollm_codegen) through `uv run python scripts/generate_provider_types.py`. The generator (the `codegen` dependency group) and OpenAI's OpenAPI file (a URL and a SHA-256 in the script) are pinned, and CI fails when a regeneration changes any file; fix a defect in aiollm-codegen and move the pin.
- **Breaking changes are allowed before 1.0, but marked.** Say so in the changelog fragment (see [Describe changes](#describe-changes)); see [Versioning](https://grehban.github.io/aiollm/guide/design/#versioning) for what counts as one.

For a new endpoint or a new provider, read [Extending](https://grehban.github.io/aiollm/guide/extending/): provider methods follow the provider's own JSON schema field by field, and the wire-contract tests check that they do.

## Format the code (code-style)

This project uses [Ruff](https://docs.astral.sh/ruff/) for formatting and linting (line length 79), so you should follow its code style. To be sure you are doing it correctly, reformat the code and fix what the linter can fix automatically.

Using traditional approach:

```bash
ruff format src tests
ruff check --fix src tests
```

Or with uv:

```bash
uv run ruff format src tests
uv run ruff check --fix src tests
```

The code is also type-checked in strict mode by two checkers, which must both pass.

Using traditional approach:

```bash
mypy src
pyright
```

Or with uv:

```bash
uv run mypy src
uv run pyright
```

Do not silence a checker to get green: a `# type: ignore`, `# noqa` or `# pyright: ignore` needs a reason in a comment next to it, and the reviewer will ask for it.

## Run tests

All changes should be tested. A new behaviour gets a test of what a caller sees (the result, the error, what was sent); a bug fix starts with a test that fails without the fix.

Using traditional approach:

```bash
pytest tests
pytest --cov                        # with branch coverage
pytest --mypy -m mypy tests/typing  # static checks of the public API
```

Or with uv:

```bash
uv run pytest tests
uv run pytest --cov
uv run pytest --mypy -m mypy tests/typing
```

What the tests expect from a change:

- **Coverage stays at 100 %** of lines and branches in practice (the configured floor is 98 %).
- **No real API.** Network tests run against the local `Recorder` server from `tests/conftest.py`, which answers only what a test scripted and fails the test on any other request, so a hidden retry cannot pass. API keys from your environment are cleared for every test.
- **No sleeps.** Wait for an event or a condition, not for time to pass.
- **Every supported Python.** CI runs the tests on CPython 3.13, 3.14 and 3.15, on Linux and Windows. To try another version locally with uv, run `UV_PYTHON=3.15 uv run --all-groups pytest` (it rebuilds `.venv` for that version; run `uv sync --all-groups` to go back).

## Docs

We are using [MkDocs](https://www.mkdocs.org/) with the Material theme to render the docs; all sources are in the `docs` directory, and the reference pages are generated from the docstrings. You can change the sources, start the live-preview server and look at what you are doing at <http://127.0.0.1:8000>.

Using traditional approach:

```bash
mkdocs serve
```

Or with uv:

```bash
uv run mkdocs serve
```

Before you push, build the site in strict mode, where any warning (a broken link, a missing page) fails, and lint the Markdown:

```bash
uv run mkdocs build --strict
uv run rumdl check README.md CHANGELOG.md CONTRIBUTING.md docs
```

Set `NO_MKDOCS_2_WARNING=1` to hide Material's notice about its end of life.

Every ```` ```python ```` block of the README and of the docs is run by `tests/docs/test_examples.py`. A block is found by the title on its opening line, such as ```` ```python title="query.py" ````, so a new example needs a unique `title` and a test that runs it; a block without one fails `test_every_example_is_tested`. Output and shell snippets use ```` ```text ```` and ```` ```bash ````, which are not run.

When the text and the code disagree, the code says what the library does: fix the text. If the code contradicts something the docs promise on purpose, that is a bug; open an issue.

## Describe changes

Describe your changes in one or more sentences so that the developers who use the library know what changed. Create a changelog fragment in the `CHANGES` directory and write the description in it; [towncrier](https://towncrier.readthedocs.io/) turns the fragments into the [changelog](https://grehban.github.io/aiollm/changelog/) when a version is released.

Using traditional approach:

```bash
towncrier create +my-change.fixed.md --content "Text of the change."
```

Or with uv:

```bash
uv run towncrier create +my-change.fixed.md --content "Text of the change."
```

- **Name.** `<PR number>.<type>.md` once your pull request exists, or `+<slug>.<type>.md` before it does.
- **Type.** One of `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`.
- **Text.** One Markdown paragraph about what a user sees, not about file names. A breaking change starts with `**Breaking:**` and says how to migrate.
- **Do not** bump the version or edit `CHANGELOG.md`: a release is a separate pull request made by the maintainer, so pull requests made side by side never conflict in the changelog.
- A change no user would notice (tests, CI, internal refactoring) needs no fragment; ask for the `no changelog` label instead.

Preview what the next release holds with `uv run towncrier build --draft`. [`CHANGES/README.md`](https://github.com/GrehBan/aiollm/blob/master/CHANGES/README.md) has the details.

## Complete

After you have made all your changes, commit them with a message that says what changed and why, push the branch to your fork and open a pull request against `master`. In its description, say what the change does and how you checked it.

CI runs on every pull request, and a pull request is merged only when all of these pass:

| Check | What it runs |
|---|---|
| `lint and types` | ruff, mypy, pyright, the typing tests, the Markdown lint and the strict docs build |
| `tests` | pytest with coverage on Linux and Windows, CPython 3.13, 3.14 and 3.15 |
| `package` | builds the wheel and imports it in a clean environment |
| `audit` | known vulnerabilities in the locked dependencies (`uv audit`) |
| `changelog` | a changelog fragment, unless the PR has the `no changelog` label |
| GitGuardian | no secrets in the diff |

Then wait for a review of these changes. Releases, version numbers and tags are made by the maintainer.

## Contributing with Claude

Many changes to this library are written with [Claude Code](https://claude.com/claude-code), and the repository is set up for it. If you use Claude (or another coding agent), these directives keep its work at the level of the rest of the code base. You are the author of the pull request and answer for every line of it, whoever typed it.

### What the repository gives Claude

- **[`CLAUDE.md`](https://github.com/GrehBan/aiollm/blob/master/CLAUDE.md)** is loaded into every session: the commands, the invariants that are easy to break and the conventions. If your change alters an invariant or a convention, update `CLAUDE.md` in the same pull request; a stale profile misleads the next session.
- **Project skills** in `.claude/skills/` carry the project's workflows, and Claude picks them up when a task matches; you can also call one by name, such as `/ai-api-endpoint`. They are listed in the table below.
- **Provider documentation.** The maintainer's sessions read the providers' schemas from a private documentation server (`unified-api-docs` in a local `.mcp.json`). Without it, point Claude at the providers' official API references and SDK types; refreshing the schema snapshot in `tests/providers/contracts/` is left to the maintainer.

| Skill | Use it for |
|---|---|
| `ai-api-endpoint` | a new endpoint or field of an existing provider, from the upstream schema to the wire contracts |
| `ai-api-provider` | a new provider package |
| `ai-api-docs` | guide pages, docstrings and tested examples |
| `ai-api-release` | the changelog fragment of a change (and, for the maintainer, releases) |

### Directives

1. **Discuss before code.** For anything beyond a small fix, ask Claude for a proposal first (what changes, the options, the trade-offs, the tests) and read it before you let it write code. Open the issue with that proposal.
2. **Facts come from sources, not from memory.** What a provider accepts or returns is checked against its schema or documentation, and the source is named. Claude must not guess a field, a status or a default.
3. **Test first for every bug.** Ask Claude to write the failing test before the fix and to show that it fails on the unfixed code.
4. **Results are run, not predicted.** Ask for the actual output of the full set of checks above. "It should pass" is not a result; neither is a check that was skipped without saying so.
5. **Nothing is weakened to get green.** No new `skip`, `xfail`, `type: ignore`, `noqa` or relaxed setting without a reason in the code and in the pull request; no lowered coverage.
6. **Small, focused diffs.** One concern per pull request, no drive-by refactoring or formatting of unrelated files, and no new abstraction without a second use for it.
7. **You decide what leaves your machine.** Review every diff yourself. Claude does not push, open or merge pull requests, publish or delete anything unless you tell it to, one action at a time.
8. **No secrets in the conversation.** Never paste a real API key into the session; the tests use fakes and a local server, and the library must never log a key, a header, a body, a query string or a file name.
9. **Your name on the commit.** Commit messages and pull requests carry no model names or AI co-author trailers. If you want to say that a pull request was made with Claude, say it in its description.
10. **One task per session.** Start a fresh session for a new task and give it the issue; a long session that drifted across topics makes worse decisions.

If you use a tool that condenses command output for the agent, exclude `pytest` from it, so that the coverage report stays visible.

## License

By contributing, you agree that your contributions are licensed under the MIT License of the project.
