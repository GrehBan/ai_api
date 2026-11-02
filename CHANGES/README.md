# Changelog fragments

Every pull request that changes what a user of the library sees adds a
fragment here; `towncrier build` turns the fragments into a section of
`CHANGELOG.md` when a version is released, and deletes them.

Name a fragment `<PR number>.<type>.md`, or `+<slug>.<type>.md` before
the PR exists (`uv run towncrier create +<slug>.<type>.md`). The types
are Keep a Changelog's: `added`, `changed`, `deprecated`, `removed`,
`fixed`, `security`.

Write what a user sees, not file names, in one paragraph of Markdown.
Start a breaking change with `**Breaking:**` and say how to migrate.
Preview the next section with `uv run towncrier build --draft`.
