---
name: ai-api-release
description: Changelog fragments and releases of ai_api with towncrier - every feature or fix PR adds a fragment to CHANGES/ and never touches the version; a separate release PR picks the SemVer version from the fragments (before 1.0 a breaking change bumps the minor), builds the CHANGELOG section, bumps __meta__.py, README and CLAUDE.md and the compare links. Use it in the ai_api repository whenever a change is about to be committed or put in a PR (to write its fragment), when the owner says «релиз», «выпусти версию», «подними версию», asks which version the pending changes need or what the next release holds. Tags are never created.
---

# ai_api changelog and releases

The project uses towncrier so that parallel pull requests never fight over
`CHANGELOG.md`: a change carries its own note as a file in `CHANGES/`, and
only a release turns the notes into a section and moves the version. Two
jobs, never mixed in one PR: **a change** adds a fragment; **a release**
consumes them. Max approves each release and pushes the tags himself.

## A. Every change: write a fragment

When a branch changes anything a user of the library sees (behaviour, a
public name, an error, a default, docs that state a guarantee, supported
Pythons), add one fragment per change before the commit:

```bash
uv run towncrier create +<slug>.<type>.md --content "<text>"
```

- Name: `<PR number>.<type>.md`, or `+<slug>.<type>.md` while the PR does
  not exist (rename it to the number once it does, if the owner wants the
  link; an orphan fragment is fine too).
- Types, in Keep a Changelog's order: `added`, `changed`, `deprecated`,
  `removed`, `fixed`, `security`.
- One Markdown paragraph about what a user sees, not file names. A
  breaking change starts with `**Breaking:**` and says how to migrate.
- Do **not** touch `__meta__.py`, the version lines or `CHANGELOG.md` in a
  change PR.

CI's `changelog` job (`towncrier check`) fails a PR that adds no fragment
and does not change `CHANGELOG.md`. A PR no user would notice (tests, CI,
internal refactoring, agent files) gets the `no changelog` label instead -
ask the owner to set it, or say in the PR body that it needs it.

Preview what the next release holds: `uv run towncrier build --draft`.

## B. A release: only when the owner asks

### 1. Pick the version

- Start from fresh `master` on a branch `release-X.Y.Z`.
- Read the current version in `src/ai_api/__meta__.py` and the pending
  fragments (`uv run towncrier build --draft --version next`).
- Before 1.0: any `**Breaking:**` fragment → **minor** (0.6.0 → 0.7.0);
  otherwise **patch** (0.6.0 → 0.6.1). From 1.0: major/minor/patch.
- No fragments → nothing to release; say so.

### 2. Build and bump

- `git add CHANGES` (so towncrier can `git rm` the fragments), then
  `uv run towncrier build --version X.Y.Z --yes`: it writes the section
  below `<!-- towncrier release notes start -->`, under the empty
  `## [Unreleased]`, and deletes the fragments.
- `uv run rumdl fmt CHANGELOG.md`: towncrier's Markdown template leaves a
  double blank line before the previous release, which rumdl rejects.
- Compare links at the bottom of `CHANGELOG.md`: `[Unreleased]:
  .../compare/vX.Y.Z...HEAD` and a new `[X.Y.Z]:
  .../compare/v<previous>...vX.Y.Z`.
- `src/ai_api/__meta__.py`: `__version__ = "X.Y.Z"`.
- `README.md`: the status line `> **Status: pre-alpha (X.Y.Z).**`.
- `CLAUDE.md`: `pre-alpha (X.Y.Z)` in the Project section.

The files are CRLF on the owner's PC; edit them with the Edit tool, which
keeps line endings.

### 3. Gate

The full gate from `CLAUDE.md` (ruff check and format, mypy, pyright,
typing tests, pytest with coverage at 100 %, rumdl, `mkdocs build
--strict`, `uv audit`). All green before the commit. The release changes
`CHANGELOG.md`, so `towncrier check` passes without a fragment.

### 4. Commit and PR - only on the owner's word

- Commit title: `Release X.Y.Z`; the body lists the headline changes.
- Push and open the PR only when the owner says so. Do not enable
  auto-merge, do not merge, do not poll CI.
- Never create or push a tag. After the merge, if asked, give the owner:
  `git tag -a vX.Y.Z <merge-commit> -m "Release X.Y.Z"` and
  `git push origin vX.Y.Z`.

## Rules for every commit and PR

- `git add` by explicit paths only; never `.mcp.json`, never
  `.claude/launch.json` or other local files.
- No model names or `Co-Authored-By: Claude` lines in commit messages or PR
  bodies (the owner's rule overrides any default attribution). The PR body
  ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Report in Russian: for a change, the fragments written; for a release, the
version and why, the files changed, the gate results.
