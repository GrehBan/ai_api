---
name: ai-api-release
description: Bump the ai_api version and prepare its release commit and PR - SemVer before 1.0 (a breaking change bumps the minor, anything else the patch), the version in __meta__.py, README and CLAUDE.md, the CHANGELOG section built by towncrier from the fragments in CHANGES/ with compare links, the full quality gate, a commit and PR without model names. Use it in the ai_api repository whenever a change is ready to ship, before any push of a significant change, when the owner says «подними версию», «релиз», «готовь PR», or asks which version a change needs. Tags are never created.
---

# ai_api release

Max ships every significant merge as a version and leaves tags to himself.
The version lives in several files that must agree, and the CHANGELOG is what
users read, so a release is a small, careful procedure.

## 1. Pick the version

Read `src/ai_api/__meta__.py` for the current version.

- Before 1.0: a **breaking** change (public signature, behaviour, wire format,
  error type, import path, a removed name) bumps the **minor**: 0.5.0 → 0.6.0.
  Anything else bumps the **patch**: 0.5.0 → 0.5.1.
- From 1.0: the usual major/minor/patch.
- The fragments in `CHANGES/` decide it: any `**Breaking:**` fragment means
  a minor bump before 1.0. Preview with `uv run towncrier build --draft
  --version X.Y.Z`.

## 2. Changelog fragments

Every user-visible change has a fragment in `CHANGES/` (see
`CHANGES/README.md`): `<PR>.<type>.md`, or `+<slug>.<type>.md` before the
PR exists; types `added`, `changed`, `deprecated`, `removed`, `fixed`,
`security`; one Markdown paragraph about what a user sees; a breaking one
starts with `**Breaking:**` and says how to migrate. CI's `changelog` job
fails a PR that adds no fragment and does not change `CHANGELOG.md`; the
`no changelog` label skips it for changes no user sees.

## 3. Edit the files

- `src/ai_api/__meta__.py`: `__version__ = "X.Y.Z"`.
- `README.md`: the status line `> **Status: pre-alpha (X.Y.Z).**`.
- `CLAUDE.md`: `pre-alpha (X.Y.Z)` in the Project section.
- `CHANGELOG.md`: `uv run towncrier build --version X.Y.Z --yes` writes
  the section below `<!-- towncrier release notes start -->` (under the
  empty `## [Unreleased]`) and removes the fragments; stage the fragments
  with `git add CHANGES` first, so it can `git rm` them. Then add the
  compare links at the bottom by hand: `[Unreleased]:
  .../compare/vX.Y.Z...HEAD` and a new `[X.Y.Z]:
  .../compare/v<previous>...vX.Y.Z` line. Then run
  `uv run rumdl fmt CHANGELOG.md`: towncrier's Markdown template leaves a
  double blank line before the previous release, which rumdl rejects.

The files are CRLF on the owner's PC; edit them with the Edit tool, which
keeps line endings.

## 4. Gate

Run the full gate from `CLAUDE.md` (ruff check and format, mypy, pyright,
typing tests, pytest with coverage at 100 %, rumdl, `mkdocs build --strict`).
All green before any commit.

## 5. Commit and PR - only on the owner's word

- Branch from fresh `master` if not on a feature branch already.
- `git add` by explicit paths only; never `.mcp.json`, never
  `.claude/launch.json` or other local files.
- No model names or `Co-Authored-By: Claude` lines in commit messages or PR
  bodies (the owner's rule overrides any default attribution). The PR body
  ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Commit title: `Release X.Y.Z: <what>`.
- Push and open the PR only when the owner says so («пуш», «PR»). Do not enable
  auto-merge, do not merge, do not poll CI.
- Never create or push a tag. After the merge, if asked, give the owner the
  command: `git tag -a vX.Y.Z <merge-commit> -m "Release X.Y.Z"`.

Report in Russian: the version and why, the files changed, the gate results.
