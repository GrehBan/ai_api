---
name: ai-api-release
description: Bump the ai_api version and prepare its release commit and PR - SemVer before 1.0 (a breaking change bumps the minor, anything else the patch), the version in __meta__.py, README and CLAUDE.md, the CHANGELOG section with compare links, the full quality gate, a commit and PR without model names. Use it in the ai_api repository whenever a change is ready to ship, before any push of a significant change, when the owner says «подними версию», «релиз», «готовь PR», or asks which version a change needs. Tags are never created.
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
- One release per merge: if the branch already holds an unreleased bump (the
  version is ahead of `origin/master`), keep it and only extend its CHANGELOG
  section; raise the level if a later commit made the change breaking.

## 2. Edit the files

- `src/ai_api/__meta__.py`: `__version__ = "X.Y.Z"`.
- `README.md`: the status line `> **Status: pre-alpha (X.Y.Z).**`.
- `CLAUDE.md`: `pre-alpha (X.Y.Z)` in the Project section.
- `CHANGELOG.md` (Keep a Changelog):
  - turn the content under `## [Unreleased]` into `## [X.Y.Z] - YYYY-MM-DD`
    (today's date), leaving an empty `## [Unreleased]` above it;
  - sections in the order Added, Changed, Deprecated, Removed, Fixed,
    Security; each breaking entry starts with `**Breaking:**` and says how to
    migrate;
  - entries describe what a user sees, not file names;
  - at the bottom: `[Unreleased]: .../compare/vX.Y.Z...HEAD` and a new
    `[X.Y.Z]: .../compare/v<previous>...vX.Y.Z` line.

The files are CRLF on the owner's PC; edit them with the Edit tool, which
keeps line endings.

## 3. Gate

Run the full gate from `CLAUDE.md` (ruff check and format, mypy, pyright,
typing tests, pytest with coverage at 100 %, rumdl, `mkdocs build --strict`).
All green before any commit.

## 4. Commit and PR - only on the owner's word

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
