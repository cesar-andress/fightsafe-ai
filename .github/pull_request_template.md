<!--
Delete sections that are not applicable. Keep the checklist; check items you completed.
PR title must use Conventional Commits (e.g. feat: …, fix: …). See docs/commit-convention.md.
-->

## Summary

<!-- One short paragraph: what this PR does and the intended outcome. -->



## Motivation

<!-- Why this change is needed: problem, gap, or goal. Link issues if any (e.g. Closes #123). -->



## Changes

<!-- Bullet list of notable changes. Call out public API, configs, and breaking behavior. -->

-



## Testing

<!-- How you tested: commands, data used (synthetic vs real), environments. -->



## Screenshots or demo output

<!-- If UI, overlays, or CLI output changed: paste or describe. N/A for docs-only or internal refactors. -->

- [ ] N/A (no user-visible or visual output in this PR)

or:

<!-- (attach images / paste text as needed) -->



## Risks

<!-- Assumptions, follow-ups, or anything reviewers should double-check. -->

-



## Checklist

Before requesting review, confirm the following (check every box that applies).

- [ ] **Language** — Code, comments, and user-facing strings in this change are in **English**.
- [ ] **Tests** — Tests were **added or updated** to cover the change (or a short justification in *Testing* above if not applicable).
- [ ] **Local tests** — All tests **pass locally** (`pytest` in the same way CI runs them; see *Testing* for scope).
- [ ] **Ruff** — `ruff check` and `ruff format` (or format check) pass on touched paths.
- [ ] **Mypy** — `mypy` passes for the project configuration used in CI.
- [ ] **No secrets** — No API keys, tokens, cookies, private paths, or other secrets were committed.
- [ ] **No heavy / generated content** — No **datasets**, **videos**, or **generated** artifacts (build outputs, large binaries, `htmlcov/`, local exports) were committed; only what belongs in the repo.
- [ ] **Documentation** — `README` / `docs/`, config comments, or docstrings were **updated** if the change affects how others use or maintain the code; or N/A is stated under *Summary* or *Changes*.
- [ ] Updated `../fusion2026/main.tex` if this change affects architecture, methods, experiments, evaluation, or design decisions. (If not applicable, say so in the PR; see `docs/engineering-standards.md` Section 11 and `docs/contributing.md`.)
- [ ] **Modularity** — The change is **modular and maintainable** (clear boundaries, not an unmaintainable one-off); noted under *Risks* if trade-offs exist.
- [ ] **PR title** — The **PR title** follows [Conventional Commits](https://www.conventionalcommits.org/) and [docs/commit-convention.md](../docs/commit-convention.md) (e.g. `feat(scope): short imperative description`).
