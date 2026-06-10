# FightSafe AI

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:**
Camilo José Cela University (UCJC)
Madrid, Spain

---

# Commit message convention (Conventional Commits)

FightSafe AI uses **[Conventional Commits 1.0.0](https://www.conventionalcommits.org/)** (with project-specific types). The goal is **browsable history**, safe **changelogs**, and **revertability**.

---

## Format

```text
<type>[(scope)][!]: <description>

[optional body]

[optional footers]
```

- **Type** (required) — one of the allowed values below, **lowercase** only.
- **Scope** (optional) — a short noun for the area, e.g. `risk`, `cli`, `llm`, `config`. **Lowercase**, kebab-case or dotted if you need a path (`video.cutter`).
- **Breaking** — append **`!` after the type/scope** *or* add a footer; see *Breaking changes*.
- **Description** (required) — **imperative mood** (“add”, “fix”, “remove”, not “added” or “fixes”). No trailing period.
- **Body** (optional) — *why* or extra context, wrapped at ~72 characters per line for readability.
- **Footers** (optional) — e.g. `BREAKING CHANGE:`, `Refs: #123` (one line per footer, blank line before footers after body).

**First line length:** keep the full header (type + scope + `:` + subject) to **at most 72 characters** when possible.

---

## Allowed types

| Type | Use for |
|------|--------|
| `feat` | New user-facing behavior or public API |
| `fix` | Bug fix (corrects unintended behavior) |
| `docs` | Documentation and comments in docs/ only (no code behavior change) |
| `style` | Formatting, white space, or style that does *not* change meaning (e.g. Ruff format-only) |
| `refactor` | Internal restructuring without changing intended external behavior |
| `test` | Tests only (new tests, rewrites, fixtures) |
| `chore` | Maintenance, deps bumps, non-user tooling (excluding CI-only) |
| `ci` | CI, workflows, automation that runs in GitHub Actions or similar |
| `perf` | Performance improvement (same behavior, better speed or memory) |
| `build` | Build system, packaging, release, `pyproject`, lockfiles, Docker |
| `revert` | Reverts a previous commit; body should reference the SHA and reason |

**Do not invent new types**; use the closest of the list or ask maintainers to extend the policy in this file.

---

## Rules (strict)

1. **One logical change per commit** — Do not mix unrelated refactors, features, and typo fixes. Split with `git rebase -i` or multiple commits.
2. **Lowercase type** — e.g. `feat`, not `Feat` or `FEAT`.
3. **Imperative mood** in the subject — “add buffer limit”, not “added” / “adds” / “adding.”
4. **Subject at most 72 characters** (header line). If you need more detail, use the **body**.
5. **No vague subjects** — Avoid `update`, `changes`, `fix stuff`, `misc`, `tweaks`, or single-word dumps that do not name *what* changed. Prefer concrete nouns/verbs.
6. **Breaking changes** must be visible to users of the API or config: use **`!`** in the header and/or a **`BREAKING CHANGE:`** footer with migration notes (see below).

---

## Breaking changes

- **Header (preferred for discoverability):** `feat(api)!: remove batch endpoint`
- **Footer (required when the migration needs more than one line):**

```text
BREAKING CHANGE: batch scoring is removed. Use the streaming CLI until batch returns in v0.2.
```

Use **BREAKING CHANGE** (or **BREAKING-CHANGE**) as the token; describe what broke and what to do instead.

---

## Examples — good

```text
feat(risk): add event deduplication window
```

```text
fix(cli): require explicit output when input is a directory
```

```text
docs: expand dataset disclaimer for consent-limited use
```

```text
test(biomechanics): add synthetic case for high jerk spike
```

```text
chore: bump ruff to 0.8.x
```

```text
ci: run commitlint on pull requests
```

```text
revert: revert "feat(overlay): add glow filter"

Reverts 1a2b3c4d5e6f7 because glow breaks headless frame export in CI.
```

```text
feat(rules)!: rename yaml key risk.window_ms to event.window_ms

BREAKING CHANGE: configs must use `event.window_ms` under the risk section.
```

---

## Examples — bad (and what to do instead)

| Bad | Why | Better |
|-----|-----|--------|
| `Update stuff` | No type; vague | `chore: clean unused imports in risk engine` |
| `fix: things` | Vague subject | `fix(timecode): parse hh:mm:ss.ff for fractional parts` |
| `feat: Added parser` | Past tense; “Added” in subject | `feat(keypoints): add COCO json parser` |
| `chore: update` | Vague | `chore: refresh pandas-stubs in dev extras` |
| `Misc changes` | No type; useless | split into one commit per logical change, each with a proper `type: subject` |
| `Style` | Not allowed style (type must be from list) | `style(overlay): align help text in argparse` (or `refactor` if behavior clarifies) |

---

## Local setup and checks

**CI** runs [commitlint](https://github.com/conventional-changelog/commitlint) on every **pull request** to `main`, using the same config as this repo. If any commit in the range fails, the **Commitlint** check is red and the PR should not be merged until messages are fixed (e.g. with `git rebase -i` and `git commit --amend`).

### Prerequisites

- **Node.js 18+** and **npm** (see `engines` in `package.json`).

### One-time install (repository root)

```bash
cd path/to/fightsafe-ai
npm ci
```

This installs `@commitlint/cli` and `@commitlint/config-conventional` from `package-lock.json`. Use **`npm ci`**, not `npm install`, so versions match CI.

### Verify before you push

**Last commit only** (typical after `git commit`):

```bash
npx commitlint --last --verbose
```

**All commits on your branch** not in `main` (after `git fetch origin`):

```bash
npx commitlint --from origin/main --to HEAD --verbose
```

If your default branch is not `main`, replace `origin/main` with the correct base (e.g. `origin/master`).

**Single message from stdin** (no commit required):

```bash
echo "feat(risk): add event deduplication window" | npx commitlint --verbose
```

Config file: **`commitlint.config.js`** (extends `@commitlint/config-conventional` and adds project rules). Commitlint auto-discovers it; you can pass `--config commitlint.config.js` explicitly if needed.

### What is enforced

CI and local `commitlint` use the same rules: **structural** Conventional Commits (type, case, header length, allowed types, break footers from the shared preset, etc.). They do **not** catch vague but syntactically valid English—write clear, imperative subjects yourself.

---

## See also

- [Contributing](contributing.md) — branch names and PR process.
- [Conventional Commits](https://www.conventionalcommits.org/) — full specification.
