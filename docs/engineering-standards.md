# FightSafe AI

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:**
Camilo José Cela University (UCJC)
Madrid, Spain

---

# Engineering standards — FightSafe AI

**Repository:** [github.com/cesar-andress/fightsafe-ai](https://github.com/cesar-andress/fightsafe-ai)

This document is **normative** for all contributors, maintainers, and automated tooling. It exists to keep the codebase **predictable, safe, and reproducible** in a research and open-source context. If a rule here conflicts with an informal habit, **this document wins** until the team explicitly revises it.

**Language policy:** All **code, comments, documentation, commit messages, branch names, issue and PR titles, and configuration** in this repository are written in **English** only.

For contribution workflow and ethical context, see [contributing.md](contributing.md).

---

## 1. Python version

- **Target interpreter:** **Python 3.12** only. Local development, CI, and production environments must use 3.12.x.
- **Packaging:** `requires-python` in `pyproject.toml` must remain aligned with this standard (e.g. `>=3.12,<3.13` unless the project explicitly broadens support in a reviewed change).
- **Features:** You may use language features available in 3.12. Do not rely on pre-release or unreleased CPython behavior.
- **Virtual environments:** Document the expected creation of a venv in the project README; pin **dev** tool versions where instability has been observed (e.g. Ruff, pytest).

**Rationale:** A single version reduces “works on my machine” drift and keeps CI and type-checking behavior consistent.

---

## 2. Code style

- **Linter / formatter:** **Ruff** is the default for linting. Run `ruff check src tests` (and the project’s format command, if enabled) before opening a PR.
- **Line length:** Follow `pyproject.toml` / Ruff config (e.g. 100 characters) unless a justified exception is documented in the PR.
- **Imports:** Sorted and grouped (stdlib, third party, first party). No wildcard imports in production code except where a framework requires it; justify in review.
- **Structure:** **Modular** packages: clear boundaries between `video`, `pose`, `features`, `risk`, `llm`, etc. Avoid “god” modules. Prefer small, testable functions over long scripts.
- **Path handling:** Use `pathlib.Path` for filesystem paths. **No hardcoded absolute paths** to developer machines, cloud buckets, or local datasets in committed code. Resolve paths from configuration, environment, or CLI arguments.
- **Dead code:** Remove unused imports, variables, and commented-out blocks; use version control for history.
- **Generated artifacts:** Do not commit `__pycache__`, `.pytest_cache` noise, or build products unless a documented process requires them and `.gitignore` is updated.

---

## 3. Type hints

- **Public APIs** (exported functions, classes, module-level attributes intended for use outside the package) require **complete, accurate type annotations** for parameters and return types.
- Prefer modern syntax: `list[str]`, `dict[str, Any]` only when necessary, `X | Y` for unions, `Literal[...]` for fixed sets.
- **Avoid** untyped public surfaces “to save time.” Use `typing.Protocol` or `ABC` for pluggable components where appropriate.
- **Third-party** gaps: add minimal types or `typing.cast` with a comment at trust boundaries, not wide silent `Any` without reason.
- **Mypy** (or the project’s chosen checker) should pass on `src/` for changes that touch typed modules; new code must not introduce a class of new suppressions without maintainer agreement.

---

## 4. Docstrings

- **Style:** Use consistent style (project default: **Google- or NumPy-style** docstrings; pick one per subpackage and stick to it).
- **Content:** For public callables, document **purpose, parameters, return value, and raised exceptions** that callers must handle. Include **brief notes** on non-obvious invariants (e.g. “positive hip velocity means downward in image coordinates”).
- **Scope:** Private helpers (`_leading_underscore`) may use a one-line docstring or concise comment if the name and types are self-explanatory; non-clinical and safety disclaimers belong in `risk` and `llm` modules where outputs could be misinterpreted.
- **No** redundant restatement of the function name; **do** state side effects (I/O, global state, logging at warning or above).

---

## 5. Error handling

- **Domain errors:** Raise **specific** exceptions (see `fightsafe_ai.exceptions` and related modules). Subclass a small hierarchy where it improves `except` precision.
- **User-facing tools (CLI):** Catch predictable failures, print a **clear, non-leaky** message to stderr, and exit with a **non-zero** code. Do not dump raw stack traces to users unless `--verbose` or debug mode is specified.
- **Library code:** Do not blanket-catch `Exception` without re-raise or a documented reason. Prefer `try`/`except` around **narrow** expected failures (I/O, parsing, network).
- **Resource handling:** Use context managers for files, HTTP clients, and locks. Never rely on CPython’s refcount for critical cleanup in library code.
- **Never** use errors to control normal program flow (e.g. `StopIteration` misuse, flow via generic `except`).

---

## 6. Logging

- **Library code:** Use `logging.getLogger(__name__)`. **No** `print()` in library paths that ship as a package, except in documented CLI output paths.
- **Levels:** `DEBUG` for detailed traces, `INFO` for high-level progress, `WARNING` for recoverable issues, `ERROR` for failures that may leave the run inconsistent, `CRITICAL` for process-level aborts (rare).
- **Content:** **Never** log secrets, tokens, cookies, PII, or full paths to private data stores. Redact or hash identifiers if logging is necessary.
- **Configuration:** CLI and apps set handlers and levels; library code does not assume a particular handler is configured.

---

## 7. Testing

- **Framework:** **pytest** is the standard test runner.
- **Location:** Tests live under `tests/`, mirroring the package where practical. **Fast, deterministic** unit tests are mandatory for new logic. Prefer **synthetic** tabular data and stubs over real videos in CI.
- **Coverage:** Aim for **high** coverage of new and changed code. Critical paths (risk aggregation, event merging, config loading) should have direct tests. Gaps need an explicit reason in the PR.
- **Isolation:** Where imports pull heavy optional dependencies, use the project’s **isolated** loading pattern or environment markers so `pytest` collection does not require GPU, large model downloads, or private assets.
- **Skips:** Use `@pytest.mark.skip` / `skipif` with a **clear reason** (e.g. “optional dependency X not installed”). No silent skips in CI for required behavior.
- **Regressions:** A bug fix should include a test that fails before the fix unless maintainers agree otherwise.

---

## 8. Configuration

- **Source of truth:** Runtime-tunable values belong in **YAML** (e.g. `configs/`) or **environment variables**, loaded through documented loaders. **No** magic numbers scattered across multiple modules for the same concept.
- **Schema:** Configuration structures should be validated at load time (type checks, required keys, value ranges) and raise **`ConfigurationError`** (or the project’s equivalent) on invalid input.
- **Paths:** All paths must be **configurable**. Defaults may point to **project-relative** locations (resolved via `Path(__file__)`, package data, or documented cwd assumptions), not hardcoded `C:\` or `/home/...` paths.
- **Secrets:** Never commit credentials, API keys, or cookies. Use environment variables and `.env` (gitignored) for local only; document variable names in README, not values.
- **LLM / Ollama:** `configs/llm.yaml` and related flags control optional narrative features; they **must not** replace the rule-based or vision pipeline as the source of truth for risk.

---

## 9. Git commits

- **Language:** English commit messages.
- **Format:** Imperative, concise subject line (e.g. `Add risk event explanation CLI`). **Optional** body for **why** and **breaking** changes.
- **Atomicity:** One logical change per commit where possible. Do not mix refactors and unrelated features in a single commit.
- **Prohibited in commits:** Secrets, large binaries, full datasets, raw videos, or notebook cell outputs. Use [gitignore](.gitignore) and pre-commit checks if configured.

---

## 10. Pull requests

- **Title and description (English):** State **what** changed, **why**, and how to **verify** (tests run, example command). Link related issues.
- **Scope:** One main concern per PR. **Breaking changes** and migration steps must be called out in the description.
- **Review bar:** PRs are reviewed for correctness, **tests**, **typing**, **docs/config**, and alignment with this document.
- **Drafts:** Use GitHub **Draft** until CI passes and the author considers it ready.
- **Merge:** Prefer **linear history** or the maintainers’ chosen strategy (documented in repo settings). No merge commits that obscure intent unless the team standard says otherwise.
- **Living research document** — **Section 11** below; see also the *Review checklist* in [contributing.md](contributing.md).

---

## 11. Living research document (fusion2026/main.tex)

The **Information Fusion manuscript** (`../fusion2026/main.tex` in the monorepo layout) is the **authoritative, version-controlled research narrative** for the fusion line of FightSafe AI (motivation, architecture, evaluation posture, and design decisions). It must stay **aligned** with the codebase and experiments.

- **When an update is required** — Any **meaningful** architectural, methodological, experimental, **evaluation**, or **design** change should be reflected in `../fusion2026/main.tex` in the same PR (or a tightly coupled follow-up that lands before the next release), so readers can cite one coherent document.

- **Examples that require paper updates (non-exhaustive):**
  - New **framework** module (pipeline stage, new package boundary, or new pluggable component class).
  - New **risk rule** or change to how risk is aggregated, fused, or configured.
  - New **referee alert** or **risk level** (or change to their semantics or mapping).
  - New **dataset** or **data-format** support in the public pipeline.
  - Any substantive change to **annotation** (schema, file format, CLI), **ground truth** policy, or **human labeling** workflow.
  - Any change to **evaluation** methodology: **event matching** rules, **metrics** definitions, tolerances, or how `fightsafe evaluate` (or related modules) compares predictions to reference.
  - New **experiment** protocol, ablation, or **evaluation metric** documented in code or `docs/`.
  - A **new limitation** discovered, or a **design decision** recorded in code and configs that is not already described in the paper.
  - Changed **Ollama** (or other LLM) **explainability** behavior that affects when text is generated or what it is allowed to say.

- **Examples that do *not* require paper updates (non-exhaustive):**
  - **Typo** or copy fixes with no change to meaning.
  - **Formatting-only** edits (e.g. Ruff, whitespace) with no behavior change.
  - **Internal refactors** with **no** conceptual or user-visible impact.
  - **Minor test-only** changes (assertions, fixtures) that do not document new research or evaluation policy.

- **Build:** Contributors may use `make fusion-pdf` from the repository root; CI builds the PDF in a separate workflow when the manuscript directory is present. See [contributing.md](contributing.md) for the PR checklist item on `../fusion2026/main.tex`.

**Rationale:** The paper is the single place where **research intent, limitations, and design trade-offs** stay traceable alongside `git` history, without duplicating the whole of `docs/` in prose.

---

## 12. CI/CD

- **CI must run** on every PR and default branch push: ``make ci`` (Ruff format + check, Mypy on `src/fightsafe_ai` and `tests`, pytest with coverage). Failures block merge unless a documented exception.
- **Python 3.12** must be the version used in CI.
- **Duration:** Keep the default pipeline **fast** (minutes, not hours). Long jobs are optional, scheduled, or behind labels.
- **Artifacts:** **Do not** store large model weights, videos, or datasets in CI artifacts long-term. Cache dependencies sensibly; do not cache secrets.
- **Releases:** Tagging and PyPI (if used) are documented; release notes in English.

---

## 13. Security

- **Dependabot** (or equivalent) should be enabled for the repository where practical.
- **Dependencies:** Pin or constrain versions for reproducibility; review license compatibility for new dependencies.
- **Code execution:** Avoid `eval` / `exec` on untrusted input. For YAML, use **safe** loaders. For subprocess calls, **never** pass unsanitized user strings to the shell.
- **Supply chain:** Verify checksums for downloaded tooling when the project documents bootstrap scripts.
- **Vulnerability response:** Security-sensitive reports should go through a **private** channel if GitHub **Security Advisories** or a maintainer email is published.

---

## 14. Data handling

- **Repository contents:** **No** large datasets, raw match videos, or high-resolution image dumps in git. Use external storage, DVC, or a documented download script with a stable manifest.
- **PII and athletes:** **Do not** commit identifying media, names, or metadata. Examples in `examples/` use **synthetic** or **anonymized** data only.
- **Notebooks:** **No** heavy outputs (large embedded base64, huge tables) in committed notebooks. Clear outputs before commit unless the team explicitly allows a small curated notebook with documented artifacts.
- **Outputs:** `outputs/`, `dist/`, and local run directories are gitignored. Generated reports are documented in README or docs, not stored in the tree by default.
- **Ethics:** Combat-sports data can be sensitive. Follow institutional review and data-use agreements; the codebase does not substitute for them.

---

## 15. Research reproducibility

- **Versioning:** The **version** in `__version__` / `pyproject.toml` and **git tag** (when applicable) must allow citing a **specific** snapshot of the software.
- **Config:** Reproduce runs by **commit hash + `configs/*.yaml` + environment** (Python 3.12, dependency versions from lock file or `pip freeze` in docs if required).
- **Seeds:** Where randomness is used, **document** default seeding and make seeds settable for replication.
- **Pipelines:** Document the **end-to-end** command sequence (e.g. CLI) or notebook entry point for a “reference” experiment. Heuristic thresholds in YAML should be version-controlled with a short **changelog** in PRs when they change.
- **Claims:** Reports and papers should separate **engineered metrics** (latency, code coverage) from **domain claims** (safety, injury), which require appropriate study design outside this document.

---

## Compliance and updates

- **Contributors** are expected to self-check against this file before requesting review. **Reviewers** may require changes for non-compliance.
- **Amendments** to this document are made via the normal PR process, with a clear description of the rule change and migration impact.

**Last updated:** Engineering standards for FightSafe AI — [github.com/cesar-andress/fightsafe-ai](https://github.com/cesar-andress/fightsafe-ai).
