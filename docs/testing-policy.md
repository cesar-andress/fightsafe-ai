# FightSafe AI

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:**
Camilo José Cela University (UCJC)
Madrid, Spain

---

# Testing policy (FightSafe AI)

This document defines **strict** expectations for automated tests. It applies to all new and changed code in the repository. For contributor workflow, see [contributing.md](contributing.md) and the [CI workflow](../.github/workflows/ci.yml).

---

## Non-negotiables

1. **Every feature must include tests** — A feature is not complete without tests that exercise its public behavior. Bug fixes should add a **regression** test when feasible.
2. **Pure functions must have unit tests** — Deterministic inputs and exact or tight assertions on outputs, edge cases, and invariants.
3. **CLI commands must have integration tests** where **technically possible** without violating the rules below (no network, no large files). Subprocess or `Typer`’s `invoke` on a **thin** CLI path is preferred; if a path cannot be tested without `ffmpeg` or other heavy runtimes, document the skip or use **stubs** (as in `tests.support.isolated`) and test the **parser** and **orchestration** in isolation.
4. **No internet access in tests** — No live HTTP(S) calls, no model downloads, no `yt-dlp` to real hosts. **Mock** or **stub** external services; use local files only if they are **tiny** fixtures committed on purpose.
5. **No large video files** — Do not require multi-megabyte or long clips. Use **synthetic** frames or **generated** `numpy` / `pandas` tables.
6. **Synthetic data or tiny fixtures** — Prefer in-memory `DataFrame` / array builders (see `tests/fixtures/synthetic.py`, `tests/fixtures/mvp_runs.py`) or small checked-in samples (bytes or CSV under a few kilobytes, if ever needed).
7. **Deterministic tests** — **No** `time.time()`-based flakiness, no unordered iteration over sets when order matters, no unseeded randomness without an explicit `pytest` fixture with a **fixed** seed. `numpy.random.Generator` with a known seed is acceptable.
8. **Tests must run in CI** — CI runs **unit** and **integration** tests (`tests/unit` and `tests/integration`); e2e under `tests/e2e` is opt-in (``make test-e2e``). Default `pytest` (see `pytest.ini` / tool config) runs **unit and integration**. No “works only on my machine” assumptions.

---

## Coverage targets

| Phase | Line coverage (package `fightsafe_ai`) |
|--------|----------------------------------------|
| **MVP** | **≥ 74%** (CI gate in `pyproject.toml`; raise when the tree consistently exceeds it) |
| **Long-term** | **≥ 90%** (goal for stable releases) |

**Interpretation:** Coverage is **necessary, not sufficient**. A high number with weak assertions is not acceptable. The numbers above are **goals** for the **shipped package**; third-party or UI-only code paths may be listed under `omit` in coverage config with justification.

**CI** enforces `fail_under` from `pyproject.toml` (currently aligned with meaningful coverage on optional-heavy modules).

---

## Design hints

- **Isolation** — Prefer `tests.support.isolated` to load submodules when importing the top-level `fightsafe_ai` package would pull in optional heavy dependencies.
- **Naming** — `test_*_synthetic.py` for data-driven numeric tests; keep **one logical behavior per test** when practical.
- **Policy examples** — `tests/unit/test_policy_synthetic_examples.py` and `tests/fixtures/synthetic.py` illustrate the rules end-to-end for several domains.

---

## See also

- [engineering-standards.md](engineering-standards.md) — tool versions and Ruff / Mypy.
- [commit-convention.md](commit-convention.md) — commit messages (including `test:`).
