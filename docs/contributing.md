# FightSafe AI

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:**
Camilo José Cela University (UCJC)
Madrid, Spain

---

# Contributing to FightSafe AI

Thank you for taking time to improve this project. FightSafe AI is an **open-source, research-oriented** toolkit for studying **safety-related cues** in **combat sports** video (**real-time risk estimation**, **event-level safety alerts**, **decision support**). **FightSafe AI is a research prototype for decision support, not a certified medical or officiating system.** Contributions that strengthen **reproducibility**, **clarity of assumptions**, and **responsible use** are especially welcome.

---

## Project goals

- **Transparent pipelines** — End-to-end stages (video → frames → pose → features → risk) should remain explicit, with versioned configuration and inspectable outputs (CSVs, overlays), not hidden behavior.
- **Research support** — Help analysts and researchers **highlight candidate intervals** for human review, not to replace medical judgment, officiating, or ground-truth labeling.
- **Modular design** — New pose backends, risk heuristics, and visualization should plug in through **small, well-defined interfaces** (`BasePoseEstimator`, rule configs, feature tables) so experiments stay comparable across commits.
- **Iterative validation** — Thresholds and heuristics are expected to be **tuned and validated** on each team’s data; the repository ships defaults for development, not universal truth.

For a high-level data flow, see [architecture.md](architecture.md). For direction of travel, see [roadmap.md](roadmap.md).

The **Information Fusion manuscript** (`../fusion2026/main.tex` in the monorepo layout) must stay aligned with meaningful product and research changes; see **Section 11** in [engineering-standards.md](engineering-standards.md) for the normative rule, examples, and exceptions.

---

## Coding standards

- **Python 3.12** — See [engineering-standards.md](engineering-standards.md) for interpreter and tooling policy.
- **Style** — Run **`ruff check src tests`** (and format if the project enables Ruff format) before opening a PR.
- **Public APIs** — Prefer explicit type hints and short docstrings on **exported** functions and dataclasses; keep module-level disclaimers where outputs are not clinical (see `fightsafe_ai.risk.rules`).
- **Configuration** — Prefer **YAML** and dataclass loaders for tunable values; avoid hard-coding thresholds in multiple places without a single source of truth.
- **Tests** — Add or update tests under `tests/` for new behavior. Prefer **fast, deterministic** unit tests (synthetic tabular data) over heavy end-to-end video runs in CI.
- **Dependencies** — Keep optional heavy stacks (e.g. full pose runtimes) clearly separated; do not require them for every import path if the project uses lazy loading for tests.
- **No secrets** — Do not commit API keys, cookies, or paths to private datasets. Use a local `.env` if needed (and ensure `.env` is gitignored).

### Paper / `main.tex` helper

From the repository root, `make check-paper-update` runs `tools/check_paper_update.py`. It compares the **working tree and the index** to the **`main` branch (or `origin/main` if `main` is missing)**: if any file under `src/fightsafe_ai/` changed and **`../fusion2026/main.tex` did not**, the script **prints a reminder**; the default exit code is **0** (warning only, not a failure).

To make the process exit with status **1** in CI or a hook, pass `--fail` (e.g. `make check-paper-update PAPER_CHECK_FLAGS=--fail` or `python tools/check_paper_update.py --fail`).

The script is intentionally simple: it does not judge whether your change *requires* a paper edit—**you** still decide using **Section 11** in [engineering-standards.md](engineering-standards.md).

---

## Branch naming

Use a **short prefix** and a **kebab-case** description:

| Prefix | Use for |
|--------|---------|
| `feat/` | New user-visible behavior or module |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `test/` | Test-only changes |
| `refactor/` | Internal restructuring without intended behavior change |
| `chore/` | Tooling, CI, packaging |

Examples: `feat/interpretable-risk-export`, `fix/keypoint-csv-ordering`, `docs/contributing-pose-backends`.

---

## Pull request process

1. **One logical change per PR** — Easier to review and bisect. Split large refactors from feature work when possible.
2. **Describe intent** — Title + short summary: *what* changed, *why* it matters, and any **breaking** API or config changes.
3. **Tests** — Call out new tests or update existing ones; if a test is skipped in some environments, document why.
4. **Config & docs** — If you add YAML keys, add comments in `configs/*.yaml` and mention them in the PR.
5. **Paper** — If your change affects **architecture, methods, experiments, evaluation, or design** (see **Section 11** in [engineering-standards.md](engineering-standards.md)), update **`../fusion2026/main.tex`** in the same PR or note a follow-up in the description.
6. **Review** — Maintainers may request smaller follow-ups; keeping the first diff focused speeds merge.

**Before submitting**

```bash
cd fightsafe-ai
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make ci          # ruff (format+lint), mypy, pytest + coverage (unit + integration)
# or step-by-step: ruff format --check src tests && ruff check src tests
#                 && mypy src/fightsafe_ai tests
#                 && python -m pytest tests/unit tests/integration --cov=fightsafe_ai
pre-commit run --all-files
```

### Review checklist (for authors and reviewers)

- [ ] Public functions/modules have docstrings where behavior is non-obvious.
- [ ] New configuration keys are documented in YAML or in code.
- [ ] Risk-related changes restate **non-clinical** scope where appropriate.
- [ ] Updated `../fusion2026/main.tex` if this change affects architecture, methods, experiments, evaluation, or design decisions. (If not applicable, state why, e.g. typo-only or test-only with no conceptual impact; see **Section 11** in [engineering-standards.md](engineering-standards.md).)
- [ ] No identifying media or credentials in the diff or PR description.

---

## How to add a new pose estimator

The pipeline expects a **single consolidated keypoint table** that downstream code can load with the helpers in `fightsafe_ai.keypoints.io`. The default implementation is **MediaPipe** (`fightsafe_ai.pose.backends.mediapipe_backend`, also exported from `fightsafe_ai.pose`).

1. **Subclass** `fightsafe_ai.pose.base.BasePoseEstimator` and implement:
   - `estimate_frame(self, image: np.ndarray) -> PoseResult` — BGR `HxWxC` input; return `PoseResult` with `Keypoint` entries (use **lower_snake_case** names when possible for consistency with feature code).
   - `estimate_folder(self, input_dir: Path, output_csv: Path) -> Path` — Write **one** CSV with at least: `frame_id`, `keypoint_name`, `x`, `y`, `z`, `visibility` (match the column names expected by existing loaders; see `MediaPipePoseEstimator` for a reference schema).
2. **Export** your class from `fightsafe_ai.pose` (update `pose/__init__.py` and `__all__` as needed).
3. **Wire configuration** — Extend `configs/default.yaml` under a `pose:` block (e.g. `provider: your_backend`) and document new keys. The CLI currently calls `MediaPipePoseEstimator` directly in `estimate-pose` and `run-pipeline`; add a small **factory** or `if provider ==` branch in `fightsafe_ai/cli.py` so users can select your backend without duplicating pipeline logic.
4. **Tests** — Prefer unit tests that feed **synthetic arrays** or tiny fixtures, and assert CSV shape/column names, without requiring large model weights in CI if avoidable.

---

## How to add a new risk rule

The codebase has two related layers; extend the one that matches your use case (or both, if you need parity).

### A. Interpretable rules (recommended for transparency)

Used by `fightsafe_ai.risk.rules`, `fightsafe_ai.risk.scorer.compute_interpretable_risk`, and the **`interpretable_*` sections** of `configs/risk_rules.yaml`.

1. Add a **stable string name** in `fightsafe_ai.risk.rules` (e.g. extend `ALL_RULE_NAMES` and add a `RULE_*` constant).
2. Add a **frozen dataclass** for parameters (see `FastDownwardConfig`, `LargeTorsoConfig`, …).
3. Implement a **`component_*` function** that maps feature arrays to **values in [0, 1]** per frame.
4. Update **`InterpretableRiskConfig`**, **`load_interpretable_risk_config`**, and **`build_rule_components`** to wire columns → component → weight.
5. Extend **`InterpretableAggregationConfig`** with a new **weight** field; update **`_weight_map`** in `fightsafe_ai.risk.scorer` and any YAML load path.
6. Add **YAML** under `configs/risk_rules.yaml` and **document** new keys. Add **unit tests** (synthetic `DataFrame` rows) for monotonicity, missing columns, and weight renormalization where relevant.

### B. Legacy engine aggregation (`detect_risk_events`)

Used by `fightsafe_ai.risk.engine.detect_risk_events` and the **top-level** blocks in `configs/risk_rules.yaml` (`tilt_velocity`, `ground_contact`, `erratic_motion`, `aggregation`).

- Extend **`RiskRuleParams`** in `fightsafe_ai.risk.models` and **`risk_rules_from_yaml`** to parse new fields.
- Update **`detect_risk_events`** to consume any **new required feature columns** (document them in the function docstring) and add corresponding tests.

**General guidance:** Keep rules **interpretable** (document sign conventions, e.g. image-coordinate “down” for hip velocity), avoid implying clinical meaning in names or docstrings, and prefer YAML tuning over scattered magic numbers.

---

## How to add a new model

In this repository, **“model”** usually means one of the following; pick the path that matches your work.

### 1. Configuration / parameter model (most common)

- **Risk rules** — Extend `RiskRuleParams` and/or `InterpretableRiskConfig` and load from `configs/risk_rules.yaml` (see [How to add a new risk rule](#how-to-add-a-new-risk-rule)).
- **Defaults** — If you add a new named preset, document it next to existing YAML and in `docs/` if it changes user-facing behavior.

### 2. Learned or external scoring model (ML)

- **Contract** — The natural integration surface is a **per-frame `DataFrame`** already produced by `fightsafe_ai.features` (or compatible columns). Your model should read that table and **append** columns (e.g. `learned_risk_score`) without mutating raw keypoint CSVs in place.
- **Packaging** — Add optional dependencies (e.g. in `pyproject.toml` extras) so core installs stay lightweight. Document **training data expectations**, **input column names**, and **versioning** of weights in the PR.
- **Evaluation** — If you report metrics, separate **in-distribution** validation from claims about real-world fight safety; see [Ethical considerations](#ethical-considerations).

### 3. New pose / vision backbone

- Treated as a [new pose estimator](#how-to-add-a-new-pose-estimator); the “model” is the underlying checkpoint or API, exposed only through `BasePoseEstimator` and the agreed CSV schema.

---

## How to report issues

- Use the project **issue tracker** (GitHub/GitLab, as applicable) and choose labels if available (`bug`, `docs`, `enhancement`).
- **Title** — Concise, specific (e.g. “CLI `run-pipeline` fails when `pose.csv` is empty” not “It broke”).
- **Reproduction** — Minimal steps: OS, Python version, **commit or version** of FightSafe AI, and exact command. If the bug is data-dependent, describe **synthetic** or **anonymized** inputs; do **not** attach identifiable videos or personal data.
- **Expected vs actual** — One short paragraph each.
- **Logs** — Redact paths and secrets. If relevant, attach a **small** CSV snippet with fake values.
- **Scope** — Separate **feature requests** from **bugs**; for research ideas, link a paper or spec so maintainers can gauge fit with [Project goals](#project-goals).

---

## Ethical considerations

Combat-sports analytics touches **athlete welfare**, **privacy**, and **public perception of risk**. Contributors are expected to:

- **No clinical claims** — This software is **not** a medical device, diagnostic tool, or substitute for qualified medical and coaching judgment. Do not market outputs as detecting concussion, injury, or fitness for contact without rigorous, domain-appropriate validation.
- **Privacy** — Do not share identifying video, names, or metadata in issues, PRs, or public datasets without explicit consent and legal review. Prefer synthetic or de-identified examples in documentation and tests.
- **Human in the loop** — Automated scores are **aids to review**; they should not be the sole basis for sanctions, match stoppage, or medical decisions in real events unless your organization has a validated, governed process.
- **Bias and context** — Pose and heuristics vary with camera angle, lighting, clothing, and rule set. Document limitations; avoid overgeneralizing from single-dataset experiments.
- **Research integrity** — When reporting results, separate **engineering performance** (e.g. pipeline runtime) from **safety impact** (which requires appropriate study design and often institutional oversight).

If you are unsure whether a change could be misread as a health claim, err on the side of **clear labeling** in code, CLI output, and documentation.

---

## License and conduct

By contributing, you agree that your contributions are licensed under the same terms as the project (see the repository `LICENSE`). Follow the community **Code of Conduct** if one is published; be constructive and specific in code review.

Thank you again for helping make FightSafe AI more useful and responsible for research and education.
