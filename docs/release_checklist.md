# Release checklist — GitHub releases and Zenodo archive

**Current release:** `v1.0.0` (Zenodo DOI [`10.5281/zenodo.21698326`](https://doi.org/10.5281/zenodo.21698326)).

The **GitHub repository root** is the canonical FightSafe AI v1.0.0 research artefact. There is no nested Zenodo staging directory.

Use this checklist for public software releases. All documentation updates must stay in **English**.

Companion LaTeX manuscripts outside this repository (`../fusion2026`, `../iswa2026`, `../sports`, `../paper1`) may cite the shared software entry. The EAAI manuscript lives in sibling `../paper1/` (not inside this GitHub software repository).

---

## 1. Push to GitHub

From the software repository root:

```bash
cd /path/to/fightsafe-ai
git status
git log --oneline -5
git push origin main
```

Confirm on GitHub:

- Default branch is `main`
- `README.md`, `CITATION.cff`, `LICENSE`, `NOTICE_DATA.md`, and `docs/REPRODUCIBILITY.md` render correctly
- No secrets, private paths, restricted video/skeleton data, or `features_cache` binaries are tracked

---

## 2. Zenodo GitHub integration

1. Sign in to [Zenodo](https://zenodo.org/) with your GitHub account.
2. Open **Account → GitHub** and grant Zenodo access to `cesar-andress/fightsafe-ai`.
3. Ensure the repository toggle is **ON** so Zenodo can archive GitHub releases.

For v1.0.0 the version DOI is already assigned: `10.5281/zenodo.21698326`. Keep `.zenodo.json` and `CITATION.cff` aligned with that DOI.

---

## 3. Create or update the GitHub release

| Field | Value |
|-------|--------|
| Tag | `v1.0.0` |
| Target | final commit on `main` |
| Title | `FightSafe v1.0.0 — EAAI Reproducibility Artefact` |
| Description | Repository root is the canonical artefact; link Zenodo DOI; point to root `README.md` for Tier A; note restricted-data exclusions |

```bash
git tag -a v1.0.0 -m "FightSafe v1.0.0 — canonical EAAI reproducibility artefact"
git push origin v1.0.0
gh release create v1.0.0 --title "FightSafe v1.0.0 — EAAI Reproducibility Artefact" --notes-file -
```

---

## 4. Metadata files (must stay consistent)

| File | Required fields |
|------|-----------------|
| [`CITATION.cff`](../CITATION.cff) | `version: 1.0.0`, `doi: "10.5281/zenodo.21698326"` |
| [`README.md`](../README.md) | Version badge/table, citation block, Zenodo URL |
| [`.zenodo.json`](../.zenodo.json) | `version`, creators, licence, related identifiers, description of the **repository itself** |
| [`pyproject.toml`](../pyproject.toml) / [`src/fightsafe_ai/__version__.py`](../src/fightsafe_ai/__version__.py) | `1.0.0` |
| [`CHANGELOG.md`](../CHANGELOG.md) | `[1.0.0]` entry |

Do **not** commit placeholder DOIs (`10.5281/zenodo.PENDING` / `XXXXXXX`) or placeholder ORCIDs.

---

## 5. Pre-release validation

```bash
cffconvert --validate -i CITATION.cff   # if available
python3.12 scripts/verify_checksums.py
python3.12 scripts/validate_tier_a.py
python3.12 -m pytest tests/unit/test_aggregation_schemes.py -q
```

Expected: Tier A `overall: PASS`; checksums match; no nested `release/` tree.

---

## 6. Companion manuscripts (optional monorepo)

Recompile external companion papers if they cite this software entry, and ensure bibliography DOIs match `10.5281/zenodo.21698326` / version `1.0.0`.

EAAI manuscript (sibling `../paper1/`):

```bash
python3.12 scripts/generate_eaai_assets.py
cd ../paper1 && latexmk -pdf -interaction=nonstopmode main.tex
# bibliography: bibtex main   (not bibtex main.aux)
```

---

## Zenodo notes

- If both `.zenodo.json` and `CITATION.cff` exist, Zenodo uses **only** `.zenodo.json` for GitHub-triggered archiving.
- Use `"license": "mit"` (lowercase) in `.zenodo.json`.
- Describe the **repository root**, not a nested ZIP or staging folder, as the canonical project.

---

## Quick reference

| Artifact | Identifier |
|----------|------------|
| Software (Zenodo + GitHub) | `fightsafe_ai_2026` / DOI `10.5281/zenodo.21698326` |
| GitHub release tag | `v1.0.0` |
| CFF / package version | `1.0.0` |
| Canonical scientific run | `canonical_results/run_20260730_005150/` |
| EAAI manuscript | sibling `../paper1/` |
