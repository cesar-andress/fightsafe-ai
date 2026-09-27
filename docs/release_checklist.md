# Release checklist — GitHub releases and Zenodo archive

**Current release:** `v2.0.0` (Zenodo version DOI [`10.5281/zenodo.23003746`](https://doi.org/10.5281/zenodo.23003746); concept / all-versions [`10.5281/zenodo.20622868`](https://doi.org/10.5281/zenodo.20622868)).

The **GitHub repository root** is the canonical FightSafe AI research-software artefact. There is no nested Zenodo staging directory.

Use this checklist for public software releases. All documentation updates must stay in **English**.

Companion LaTeX manuscripts outside this repository (`../paper1`, `../iswa2026`, `../sports`; archived workspaces under `../legacy/`) may cite the shared software entry. Manuscripts are not shipped inside this GitHub software repository.

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

After a new GitHub Release is published, Zenodo assigns a **version-specific** DOI under the concept record `10.5281/zenodo.20622868`. For the published `v2.0.0` archive the version DOI is `10.5281/zenodo.23003746`. Keep `.zenodo.json` journal-neutral; do not invent a future version DOI before Zenodo creates it.

---

## 3. Create or update the GitHub release

| Field | Value |
|-------|--------|
| Tag | `v2.0.0` |
| Target | final commit on `main` |
| Title | `FightSafe AI v2.0.0` |
| Description | Repository root is the canonical artefact; link Zenodo concept DOI; point to root `README.md` for Tier A; note restricted-data exclusions |

```bash
git tag -a v2.0.0 -m "FightSafe AI v2.0.0"
git push origin v2.0.0
gh release create v2.0.0 --title "FightSafe AI v2.0.0" --notes-file -
```

---

## 4. Metadata files (must stay consistent)

| File | Required fields |
|------|-----------------|
| [`CITATION.cff`](../CITATION.cff) | `version: 2.0.0`, concept DOI if version DOI unknown |
| [`README.md`](../README.md) | Version table, citation block, Zenodo URL |
| [`.zenodo.json`](../.zenodo.json) | `version`, creators, licence, related identifiers, description of the **repository itself** |
| [`pyproject.toml`](../pyproject.toml) / [`src/fightsafe_ai/__version__.py`](../src/fightsafe_ai/__version__.py) | `2.0.0` |
| [`CHANGELOG.md`](../CHANGELOG.md) | `[2.0.0]` entry |

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

Recompile external companion papers if they cite this software entry, and update bibliography DOIs after Zenodo publishes the version-specific record for `v2.0.0`.

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
| Software (Zenodo v2.0.0) | version DOI `10.5281/zenodo.23003746` |
| Software (Zenodo concept / all versions) | concept DOI `10.5281/zenodo.20622868` |
| GitHub release tag | `v2.0.0` |
| CFF / package version | `2.0.0` |
| Canonical scientific run | `canonical_results/run_20260730_005150/` |
| Companion manuscript (optional) | sibling `../paper1/` |
