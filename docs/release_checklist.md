# Release checklist — GitHub v0.1.0 and Zenodo archive

Use this checklist for the **first public software release** of FightSafe AI. Complete the steps in order. All documentation updates must stay in **English**.

Companion LaTeX manuscripts (`../fusion2026`, `../sinica2026`, `../sports`) cite the shared software entry `fightsafe_ai_2026` in `/home/cesar/papers/bibliography.bib`.

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
- `README.md`, `CITATION.cff`, `LICENSE`, and `docs/REPRODUCIBILITY.md` render correctly
- No secrets, private paths, or large media files are tracked (see [`security-and-data-policy.md`](security-and-data-policy.md))

---

## 2. Enable Zenodo for the GitHub repository

1. Sign in to [Zenodo](https://zenodo.org/) with your GitHub account.
2. Open **Account → GitHub** and grant Zenodo access to the `cesar-andress/fightsafe-ai` repository.
3. Toggle **ON** for that repository so Zenodo can archive new GitHub releases.

Zenodo will create a new deposition automatically when you publish a GitHub release (step 3).

---

## 3. Create GitHub release v0.1.0

On GitHub: **Releases → Draft a new release**

| Field | Value |
|-------|--------|
| Tag | `v0.1.0` |
| Target | `main` |
| Title | `v0.1.0 — Initial public release` |
| Description | Summarize scope: decision-support software, reproducibility scripts, curated reference data; link companion manuscripts; note large videos are external (see `data/README.md`). |

Optional local tag (if you prefer the CLI):

```bash
git tag -a v0.1.0 -m "Initial public release for Zenodo archiving."
git push origin v0.1.0
```

Then publish the release on GitHub so Zenodo picks it up.

---

## 4. Get the Zenodo DOI

1. Wait for Zenodo to finish processing the GitHub release (usually minutes).
2. Open the new Zenodo record linked from the GitHub release or your Zenodo profile.
3. Copy the **concept DOI** (recommended for software that will receive follow-up releases), e.g. `10.5281/zenodo.1234567`.
4. Copy the **version-specific DOI** for v0.1.0 if you need to pin an exact archive snapshot.

Record both DOIs in your lab notes. The concept DOI is the stable identifier for citations across versions.

---

## 5. Update CITATION.cff, README.md, and the shared BibTeX entry

Replace every `10.5281/zenodo.XXXXXXX` placeholder with the assigned DOI.

### Software repository (`fightsafe-ai/`)

| File | What to update |
|------|----------------|
| [`CITATION.cff`](../CITATION.cff) | Top-level `doi` and `preferred-citation.doi` |
| [`README.md`](../README.md) | Zenodo URL in the resource table and BibTeX block under [Citation](../README.md#citation) |
| [`.zenodo.json`](../.zenodo.json) | `related_identifiers` placeholder (and ORCIDs if real values are available) |

### Shared bibliography (monorepo parent)

Edit `/home/cesar/papers/bibliography.bib` — entry key **`fightsafe_ai_2026`** (do not rename; all papers use this key):

```bibtex
@misc{fightsafe_ai_2026,
  author       = {Mart{\'i}n Moncunill, David and S{\'a}nchez, C{\'e}sar Andr{\'e}s},
  title        = {FightSafe {AI}: Decision-Support Software for Combat-Sports Safety Monitoring},
  year         = {2026},
  version      = {0.1.0},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.<ASSIGNED>},
  url          = {https://doi.org/10.5281/zenodo.<ASSIGNED>},
  note         = {Open-source research software; GitHub release v0.1.0}
}
```

### Papers that cite `fightsafe_ai_2026`

No citation-key changes are required if you only replace the DOI in `bibliography.bib`:

| Manuscript | File |
|------------|------|
| Information Fusion | `../fusion2026/declarations.tex` |
| JAS / TapKO | `../sinica2026/main.tex` |
| FightSafe-Bench | `../sports/main.tex` |

All use `\bibliography{../../bibliography}`.

---

## 6. Commit DOI updates

In the software repository:

```bash
cd /path/to/fightsafe-ai
# edit CITATION.cff, README.md, .zenodo.json
git add CITATION.cff README.md .zenodo.json docs/release_checklist.md
git commit -m "Update Zenodo DOI placeholders after v0.1.0 release."
git push origin main
```

Commit the shared bibliography separately in the monorepo if it is version-controlled at `/home/cesar/papers/`.

Optionally add a **post-release git tag** (e.g. `v0.1.0-doi`) only if your workflow requires a commit after Zenodo assignment; the GitHub release tag `v0.1.0` should remain the archived snapshot.

---

## 7. Recompile all papers

From each manuscript directory (or use the reproduction Makefile targets):

```bash
# fusion2026 (Elsevier elsarticle)
cd ../fusion2026
make -C ../fightsafe-ai fusion-pdf
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main

# sinica2026 (IEEEtran)
cd ../sinica2026
latexmk -pdf -interaction=nonstopmode main.tex

# sports (IEEEtran)
cd ../sports
latexmk -pdf -interaction=nonstopmode main.tex
```

Verify:

- No undefined citations for `fightsafe_ai_2026`
- Bibliography lists the Zenodo DOI (not `XXXXXXX`)
- PDFs build: `fusion2026/main.pdf`, `sinica2026/main.pdf`, `sports/main.pdf`

Smoke-test reproducibility against the released tag:

```bash
cd ../fightsafe-ai
git checkout v0.1.0   # optional: pin to release
REPRO_USE_REFERENCE=1 make reproduce-all
make verify-repro
```

---

## Pre-release validation (before step 1)

Run once before the first push:

```bash
cffconvert --validate -i CITATION.cff
REPRO_USE_REFERENCE=1 bash scripts/reproduce_all.sh
make verify-repro
```

`CITATION.cff` must pass schema validation. Placeholder DOIs are acceptable **before** Zenodo deposit; they must be replaced in step 5.

---

## Quick reference

| Artifact | Citation key / identifier |
|----------|---------------------------|
| Software (Zenodo + GitHub) | `fightsafe_ai_2026` |
| GitHub release tag | `v0.1.0` |
| CFF version field | `0.1.0` |
| Shared BibTeX file | `/home/cesar/papers/bibliography.bib` |
