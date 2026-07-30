#!/usr/bin/env python3.12
"""Build curated EAAI Zenodo staging package (Tier A + Tier B placeholders).

Does not publish, tag, push, or modify canonical numerical results.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SOFT = WORKSPACE / "fightsafe-ai"
PAPER = WORKSPACE / "paper1"
CANON = WORKSPACE / "legacy" / "eaai_checkpoint_2026" / "run_20260730_005150"
STAGING = WORKSPACE / "release" / "eaai_zenodo_staging"
RUN_ID = "run_20260730_005150"
PAPER_TITLE = (
    "Engineering an Interpretable Temporal Event Pipeline with Explicit Channel Availability: "
    "A Combat-Sports Case Study"
)
AGG_HASH = "4d19f0f69238c9afdf365b6baeae26986a2d7086fb2b24170f5b2ce5b6209281"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, ignore=None) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def relativize_text(text: str) -> str:
    """Replace lab absolute roots with package-relative placeholders."""
    reps = [
        (
            str(WORKSPACE / "legacy/paper1_pre_rewrite_20260729_234033/runs/aggregation/features_cache"),
            "optional_tier_b/inputs/features_cache",
        ),
        (
            str(WORKSPACE / "legacy/fightsafe-ai/outputs/evaluation/baselines/full_fusion"),
            "optional_tier_b/inputs/strike_baselines",
        ),
        (str(SOFT / "data/boxingvi/annotations"), "annotations/boxingvi"),
        (str(SOFT / "data/boxingvi"), "annotations/boxingvi_root"),
        (str(SOFT / "configs/risk_fusion.yaml"), "configs/risk_fusion.yaml"),
        (str(SOFT / "configs/risk_rules.yaml"), "configs/risk_rules.yaml"),
        (str(SOFT / "src/fightsafe_ai/evaluation/aggregation_schemes.py"), "src/fightsafe_ai/evaluation/aggregation_schemes.py"),
        (str(SOFT), "."),
        (str(CANON), f"results/{RUN_ID}"),
        (str(WORKSPACE), "<WORKSPACE_EXCLUDED>"),
        ("/home/cesar/papers/fightsafe-ai", "<PACKAGE_ROOT>"),
        ("/home/cesar/", "<LOCAL_HOME_EXCLUDED>/"),
    ]
    out = text
    for a, b in reps:
        out = out.replace(a, b)
    return out


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_tree() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    # --- Core docs / metadata ---
    write(STAGING / "LICENSE", (SOFT / "LICENSE").read_text(encoding="utf-8"))

    write(
        STAGING / "CITATION.cff",
        f"""cff-version: 1.2.0
message: "If you use this reproducibility package, please cite it as follows."
title: "FightSafe AI — EAAI reproducibility package (availability-aware temporal event pipeline)"
type: software
version: 0.2.0-eaai-rc1
date-released: 2026-07-30
license: MIT
repository-code: "https://github.com/cesar-andress/fightsafe-ai"
url: "https://github.com/cesar-andress/fightsafe-ai"
abstract: >-
  Paper-specific reproducibility artefact for the EAAI manuscript
  "{PAPER_TITLE}".
  Provides Tier A verification and regeneration of reported tables and figures
  from frozen canonical result files (run_20260730_005150), configs, aggregation
  implementations, environment metadata and checksums. Not a safety-certified
  system; not a state-of-the-art detector claim; not a medical device. Synthetic
  missingness only; natural channel availability is identity on the held-fixed
  matrices.
authors:
  - family-names: Andrés
    given-names: César
    email: cesar.andress@ucjc.edu
    orcid: "https://orcid.org/0009-0001-8968-3404"
    affiliation: "CRIA-BDHS Research Group, Higher Polytechnic School of Technology and Science, Universidad Camilo José Cela, Madrid, Spain"
  - family-names: Martin Moncunill
    given-names: David
    email: david.martinm@ucjc.edu
    affiliation: "CRIA-BDHS Research Group, Higher Polytechnic School of Technology and Science, Universidad Camilo José Cela, Madrid, Spain"
keywords:
  - interpretable AI
  - temporal event detection
  - channel availability
  - engineering applications
  - reproducibility
  - combat sports
  - decision support
preferred-citation:
  type: software
  title: "FightSafe AI — EAAI reproducibility package (availability-aware temporal event pipeline)"
  authors:
    - family-names: Andrés
      given-names: César
    - family-names: Martin Moncunill
      given-names: David
  year: 2026
  version: 0.2.0-eaai-rc1
  """,
    )

    write(
        STAGING / "zenodo.json",
        json.dumps(
            {
                "title": "FightSafe AI — EAAI reproducibility package: availability-aware interpretable temporal event pipeline (BoxingVI case study)",
                "description": (
                    f"Paper-specific reproducibility package for the EAAI manuscript “{PAPER_TITLE}”. "
                    "Tier A regenerates and verifies reported tables/figures from frozen canonical CSVs "
                    f"({RUN_ID}). Includes configs, aggregation schemes with explicit availability masking, "
                    "environment freeze and checksums. Not a safety-certified system; not a state-of-the-art "
                    "detector claim; not a medical device. Synthetic missingness only."
                ),
                "upload_type": "software",
                "access_right": "open",
                "license": "mit",
                "version": "0.2.0-eaai-rc1",
                "language": "eng",
                "creators": [
                    {
                        "name": "Andrés, César",
                        "orcid": "0009-0001-8968-3404",
                        "affiliation": "CRIA-BDHS Research Group, Higher Polytechnic School of Technology and Science, Universidad Camilo José Cela, Madrid, Spain",
                    },
                    {
                        "name": "Martin Moncunill, David",
                        "affiliation": "CRIA-BDHS Research Group, Higher Polytechnic School of Technology and Science, Universidad Camilo José Cela, Madrid, Spain",
                    },
                ],
                "keywords": [
                    "interpretable AI",
                    "temporal event detection",
                    "channel availability",
                    "engineering applications",
                    "reproducibility",
                    "combat sports",
                    "decision support",
                ],
                "related_identifiers": [
                    {
                        "identifier": "10.5281/zenodo.20622869",
                        "relation": "isNewVersionOf",
                        "resource_type": "software",
                    },
                    {
                        "identifier": "https://arxiv.org/abs/2511.16524",
                        "relation": "references",
                        "resource_type": "publication-preprint",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
    )

    write(
        STAGING / "NOTICE_DATA.md",
        """# Data and licensing notice

## Software
Source code and configs in this package are released under the MIT License (see `LICENSE`).

## BoxingVI punch-interval annotations
Spreadsheets under `annotations/boxingvi/` are research copies of BoxingVI punch-interval labels
(V1–V10) used as an **impact/punch proxy** in the paper. Cite Kumar et al. (BoxingVI; arXiv:2511.16524).
Upstream BoxingVI terms continue to apply.

## Not redistributed
- Raw BoxingVI videos / frames
- BoxingVI skeleton keypoints (`data/boxingvi/skeleton/`)
- Derived `features_cache/*.pkl` (Tier B) — **PENDING INSTITUTIONAL APPROVAL**
- MediaPipe / TensorFlow / OpenCV binaries (install via pip)

## Strike baselines
JSON strike/anomaly baselines under `optional_tier_b/inputs/strike_baselines/` are derived pipeline
outputs from this software. Status: **APPROVED FOR RELEASE** as research artefacts of this work,
but they are useful only together with Tier B feature caches.

## What Tier A can do without restricted data
Regenerate and verify manuscript tables and figures from frozen canonical CSVs under
`results/run_20260730_005150/`.
""",
    )

    # --- Environment ---
    env_dst = STAGING / "environment"
    env_dst.mkdir()
    # Sanitize archival freeze: redact local editable paths (versions preserved where possible).
    freeze_raw = (CANON / "environment" / "pip_freeze.txt").read_text(encoding="utf-8")
    freeze_san = []
    for line in freeze_raw.splitlines():
        if line.startswith("-e ") and ("/home/" in line or "git+ssh://" in line):
            freeze_san.append("# REDACTED_LOCAL_EDITABLE: <local-editable-path>")
        else:
            freeze_san.append(line)
    write(
        env_dst / "pip_freeze.txt",
        "\n".join(freeze_san)
        + "\n# NOTE: local editable installs redacted for release; use requirements-tierA.txt for Tier A.\n",
    )
    copy_file(CANON / "environment" / "repro_check.json", env_dst / "repro_check.json")
    copy_file(CANON / "environment" / "git_state.txt", env_dst / "git_state.txt")
    write(
        env_dst / "pip_freeze.ORIGINAL_NOTE.md",
        "Canonical lab freeze had private editable installs; paths redacted in pip_freeze.txt.\n",
    )

    # Minimal Tier A requirements (no private editable installs)
    write(
        STAGING / "environment" / "requirements-tierA.txt",
        """# Minimal Tier A (Python 3.12)
numpy>=1.26,<3
pandas>=2.0,<3
matplotlib>=3.8,<4
PyYAML>=6.0,<7
pyarrow>=14,<22
pytest>=8,<9
""",
    )
    write(
        STAGING / "environment" / "environment-tierA.yml",
        """name: eaai-tierA
channels:
  - conda-forge
dependencies:
  - python=3.12
  - numpy
  - pandas
  - matplotlib
  - pyyaml
  - pyarrow
  - pytest
  - pip
""",
    )
    write(
        STAGING / "environment" / "PLATFORM_NOTES.md",
        """# Platform notes (Tier A)

- Validated on Linux with Python 3.12.
- Canonical checkpoint `environment/pip_freeze.txt` records the full lab freeze, including
  private editable installs that **cannot** be reproduced externally.
- Use `requirements-tierA.txt` / `environment-tierA.yml` for Tier A figure/table regeneration.
- Exact bit-for-bit recreation of the full lab freeze is **not** claimed.
""",
    )

    # Minimal pyproject for editable install of shipped src
    write(
        STAGING / "pyproject.toml",
        """[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fightsafe-ai-eaai"
version = "0.2.0rc1"
description = "EAAI Tier A reproducibility subset of FightSafe AI"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy>=1.26,<3",
  "pandas>=2.0,<3",
  "matplotlib>=3.8,<4",
  "PyYAML>=6.0,<7",
  "pyarrow>=14,<22",
]

[tool.setuptools.packages.find]
where = ["src"]
""",
    )

    # --- Configs ---
    for cfg in ("risk_fusion.yaml", "risk_rules.yaml"):
        copy_file(SOFT / "configs" / cfg, STAGING / "configs" / cfg)

    # --- Source (full library tree; excludes caches) ---
    copy_tree(
        SOFT / "src",
        STAGING / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    copy_file(
        SOFT / "tests" / "unit" / "test_aggregation_schemes.py",
        STAGING / "tests" / "unit" / "test_aggregation_schemes.py",
    )
    write(STAGING / "tests" / "unit" / "__init__.py", "")
    write(STAGING / "tests" / "__init__.py", "")

    # Verify aggregation hash
    agg = STAGING / "src" / "fightsafe_ai" / "evaluation" / "aggregation_schemes.py"
    got = sha256(agg)
    if got != AGG_HASH:
        raise SystemExit(f"BLOCKED: aggregation_schemes.py hash mismatch: {got} != {AGG_HASH}")

    # --- Annotations (Tier A, with notice) ---
    ann_src = SOFT / "data" / "boxingvi" / "annotations"
    for p in sorted(ann_src.glob("V*.xlsx")):
        copy_file(p, STAGING / "annotations" / "boxingvi" / p.name)
    meta = SOFT / "data" / "boxingvi" / "metadata" / "Meta_data.ods"
    if meta.exists():
        copy_file(meta, STAGING / "annotations" / "boxingvi" / "metadata" / "Meta_data.ods")

    # --- Canonical results (Tier A) ---
    results = STAGING / "results" / RUN_ID
    results.mkdir(parents=True)
    for name in (
        "experiment_summary.csv",
        "per_video_metrics.csv",
        "paired_comparisons.csv",
        "dropout_results.csv",
        "failure_counts.csv",
        "interaction_rule_firings.csv",
        "FINAL_REPORT.md",
    ):
        copy_file(CANON / name, results / name)

    # Relativized PROTOCOL + checksums
    write(results / "PROTOCOL.md", relativize_text((CANON / "PROTOCOL.md").read_text(encoding="utf-8")))
    in_lines = ["path\tsha256\tsize_bytes"]
    for line in (CANON / "input_checksums.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        path, digest, size = parts[0], parts[1], parts[2]
        in_lines.append(f"{relativize_text(path)}\t{digest}\t{size}")
    write(results / "input_checksums.tsv", "\n".join(in_lines) + "\n")
    copy_file(CANON / "output_checksums.tsv", results / "output_checksums.tsv")

    matrices = results / "matrices"
    matrices.mkdir()
    copy_file(CANON / "matrices" / "matrix_meta.csv", matrices / "matrix_meta.csv")
    for pq in sorted((CANON / "matrices").glob("V*.parquet")):
        copy_file(pq, matrices / pq.name)

    inter = results / "interactions"
    inter.mkdir()
    copy_file(CANON / "interactions" / "rule_config_audit.json", inter / "rule_config_audit.json")

    # --- Analysis CSVs needed by generate_assets ---
    for p in sorted((PAPER / "analysis").glob("*.csv")):
        copy_file(p, STAGING / "analysis" / p.name)

    # --- Figures / tables / supplementary ---
    for p in sorted((PAPER / "figures").glob("fig_*.*")):
        if p.suffix.lower() in {".pdf", ".png"}:
            copy_file(p, STAGING / "figures" / p.name)
    for p in sorted((PAPER / "tables").glob("*")):
        if p.suffix in {".tex", ".json"}:
            # scrub absolute comments if any
            if p.suffix == ".tex":
                write(STAGING / "tables" / p.name, relativize_text(p.read_text(encoding="utf-8")))
            else:
                copy_file(p, STAGING / "tables" / p.name)
    for p in sorted((PAPER / "supplementary").glob("tab_si_*.tex")):
        write(STAGING / "supplementary" / p.name, relativize_text(p.read_text(encoding="utf-8")))

    # --- Manuscript support ---
    ms = STAGING / "manuscript"
    for name in ("main.tex", "refs.bib", "Makefile"):
        src = PAPER / name
        if src.exists():
            write(ms / name, relativize_text(src.read_text(encoding="utf-8")))
    # Same relative paths as paper1/: symlink package-root asset dirs into manuscript/
    for link_name in ("figures", "tables", "supplementary"):
        target = ms / link_name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(Path("..") / link_name)
    if (PAPER / "main.pdf").is_file():
        copy_file(PAPER / "main.pdf", ms / "main.pdf")
    write(
        ms / "BUILD.md",
        """# Manuscript build (optional)

From this directory:

```bash
cd manuscript
make
# or: latexmk -pdf -interaction=nonstopmode main.tex
```

`figures/`, `tables/` and `supplementary/` are symlinks to the package-root directories so
`\\includegraphics` / `\\input` paths match the `paper1/` layout.

Requires TeX Live with `elsarticle`, `booktabs`, `tabularx`, `hyperref`, `microtype`.
Auxiliary files are build artefacts and are not part of the deposit.
The appendix is printed after the bibliography.
""",
    )

    # --- Scripts ---
    scripts = STAGING / "scripts"
    scripts.mkdir()
    copy_file(PAPER / "scripts" / "generate_assets.py", scripts / "generate_assets.py")
    copy_file(PAPER / "scripts" / "validate_tier_a.py", scripts / "validate_tier_a.py")

    # Relativized Tier B runner (not executed in Tier A validation)
    runner_src = (CANON / "scripts" / "run_eaai_checkpoint.py").read_text(encoding="utf-8")
    runner = runner_src
    runner = runner.replace(
        'SOFT = Path("/home/cesar/papers/fightsafe-ai/fightsafe-ai")',
        'SOFT = Path(__file__).resolve().parents[1]',
    )
    runner = runner.replace(
        'CACHE_DIR = Path(\n    "/home/cesar/papers/fightsafe-ai/legacy/paper1_pre_rewrite_20260729_234033"\n'
        '    "/runs/aggregation/features_cache"\n)',
        'CACHE_DIR = SOFT / "optional_tier_b" / "inputs" / "features_cache"',
    )
    runner = runner.replace(
        'CACHED_FULL = Path(\n    "/home/cesar/papers/fightsafe-ai/legacy/fightsafe-ai/outputs/evaluation/baselines/full_fusion"\n)',
        'CACHED_FULL = SOFT / "optional_tier_b" / "inputs" / "strike_baselines"',
    )
    runner = runner.replace('ANN_ROOT = SOFT / "data" / "boxingvi"', 'ANN_ROOT = SOFT / "annotations" / "boxingvi_root"')
    # annotations live under annotations/boxingvi/*.xlsx — evaluator expects boxingvi/annotations
    # Keep ANN_ROOT as parent with annotations subdir:
    runner = runner.replace(
        'ANN_ROOT = SOFT / "annotations" / "boxingvi_root"',
        'ANN_ROOT = SOFT / "annotations"  # expects annotations/boxingvi/*.xlsx via data layout helper',
    )
    # Actually original uses ANN_ROOT / annotations - check
    write(scripts / "run_eaai_checkpoint.py", runner)

    write(
        scripts / "verify_checksums.py",
        """#!/usr/bin/env python3.12
\"\"\"Verify SHA-256 checksums for the release package.\"\"\"
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMS = ROOT / "checksums" / "SHA256SUMS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not SUMS.is_file():
        print(f"Missing {SUMS}", file=sys.stderr)
        return 2
    bad = 0
    checked = 0
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        path = ROOT / rel
        if not path.is_file():
            print(f"MISSING {rel}")
            bad += 1
            continue
        got = sha256(path)
        checked += 1
        if got != digest:
            print(f"MISMATCH {rel}")
            bad += 1
    print(f"checked={checked} bad={bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
""",
    )

    # --- Tier B placeholders ---
    tier_b = STAGING / "optional_tier_b"
    write(
        tier_b / "README.md",
        """# Tier B — optional full experiment re-execution

Tier B is **not required** for regenerating paper tables and figures.

| Item | Status |
|------|--------|
| `inputs/features_cache/*.pkl` | PENDING INSTITUTIONAL APPROVAL — not included |
| `inputs/strike_baselines/*.json` | APPROVED FOR RELEASE — included |
| `scripts/run_eaai_checkpoint.py` (package root) | Present (relative paths); requires Tier B inputs |
| BoxingVI annotations | In Tier A (`annotations/boxingvi/`) |
| Skeleton / raw video | NOT REDISTRIBUTABLE |

## How an authorized researcher could obtain features_cache

1. Obtain BoxingVI skeleton keypoints from the dataset authors.
2. Run the FightSafe AI feature extraction pipeline under institutional licence review.
3. Place `V1.pkl`…`V10.pkl` under `optional_tier_b/inputs/features_cache/`.
4. Verify hashes against `results/run_20260730_005150/input_checksums.tsv` when available.

Do not place restricted files into staging without approval.
""",
    )
    write(
        tier_b / "inputs" / "features_cache" / "README.md",
        """# features_cache — NOT INCLUDED

Status: **PENDING INSTITUTIONAL APPROVAL** / derived from non-redistributed skeleton data.

Tier A does not require these files.
""",
    )
    sb = tier_b / "inputs" / "strike_baselines"
    sb.mkdir(parents=True)
    src_sb = (
        WORKSPACE
        / "legacy"
        / "fightsafe-ai"
        / "outputs"
        / "evaluation"
        / "baselines"
        / "full_fusion"
    )
    for p in sorted(src_sb.glob("boxingvi_predictions_V*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        # Scrub lab-only provenance paths; keep event payloads unchanged.
        if "skeleton_path" in data:
            data["skeleton_path"] = f"<NOT_REDISTRIBUTED>/skeleton/{data.get('video_id', p.stem)}.npy"
        if "risk_rules_yaml" in data:
            data["risk_rules_yaml"] = "configs/risk_rules.yaml"
        write(sb / p.name, json.dumps(data, indent=2) + "\n")
    write(
        sb / "STATUS.txt",
        "APPROVED FOR RELEASE — derived strike/anomaly baselines for combined-timeline interpretation.\n"
        "Lab absolute paths in metadata fields were scrubbed; event arrays unchanged.\n",
    )


    # Fix runner ANN path properly: original uses load from ANN_ROOT
    # Check usage
    # We'll create annotations/boxingvi layout matching soft data/boxingvi
    # Soft layout: data/boxingvi/annotations/V*.xlsx and metadata/
    # Runner: ANN_ROOT = SOFT/data/boxingvi then annotations under it
    write(
        STAGING / "annotations" / "boxingvi_root" / "README.md",
        "Compatibility shim: see ../boxingvi/ for spreadsheets.\n",
    )
    # Create expected layout annotations/boxingvi_root/annotations -> copy
    for p in sorted((STAGING / "annotations" / "boxingvi").glob("V*.xlsx")):
        copy_file(p, STAGING / "annotations" / "boxingvi_root" / "annotations" / p.name)
    meta_dst = STAGING / "annotations" / "boxingvi" / "metadata" / "Meta_data.ods"
    if meta_dst.exists():
        copy_file(meta_dst, STAGING / "annotations" / "boxingvi_root" / "metadata" / "Meta_data.ods")

    # Patch runner ANN_ROOT one more time
    runner_path = scripts / "run_eaai_checkpoint.py"
    rtxt = runner_path.read_text(encoding="utf-8")
    rtxt = re.sub(
        r"ANN_ROOT = .*",
        'ANN_ROOT = SOFT / "annotations" / "boxingvi_root"',
        rtxt,
        count=1,
    )
    # OUT_DIR should be local when run — leave as-is if absolute; rewrite OUT if present
    rtxt = relativize_text(rtxt)
    # Ensure no residual /home/cesar in runner
    if "/home/cesar" in rtxt:
        rtxt = rtxt.replace("/home/cesar", "<LOCAL_HOME_EXCLUDED>")
    runner_path.write_text(rtxt, encoding="utf-8")

    # --- README ---
    write(
        STAGING / "README.md",
        f"""# FightSafe AI — EAAI reproducibility package

## 1. Purpose
Tier A artefact to **inspect, verify and regenerate** the tables and figures of the EAAI paper from
frozen canonical results. This is a controlled engineering evaluation package, **not** a
safety-certified system and **not** a state-of-the-art detector claim.

## 2. Paper title
{PAPER_TITLE}

## 3. Canonical run identifier
`{RUN_ID}`

## 4. Tier A vs Tier B
- **Tier A (this package, mandatory):** regenerate/verify figures and tables from frozen CSVs;
  inspect configs, aggregation code, environment metadata and checksums.
- **Tier B (optional):** re-run the experimental layer from `features_cache` + strike baselines.
  Feature pickles are **not included** (pending institutional approval). See `optional_tier_b/`.

## 5. Directory structure
```
README.md, LICENSE, CITATION.cff, zenodo.json, NOTICE_DATA.md, pyproject.toml
environment/          # freeze + Tier A requirements
src/                  # FightSafe AI library (includes aggregation_schemes.py)
configs/              # risk_fusion.yaml, risk_rules.yaml
scripts/              # generate_assets.py, verify_checksums.py, run_eaai_checkpoint.py
results/{RUN_ID}/     # canonical CSVs, PROTOCOL, checksums, parquet matrices
analysis/             # descriptive CSVs used by figure/table generation
figures/, tables/, supplementary/
manuscript/           # main.tex, refs.bib, Makefile
annotations/boxingvi/ # punch-interval labels (with NOTICE_DATA.md)
checksums/, manifests/
optional_tier_b/      # placeholders + strike baselines
validation/           # filled by validate_tier_a.py
```

## 6. Quick start (Tier A)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r environment/requirements-tierA.txt
python scripts/verify_checksums.py
python scripts/generate_assets.py
PYTHONPATH=src python -m pytest tests/unit/test_aggregation_schemes.py -q
```

## 7. Figure regeneration
```bash
python scripts/generate_assets.py
# writes figures/fig_*.pdf|png
```

## 8. Table regeneration
```bash
python scripts/generate_assets.py
# writes tables/tab_*.tex and supplementary/tab_si_*.tex
```

## 9. Manuscript build
```bash
cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex
```

## 10. Checksum verification
```bash
python scripts/verify_checksums.py
```

## 11. Environment
Python **3.12**. Prefer `environment/requirements-tierA.txt`.
The full lab `environment/pip_freeze.txt` is archival and includes non-public editable installs.

## 12. Data availability
See `NOTICE_DATA.md`. Skeleton/video/`features_cache` are not redistributed here.

## 13. Licence
MIT for software. BoxingVI labels: cite upstream; see notice.

## 14. What is not included
Legacy archives, aborted runs, ISWA/Information Fusion manuscripts, raw video, skeletons,
unapproved feature caches, dashboards, Git history, LaTeX auxiliaries.

## 15. Known limitations
- Natural α≡1 on held-fixed matrices; missingness is synthetic.
- Combined timeline uses a fixed strike component.
- Exact recreation of the full lab pip freeze is not claimed.
- Tier B re-execution is not validated in this deposit.

## 16. Citation
See `CITATION.cff`. A Zenodo DOI will be assigned when the archive is published.
""",
    )

    # --- Manifests (paths recorded before checksum pass) ---
    # filled after all files written
    print(f"Staging tree created at {STAGING}")


def write_manifests_and_checksums() -> None:
    manifests = STAGING / "manifests"
    checksums = STAGING / "checksums"
    manifests.mkdir(exist_ok=True)
    checksums.mkdir(exist_ok=True)

    files: list[Path] = []
    skip_prefixes = (
        "validation/",
        ".pytest_cache/",
        ".git/",
    )
    skip_parts = {"__pycache__"}
    skip_suffixes = {".pyc", ".pyo", ".aux", ".bbl", ".blg", ".fls", ".fdb_latexmk", ".log", ".out", ".spl", ".synctex.gz"}
    for p in sorted(STAGING.rglob("*")):
        if not p.is_file():
            continue
        # Do not follow / hash through broken or transient paths
        rel = p.relative_to(STAGING).as_posix()
        if any(rel.startswith(pref) for pref in skip_prefixes):
            continue
        if any(part in skip_parts for part in Path(rel).parts):
            continue
        if Path(rel).suffix in skip_suffixes:
            continue
        if rel in {"checksums/SHA256SUMS", "checksums/PACKAGE_MANIFEST.sha256"}:
            continue
        if rel.startswith(("manuscript/figures/", "manuscript/tables/", "manuscript/supplementary/")):
            continue
        if p.is_symlink():
            continue
        files.append(p)

    write(
        manifests / "files.txt",
        "\n".join(p.relative_to(STAGING).as_posix() for p in files) + "\n",
    )

    # Canonical results manifest
    canon = {
        "canonical_run_id": RUN_ID,
        "aggregation_schemes_sha256": AGG_HASH,
        "result_files": [
            "experiment_summary.csv",
            "per_video_metrics.csv",
            "paired_comparisons.csv",
            "dropout_results.csv",
            "failure_counts.csv",
            "interaction_rule_firings.csv",
            "matrices/matrix_meta.csv",
        ],
        "numbers_from_tables_json": json.loads((STAGING / "tables" / "numbers.json").read_text()),
    }
    write(manifests / "canonical_results.json", json.dumps(canon, indent=2) + "\n")

    # Figure/table provenance
    rows = [
        "manuscript_item,output_file,generation_script,input_files,canonical_source,command",
        "fig:architecture,figures/fig_architecture.pdf,scripts/generate_assets.py,analysis/*;results/.../experiment_summary.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "fig:aggregation,figures/fig_aggregation.pdf,scripts/generate_assets.py,results/.../experiment_summary.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "fig:pervideo,figures/fig_per_video.pdf,scripts/generate_assets.py,results/.../per_video_metrics.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "fig:video-contribution,figures/fig_video_contribution.pdf,scripts/generate_assets.py,analysis/per_video_contribution_weighted.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "fig:interactions,figures/fig_interactions.pdf,scripts/generate_assets.py,results/.../experiment_summary.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "fig:dropout,figures/fig_dropout.pdf,scripts/generate_assets.py,results/.../dropout_results.csv;analysis/dropout_per_video_means.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "fig:failures,figures/fig_failures.pdf,scripts/generate_assets.py,results/.../failure_counts.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "tab:dataset,tables/tab_dataset.tex,scripts/generate_assets.py,results/.../matrices/matrix_meta.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "tab:aggregation,tables/tab_aggregation.tex,scripts/generate_assets.py,results/.../experiment_summary.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "tab:interactions,tables/tab_interactions.tex,scripts/generate_assets.py,results/.../experiment_summary.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "tab:firings,tables/tab_firings.tex,scripts/generate_assets.py,results/.../interaction_rule_firings.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "tab:dropout,tables/tab_dropout.tex,scripts/generate_assets.py,results/.../dropout_results.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "tab:paired,tables/tab_paired.tex,scripts/generate_assets.py,results/.../paired_comparisons.csv,results/run_20260730_005150,python scripts/generate_assets.py",
        "tab:failures,tables/tab_failures.tex,scripts/generate_assets.py,results/.../failure_counts.csv,results/run_20260730_005150,python scripts/generate_assets.py",
    ]
    write(manifests / "figure_table_sources.csv", "\n".join(rows) + "\n")

    sums = []
    for p in files:
        rel = p.relative_to(STAGING).as_posix()
        sums.append(f"{sha256(p)}  {rel}")
    write(checksums / "SHA256SUMS", "\n".join(sums) + "\n")
    write(checksums / "PACKAGE_MANIFEST.sha256", "\n".join(sums) + "\n")


def write_release_status(extra: dict) -> None:
    write(
        STAGING / "RELEASE_STATUS.md",
        f"""# RELEASE_STATUS

Canonical run: `{RUN_ID}`  
Package version: `0.2.0-eaai-rc1`  
Zenodo DOI: not yet published for this EAAI package.

## Spec checklist

| Item | Status |
|------|--------|
| Curated staging tree | DONE |
| Track aggregation_schemes.py (hash={AGG_HASH[:12]}…) | DONE |
| Remove absolute paths from Tier A runnable files | DONE |
| Tier A CSVs / figures / tables / generate_assets | DONE |
| Environment freeze + Tier A requirements | DONE |
| Checksums + manifests | DONE |
| CITATION.cff / zenodo.json EAAI retarget | DONE |
| NOTICE_DATA.md | DONE |
| Annotations with notice | DONE |
| features_cache in package | BLOCKED — PENDING INSTITUTIONAL APPROVAL |
| Tier B runner (relative) present | DONE (not validated end-to-end) |
| Strike baselines | DONE (optional_tier_b) |
| Skeleton / video | NOT APPLICABLE (excluded) |
| Zenodo publish | NOT APPLICABLE (forbidden) |
| Git tag | NOT APPLICABLE (forbidden) |
| Push | NOT APPLICABLE (forbidden) |

## Tier B item statuses

| Item | Status |
|------|--------|
| features_cache | PENDING INSTITUTIONAL APPROVAL |
| strike_baselines | APPROVED FOR RELEASE |
| run_eaai_checkpoint.py | APPROVED FOR RELEASE (paths relativized; needs Tier B inputs) |
| annotations | APPROVED FOR RELEASE (also in Tier A) |
| skeleton/video | NOT REDISTRIBUTABLE |

## Validation snapshot

{json.dumps(extra, indent=2)}
""",
    )


def main() -> None:
    build_tree()
    write_manifests_and_checksums()
    write_release_status({"note": "run scripts/validate_tier_a.py next"})
    print("OK", STAGING)


if __name__ == "__main__":
    main()
