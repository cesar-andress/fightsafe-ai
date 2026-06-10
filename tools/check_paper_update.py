#!/usr/bin/env python3
"""Warn if src/fightsafe_ai/ changed against main but paper/main.tex was not changed."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SRC_PREFIX = "src/fightsafe_ai"
DEFAULT_PAPER_FILE = "../fusion2026/main.tex"
WARN = (
    "check-paper-update: diff vs {base!r}: {src!r} changed but {paper!r} did not.\n"
    "  If the change is architectural, methodological, or design-related, update "
    "the fusion manuscript main.tex (see docs/contributing.md, engineering-standards Section 11).\n"
    "  Files under {src!r}:\n{nfile}\n"
)

repo_root = Path(__file__).resolve().parent.parent


def _git(*args: str) -> tuple[int, str, str]:
    p = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    return p.returncode, p.stdout, p.stderr


def _ref_exists(name: str) -> bool:
    c, _, _ = _git("rev-parse", "--verify", name)
    return c == 0


def _resolve_base(explicit: str | None) -> str | None:
    if explicit:
        return explicit if _ref_exists(explicit) else None
    for candidate in ("main", "origin/main"):
        if _ref_exists(candidate):
            return candidate
    return None


def _changed_files(base: str) -> set[str]:
    """Paths differing from *base* in the working tree and the index (vs *base*)."""
    out: set[str] = set()
    for args in (("diff", "--name-only", base), ("diff", "--name-only", "--cached", base)):
        c, so, se = _git(*args)
        if c != 0:
            sys.stderr.write(
                f"check-paper-update: git {' '.join(args)} failed: {se.strip() or 'unknown'}\n"
            )
            return set()
        for line in so.splitlines():
            p = line.strip()
            if p:
                out.add(p.replace("\\", "/"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fail",
        action="store_true",
        help="exit with status 1 when a warning is printed (e.g. CI or pre-commit)",
    )
    ap.add_argument(
        "--base",
        default=None,
        metavar="REF",
        help="compare to this ref instead of main (default: first of main, origin/main)",
    )
    args = ap.parse_args()

    paper_file = os.environ.get("FUSION_MAIN", DEFAULT_PAPER_FILE).replace("\\", "/")

    base = _resolve_base(args.base)
    if base is None:
        msg = "check-paper-update: no base ref (try main, or pass --base). Skipping."
        if args.base:
            msg = f"check-paper-update: --base {args.base!r} not found. Skipping."
        sys.stderr.write(msg + "\n")
        return 0

    files = _changed_files(base)
    if not files:
        return 0

    if paper_file in files:
        return 0

    under_src = sorted(f for f in files if f == SRC_PREFIX or f.startswith(f"{SRC_PREFIX}/"))
    if not under_src:
        return 0

    nshow = 12
    head = under_src[:nshow]
    lines = "\n".join(f"    {p}" for p in head)
    if len(under_src) > nshow:
        lines += f"\n    ... ({len(under_src) - nshow} more)"

    sys.stderr.write(
        WARN.format(
            base=base,
            src=SRC_PREFIX,
            paper=paper_file,
            nfile=lines,
        )
    )
    return 1 if args.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
