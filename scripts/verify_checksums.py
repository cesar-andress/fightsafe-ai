#!/usr/bin/env python3.12
"""Verify SHA-256 checksums for the release package."""
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
