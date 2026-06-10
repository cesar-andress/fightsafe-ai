#!/usr/bin/env python3
"""Extract FightSafe-Bench per-frame features from pose_keypoints.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    default_pose = root / "outputs/tapko/jedi_submissions/pose_keypoints.csv"
    default_out = root / "outputs/benchmark_features.csv"
    default_manifest = root / "outputs/tapko/jedi_submissions/tapko_manifest.json"

    parser = argparse.ArgumentParser(
        description="Extract FightSafe-Bench per-frame features from pose_keypoints.csv."
    )
    parser.add_argument(
        "--pose-csv",
        type=Path,
        default=default_pose,
        help=f"Input pose CSV (default: {default_pose})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help=f"Output CSV path (default: {default_out})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help="Optional manifest JSON with fps field",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override frame rate (Hz); default from manifest or 30",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(root / "src"))
    from fightsafe_ai.benchmark.feature_extraction import write_benchmark_features_csv

    manifest = args.manifest if args.manifest.is_file() else None
    df = write_benchmark_features_csv(
        args.pose_csv,
        args.output,
        fps=args.fps,
        manifest=manifest,
    )
    print(f"Wrote {len(df)} frames x {len(df.columns)} columns -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
