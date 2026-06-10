"""Standard on-disk layout for a single pipeline run (single source of truth)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MVPOutputPaths:
    """Expected artifact locations for any full run rooted at ``root``."""

    root: Path
    frames_dir: Path
    pose_keypoints_csv: Path
    features_csv: Path
    risk_scores_csv: Path
    events_json: Path
    output_overlay_mp4: Path
    report_md: Path
    stitched_preview_mp4: Path


def paths_for_run_root(output_root: Path) -> MVPOutputPaths:
    """Build path bundle for a (typically resolved) run directory."""
    r = output_root.expanduser().resolve()
    return MVPOutputPaths(
        root=r,
        frames_dir=r / "frames",
        pose_keypoints_csv=r / "pose_keypoints.csv",
        features_csv=r / "features.csv",
        risk_scores_csv=r / "risk_scores.csv",
        events_json=r / "events.json",
        output_overlay_mp4=r / "output_overlay.mp4",
        report_md=r / "report.md",
        stitched_preview_mp4=r / "._stitched_preview.mp4",
    )


__all__ = ["MVPOutputPaths", "paths_for_run_root"]
