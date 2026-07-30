"""
FightSafe AI

AI-assisted safety detection for combat sports officiating.

Authors:
- David Martin Moncunill (david.martinm@ucjc.edu)
- César Andrés Sánchez (cesar.andress@ucjc.edu)

Affiliation:
Camilo José Cela University (UCJC)
Madrid, Spain

This module is part of a research-oriented system for human-in-the-loop safety analysis.

Keypoint I/O and schema constants.
"""

from fightsafe_ai.keypoints.io import (
    load_indexed_sequence,
    load_keypoint_csv,
    load_keypoint_csv_indexed,
    load_landmark_maps_ordered,
)


__all__ = [
    "load_indexed_sequence",
    "load_keypoint_csv",
    "load_keypoint_csv_indexed",
    "load_landmark_maps_ordered",
]
