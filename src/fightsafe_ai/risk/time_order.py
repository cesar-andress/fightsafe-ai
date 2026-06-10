"""
Time-ordering of per-frame feature rows and wall-clock ``timestamp`` for risk exports.

Not medical data — research / decision-support timing only.
"""

from __future__ import annotations

import pandas as pd

from fightsafe_ai.utils.sorting import natural_sort_strings


def sort_frames_add_timestamp(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    One row per frame, natural-sorted by ``frame_id``; set ``timestamp`` = row_index / ``fps`` (seconds).
    """
    w = df.copy()
    w["frame_id"] = w["frame_id"].astype(str)
    fids = natural_sort_strings(w["frame_id"].unique().tolist())
    order = {fid: i for i, fid in enumerate(fids)}
    w["_o"] = w["frame_id"].map(lambda x: order[str(x)])
    w = w.sort_values("_o").drop(columns=["_o"])
    w = w.reset_index(drop=True)
    w["timestamp"] = w.index / float(fps)
    return w
