"""Synthetic BoxingVI-style layout (small Excel + NumPy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fightsafe_ai.datasets.boxingvi import (
    BoxingVIDataset,
    BoxingVIEvent,
    _rows_to_events,
    inspect_annotation_xlsx,
    inspect_dataset,
    load_events_from_xlsx,
)


pytest.importorskip("openpyxl")


pytestmark = pytest.mark.unit


def _write_minimal_tree(root: Path) -> None:
    """data/boxingvi/{annotations,skeleton,rgb,metadata}/ with tiny artefacts."""
    ann = root / "annotations"
    sk = root / "skeleton"
    rgb = root / "rgb"
    meta = root / "metadata"
    for d in (ann, sk, rgb, meta):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "Start Frame": [0, 100],
            "End Frame": [30, 160],
            "Class": ["jab", "cross"],
        }
    )
    df.to_excel(ann / "V1.xlsx", index=False, engine="openpyxl")

    df2 = pd.DataFrame(
        {
            "Start Frame": [5],
            "End Frame": [12],
            "Class": ["hook"],
        }
    )
    df2.to_excel(ann / "V2.xlsx", index=False, engine="openpyxl")

    np.save(sk / "V1.npy", np.zeros((4, 17, 2), dtype=np.float32))
    np.save(sk / "V2.npy", np.ones((2, 17, 2), dtype=np.float32))

    (meta / "Meta_data.ods").write_bytes(b"synthetic placeholder")

    # Deliberately omit rgb/V1.mp4 — dataset must still resolve paths.


def test_list_video_ids_union(tmp_path: Path) -> None:
    root = tmp_path / "data" / "boxingvi"
    _write_minimal_tree(root)
    ds = BoxingVIDataset(root)
    assert ds.list_video_ids() == ["V1", "V2"]


def test_load_annotations_and_skeleton(tmp_path: Path) -> None:
    root = tmp_path / "data" / "boxingvi"
    _write_minimal_tree(root)
    ds = BoxingVIDataset(root)

    evs = ds.load_annotations("V1")
    assert evs == [
        BoxingVIEvent(0, 30, "jab"),
        BoxingVIEvent(100, 160, "cross"),
    ]

    sk = ds.load_skeleton("V1")
    assert sk.shape == (4, 17, 2)
    assert sk.dtype == np.float32
    assert float(sk.max()) == 0.0


def test_get_video_path_without_rgb_file(tmp_path: Path) -> None:
    root = tmp_path / "data" / "boxingvi"
    _write_minimal_tree(root)
    ds = BoxingVIDataset(root)
    p = ds.get_video_path("V1")
    assert p == root / "rgb" / "V1.mp4"
    assert not p.is_file()


def test_skeleton_only_list_ids(tmp_path: Path) -> None:
    root = tmp_path / "boxingvi"
    sk = root / "skeleton"
    sk.mkdir(parents=True)
    np.save(sk / "solo.npy", np.arange(6).reshape(1, 3, 2))
    ds = BoxingVIDataset(root)
    assert ds.list_video_ids() == ["solo"]
    arr = ds.load_skeleton("solo")
    assert arr.shape == (1, 3, 2)


def test_missing_annotation_raises(tmp_path: Path) -> None:
    root = tmp_path / "boxingvi"
    (root / "annotations").mkdir(parents=True)
    ds = BoxingVIDataset(root)
    with pytest.raises(FileNotFoundError):
        ds.load_annotations("missing")


def test_missing_skeleton_raises(tmp_path: Path) -> None:
    root = tmp_path / "boxingvi"
    (root / "skeleton").mkdir(parents=True)
    ds = BoxingVIDataset(root)
    with pytest.raises(FileNotFoundError):
        ds.load_skeleton("missing")


def test_empty_annotation_sheet(tmp_path: Path) -> None:
    root = tmp_path / "boxingvi"
    ann = root / "annotations"
    ann.mkdir(parents=True)
    pd.DataFrame(columns=["Start Frame", "End Frame", "Class"]).to_excel(
        ann / "empty.xlsx", index=False, engine="openpyxl"
    )
    ds = BoxingVIDataset(root)
    assert ds.load_annotations("empty") == []


def test_rows_to_events_boxingvi_headers_and_unnamed_ignored() -> None:
    """Real spreadsheets may use Start_Frame / Ending_Frame and padded Unnamed columns."""
    df = pd.DataFrame(
        {
            "Start_Frame": [10, 20],
            "Unnamed: 1": [999, 888],
            "Ending_Frame": [15, 25],
            "Unnamed: 3": [0.0, 0.0],
            "Class": ["jab", "cross"],
            "Unnamed: 5": [1, 2],
        }
    )
    evs = _rows_to_events(df)
    assert evs == [
        BoxingVIEvent(10, 15, "jab"),
        BoxingVIEvent(20, 25, "cross"),
    ]


def test_rows_to_events_missing_required_columns_error_shows_normalized_keys() -> None:
    df = pd.DataFrame({"Wrong Col": [1], "Unnamed: 2": [9]})
    with pytest.raises(ValueError, match="Normalized column keys"):
        _rows_to_events(df)


def test_type_column_alias_and_multi_sheet(tmp_path: Path) -> None:
    root = tmp_path / "data" / "boxingvi"
    ann = root / "annotations"
    ann.mkdir(parents=True)
    out = ann / "V3.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(
            {"Start Frame": [1], "End Frame": [2], "Type": ["hook"]},
        ).to_excel(w, sheet_name="Round1", index=False)
        pd.DataFrame(
            {"StartFrame": [50], "EndFrame": [60], "Label": ["uppercut"]},
        ).to_excel(w, sheet_name="Round2", index=False)
    ds = BoxingVIDataset(root)
    evs = ds.load_annotations("V3")
    assert len(evs) == 2
    assert evs[0].class_name == "hook"
    assert evs[1].class_name == "uppercut"


def test_load_events_spacer_five_columns(tmp_path: Path) -> None:
    """Pattern C: 5 columns with blank spacers; start col0, end col2, class col4."""
    ann = tmp_path / "annotations"
    ann.mkdir(parents=True)
    path = ann / "V9like.xlsx"
    raw = pd.DataFrame(
        [
            [6675, float("nan"), 6688, float("nan"), "Jab"],
            [100, float("nan"), 120, float("nan"), "Lead Hook"],
        ]
    )
    raw.to_excel(path, header=False, index=False, engine="openpyxl")
    evs = load_events_from_xlsx(path)
    assert len(evs) == 2
    assert BoxingVIEvent(6675, 6688, "Jab") in evs
    assert any("Hook" in e.class_name for e in evs)


def test_load_events_three_columns_no_header(tmp_path: Path) -> None:
    """Pattern B: three columns, first row is data (no header row)."""
    ann = tmp_path / "annotations"
    ann.mkdir(parents=True)
    path = ann / "nohdr.xlsx"
    raw = pd.DataFrame([[6675, 6688, "Jab"], [10, 20, "Cross"]])
    raw.to_excel(path, header=False, index=False, engine="openpyxl")
    evs = load_events_from_xlsx(path)
    assert [e.start_frame for e in evs] == [10, 6675]
    assert [e.class_name for e in evs] == ["Cross", "Jab"]


def test_load_events_header_not_on_first_row(tmp_path: Path) -> None:
    """Header row after a title row; columns map via names on row 1."""
    ann = tmp_path / "annotations"
    ann.mkdir(parents=True)
    path = ann / "latehdr.xlsx"
    raw = pd.DataFrame(
        [
            ["Title row", None, None, None, None, None],
            ["Start_Frame", None, "Ending_Frame", None, "Class", None],
            [10, 0, 15, 0, "Jab", 0],
            [20, 0, 25, 0, "Rear Hook", 0],
        ]
    )
    raw.to_excel(path, header=False, index=False, engine="openpyxl")
    evs = load_events_from_xlsx(path)
    assert len(evs) == 2
    assert evs[0].start_frame == 10 and evs[0].end_frame == 15
    assert evs[1].class_name == "Rear Hook"


def test_inspect_failed_includes_debug(tmp_path: Path) -> None:
    """FAILED inspect carries sheet preview (columns + first rows)."""
    ann = tmp_path / "annotations"
    ann.mkdir(parents=True)
    path = ann / "broken.xlsx"
    pd.DataFrame({"only": [1, 2]}).to_excel(path, index=False, engine="openpyxl")
    rep = inspect_annotation_xlsx(path)
    assert not rep.ok
    assert rep.inspect_debug is not None
    assert "sheet_names" in rep.inspect_debug
    assert "raw_columns" in rep.inspect_debug


def test_inspect_dataset_reports_per_file(tmp_path: Path) -> None:
    root = tmp_path / "data" / "boxingvi"
    _write_minimal_tree(root)
    reports = inspect_dataset(root)
    assert len(reports) >= 1
    v1 = next(r for r in reports if r.path.name == "V1.xlsx")
    assert v1.ok
    assert v1.status == "OK"
    assert v1.valid_event_count >= 1
    assert v1.first_event is not None
    assert v1.raw_shape is not None
    assert v1.raw_shape[1] >= 3
    assert "Start" in "".join(v1.columns_detected or ())
    assert v1.sheets_used  # default sheet name from pandas/openpyxl
