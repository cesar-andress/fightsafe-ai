# FightSafe AI

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:**
Camilo José Cela University (UCJC)
Madrid, Spain

---

# Roadmap

Short-term goals focus on a reproducible **MVP pipeline**; longer-term items move toward validated safety analytics.

## Near term

- [x] CI workflow: `make ci` (Ruff format/check, `mypy`, `pytest` with coverage ≥ 80%).
- [ ] Golden-file tests for CSV schemas and one end-to-end smoke test on a tiny synthetic clip.
- [ ] Optional `python-dotenv` integration for loading `.env` in CLI.
- [ ] Document calibration: mapping heuristic scores to venue-specific thresholds.

## Medium term

- [ ] Additional pose backends (e.g. ONNX exports) behind the same `BasePoseEstimator` interface.
- [ ] Temporal smoothing / Kalman filtering for keypoints.
- [ ] Export **Parquet** / **Feather** alongside CSV for large batches.
- [ ] **TapKO:** protocol-aligned **real** clip corpus + reviewed annotations (schema in [`tapko_annotation.md`](tapko_annotation.md)); extend evaluation reports beyond synthetic/unit tests. CLI entry points: `fightsafe tapko-detect`, `tapko-evaluate` (see [`evaluation.md`](evaluation.md)).

## Long term

- [ ] Learned risk models trained on labeled events with fairness review.
- [ ] Multi-person tracking and interaction-aware risk for sparring analytics.
- [ ] Edge deployment packaging (TensorRT / Core ML) — scope TBD.

Contributions should align with `docs/contributing.md` and open an issue before large architectural changes.
