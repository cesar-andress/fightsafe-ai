# Tier B — optional full experiment re-execution

Tier B is **not required** for regenerating paper tables and figures (Tier A).

| Item | Status |
|------|--------|
| `inputs/features_cache/*.pkl` | not included in the public package |
| `inputs/strike_baselines/*.json` | included |
| `scripts/run_eaai_checkpoint.py` | optional runner; requires Tier B inputs |
| BoxingVI annotations | `annotations/boxingvi/` |
| Skeleton / raw video | not redistributable |

## Obtaining features_cache (authorized use only)

1. Obtain BoxingVI skeleton keypoints from the dataset authors.
2. Run the FightSafe AI feature extraction pipeline under institutional licence review.
3. Place `V1.pkl`…`V10.pkl` under `optional_tier_b/inputs/features_cache/`.
4. Verify hashes against `canonical_results/run_20260730_005150/input_checksums.tsv` when available.

Do not commit restricted feature caches to this repository.
