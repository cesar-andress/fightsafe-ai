# Tier B — optional full experiment re-execution

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
