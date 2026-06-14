# TapKO error analysis

Decision-support evaluation — not officiating metrics.

## Counts by category

| Category | Count |
|----------|-------|
| false_positive | 335 |
| missed_event | 8 |
| wrong_subtype | 2 |
| late_detection | 1 |

## Examples (up to 40 rows)

| video_id | category | detail | ref | pred | ref interval | pred interval | IoU |
|----------|----------|--------|-----|------|--------------|---------------|-----|
| jedi_submissions | wrong_subtype | IoU match with same namespace but different subtype (family TP) | submission_signal.hand_tap | submission_signal.foot_tap | 480.000–486.000 | 482.500–484.233 | 0.2889 |
| jedi_submissions | late_detection | pred onset later than ref by 2.500s | submission_signal.hand_tap | submission_signal.foot_tap | 480.000–486.000 | 482.500–484.233 | 0.2889 |
| jedi_submissions | wrong_subtype | structural match across namespaces (family mismatch) | submission_signal.hand_tap | extreme_vulnerability.no_intelligent_defense | 130.000–136.000 | 131.033–132.667 | 0.2722 |
| jedi_submissions | missed_event | no prediction reached IoU threshold | submission_signal.hand_tap |  | 83.000–88.000 |  |  |
| jedi_submissions | missed_event | no prediction reached IoU threshold | submission_signal.hand_tap |  | 192.000–201.000 |  |  |
| jedi_submissions | missed_event | no prediction reached IoU threshold | submission_signal.hand_tap |  | 254.000–260.000 |  |  |
| jedi_submissions | missed_event | no prediction reached IoU threshold | submission_signal.hand_tap |  | 322.000–328.000 |  |  |
| jedi_submissions | missed_event | no prediction reached IoU threshold | submission_signal.hand_tap |  | 383.000–387.000 |  |  |
| jedi_submissions | missed_event | no prediction reached IoU threshold | submission_signal.hand_tap |  | 432.000–438.000 |  |  |
| jedi_submissions | missed_event | no prediction reached IoU threshold | submission_signal.hand_tap |  | 529.000–536.000 |  |  |
| jedi_submissions | missed_event | no prediction reached IoU threshold | submission_signal.hand_tap |  | 586.000–590.000 |  |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 4.733–5.800 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 6.533–8.167 |  |
| jedi_submissions | false_positive | unmatched prediction |  | extreme_vulnerability.no_intelligent_defense |  | 8.067–8.800 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 15.367–17.133 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 17.067–18.033 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 19.233–20.967 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 30.700–31.000 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 46.833–48.600 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.hand_tap |  | 47.367–48.633 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 48.367–50.067 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 50.033–51.767 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 51.600–53.367 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.hand_tap |  | 52.967–53.933 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 53.167–54.867 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 54.833–56.067 |  |
| jedi_submissions | false_positive | unmatched prediction |  | extreme_vulnerability.no_intelligent_defense |  | 55.867–57.567 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 56.400–58.100 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 57.933–58.700 |  |
| jedi_submissions | false_positive | unmatched prediction |  | extreme_vulnerability.no_intelligent_defense |  | 59.233–59.833 |  |
| jedi_submissions | false_positive | unmatched prediction |  | extreme_vulnerability.no_intelligent_defense |  | 67.300–72.300 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 71.967–73.167 |  |
| jedi_submissions | false_positive | unmatched prediction |  | extreme_vulnerability.no_intelligent_defense |  | 73.633–74.400 |  |
| jedi_submissions | false_positive | unmatched prediction |  | extreme_vulnerability.no_intelligent_defense |  | 75.200–75.700 |  |
| jedi_submissions | false_positive | unmatched prediction |  | extreme_vulnerability.no_intelligent_defense |  | 76.833–77.833 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 77.667–79.367 |  |
| jedi_submissions | false_positive | unmatched prediction |  | extreme_vulnerability.no_intelligent_defense |  | 78.467–79.067 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 79.200–80.967 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 80.767–82.533 |  |
| jedi_submissions | false_positive | unmatched prediction |  | submission_signal.foot_tap |  | 87.667–89.433 |  |
