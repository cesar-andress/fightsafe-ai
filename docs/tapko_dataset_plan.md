# TapKO dataset planning

Planning document for building and maintaining **FightSafe-TapKO** reference data: positive targets for detectors/evaluators and hard negatives for precision. This complements the machine-readable schema in `src/fightsafe_ai/annotation/tapko_schema.py` and `docs/tapko_annotation.md`.

## Storage, licensing, and ethics (mandatory)

- **Do not redistribute copyrighted videos.** Treat broadcast, promotion, and commercial footage as restricted unless you have explicit written permission for redistribution.
- **Default storage model:** persist **metadata**, **validated TapKO JSON annotations**, **derived skeletons** (or other non-pixel features produced locally), and **local paths** or stable internal URIs to media the team already has lawful access to. Do **not** mirror full-resolution video to public buckets unless licensing explicitly allows it.
- **Ethics:** labels describe **annotator judgments** on observable cues, not commission outcomes or medical diagnoses. Choke/unconsciousness-like classes remain **candidates** only; document uncertainty in `notes` and use conservative language in downstream reporting.

## Annotation fields (baseline for every clip)

Every labeled interval must satisfy `TapkoAnnotation` / `TapkoAnnotationDocument` validation:

| Field | Role in dataset QA |
|-------|---------------------|
| `video_id`, `source_uri` | Stable linkage to media you are allowed to use internally |
| `start_time`, `end_time` | Seconds on the evaluation timeline |
| `event_type` | Maps to schema enum (see mapping below) |
| `visibility`, `occlusion_level` | Stratification and difficulty analysis |
| `actor_id`, `target_id` | Who performs / receives the action when relevant |
| `confidence` | Per-interval annotator confidence |
| `notes` | **Required at protocol level** for fine-grained subclasses (e.g. tap on mat vs on opponent) until enums are extended |
| `rater_id` | Audit trail |
| `requires_audio` | Must reflect verbal/audio-dependent labels |

**Subclass detail:** Where one schema type covers multiple planning classes (e.g. all “hand tap” variants map to `submission_signal.hand_tap`), encode the planning subclass in `notes` using a consistent keyword (e.g. `tap_surface:mat`, `tap_surface:opponent`) so exports can be filtered before any schema bump.

---

## Collection checklist

Operational summary for **collection targets**, **capture constraints**, and **confusion risks**. Numeric minima align with the detailed class sections below unless noted; adjust only via a versioned bump to this document.

### Reporting gate (strict)

**No precision/recall/F1 or headline discrimination metric may be reported for a positive class until that class has a stratified set of matching hard negatives**—i.e. negatives curated to attack the **same failure modes** (same camera domains, similar motion vigor, and overlapping poses) so reported scores are not inflated by background-only negatives. Dataset-wide “easy negative” pools are allowed for training but **do not** satisfy this gate for publication metrics.

---

### Positive Tap classes

| Class | Min. clips (goal) | Preferred camera angle | Audio required | Annotation difficulty | Common false positives |
|-------|-------------------|------------------------|----------------|------------------------|-------------------------|
| Hand tap on mat | 120 | Elevated broadcast (~30–45°), wide enough to see mat contact | No | Medium | Hand posting; coach signaling; wiping sweat; cage drum |
| Hand tap on opponent | 120 | Side / three-quarter cage (both athletes visible) | No | Medium–high | Grip fighting bursts; parrying slaps; strikes mistaken as taps |
| Foot tap (hands trapped) | 80 | Side or corner view showing legs + upper-body constraint | No | High | Walking on knees; bridging; pushing off opponent’s thigh |
| Verbal tap | 60 | Any angle where mouth + reaction visible; **paired** isolated audio when possible | **Yes** for high-confidence rows | Very high | Lip-sync ambiguity; corner yelling without surrender; crowd bleed |
| Technical submission / choke unconsciousness candidate | 120 total (split technical vs choke-proxy in `notes`) | Angle showing referee + athletes’ heads/torsos | Optional (referee audio helps) | Very high | Fatigue collapse; voluntary stop without injury; editing cuts |

---

### Positive KO / vulnerability classes

| Class | Min. clips (goal) | Preferred camera angle | Audio required | Annotation difficulty | Common false positives |
|-------|-------------------|------------------------|----------------|------------------------|-------------------------|
| KO collapse | 100 | Replay-friendly angles showing strike-to-drop chain | No | Medium | Slip/trip; seated exhaustion; rope/cage lean mistaken as collapse |
| Post-impact inactivity | 90 | Wide or elevated showing recovery trajectory | No | High | Voluntary rest; referee pause; camera occlusion framed as “inactive” |
| No intelligent defense | 50 | Top-down or cage-side G&P context | No | Very high | Tactical shell; cardio conserve; poor angle mimicking “frozen” |
| Ground-and-pound vulnerability | 70 | Overhead replay or cage-side that shows posture under strikes | No | High | Active defensive framing from wrong angle; stock wrestling rides |
| Choke unconsciousness candidate | 70 | Side angle on neck control + limb tone | Optional | Very high | Conscious fatigue; defensive neck tuck; referee repositioning |
| Referee intervention window | 60 | Wide shot with referee entry + athletes (may overlap other labels) | Optional | High | Doctor check between rounds; equipment adjustment; corner stool |

---

### Hard negatives

| Class | Min. clips (goal) | Preferred camera angle | Audio required | Annotation difficulty | Common false positives |
|-------|-------------------|------------------------|----------------|------------------------|-------------------------|
| Hand posting | 150 | Side / three-quarter scrambles | No | Low–medium | Mis-framed single taps; brief drum on mat |
| Grip fighting | 120 | Gi collar angles or chest-to-chest views | No | Medium | Short slap bursts during grip breaks |
| Normal scramble | 200 | Mixed (must match positive domains) | No | Medium | Fast taps during transitions (border clips) |
| Celebration slap | 40 | Post-bout wide | No | Low | Aggressive corner slap rhythm confusing tap cadence |
| Mat slap unrelated to submission | 80 | Wide + floor-visible | Rarely | Medium | Taunt stomps; corner counts (rules education clips) |
| Foot movement during escape | 90 | Focus on legs during entanglements | No | High | Foot taps during escapes (border); shrimping bursts |
| Normal takedown | 150 | Wrestling/MMA entry angles | No | Medium | Hard landings mistaken for KO collapse |
| Fall without KO | 120 | Same KO-positive domains | No | Medium | Knockdown with fast recovery vs KO slide |
| Guard recovery | 120 | Bottom-player dominant framing | No | Medium | Fake “limp” recoveries after submission attempts |

**Foot movement during escape:** annotate under `negative.normal_scramble` or extend schema later; use `notes`: `scenario:foot_escape_non_tap` for exports.

---

### Matching negatives to positives (minimum QA rule)

For each **positive TapKO class** you intend to cite in results, maintain at least **two** negative buckets that explicitly address its top confusions (see “Common false positives” columns): one **within-domain motion** negative (e.g. scramble vs tap) and one **appearance-colliding** negative (e.g. posting vs mat tap). Until both buckets meet minimum counts **for that class**, treat metrics as **internal-only**—do not report them as validated TapKO performance.

---

## Lessons from the first pilot

The first end-to-end TapKO pilot on **`jedi_submissions`** produced **TP=1**, **FP=336**, **FN=9** on one long instructional BJJ clip with draft (transcript-derived) labels. That outcome **confirms** that the technical pipeline (annotation validation → pose → heuristic detectors → predictions → evaluator) runs on real media, but also shows that **raw geometric heuristics are far too sensitive** on instructional footage relative to sparse draft positives—high false-positive rate per minute and weak overlap with unverified windows.

### Operational rules (post-pilot)

1. **Do not use transcript-derived windows as final ground truth.** They may seed hypotheses only; every cited positive interval must graduate through visual review.
2. **Every positive interval must be visually confirmed** by a trained annotator before it counts toward publishable evaluation.
3. **Every video** in an evaluation fold must include **hard negatives** matched to the failure modes of the classes reported (see reporting gate above).
4. **Instructional footage** is acceptable for **bootstrap / debugging / tooling QA**, not for **final** headline evaluation unless the study explicitly scopes limitations and refrains from precision claims that assume competition-like behaviour.
5. **Competition footage** (or equally controlled match-context media) is **preferred** for publishable TapKO evaluation where discrimination metrics are interpreted as research signals.
6. **Store provenance for every clip:** **source URL** (when lawful to reference), **local path** or internal URI, **annotation status** (draft vs confirmed), and **license / usage rights status**—aligned with the storage policy in this document.

### Minimum next dataset (confirmed labels + negatives)

Before treating TapKO metrics as externally meaningful, aim for at least:

- **10** visually confirmed **hand tap** intervals (schema `submission_signal.hand_tap`) across diverse contexts.
- **5** visually confirmed **foot tap** intervals (`submission_signal.foot_tap`).
- **10** visually confirmed **KO / extreme vulnerability** clips spanning the vulnerability namespaces used in evaluation (not clinical labels—visual proxies only).
- **20** **hard negative** intervals total, stratified across the confusion types below (not background-only filler).

These minima are **staging targets** toward the larger order-of-magnitude goals in **Summary targets (order-of-magnitude)** later in this document; bump counts via a versioned edit to this file.

### Hard negative types to prioritize (pilot-confirmed confusions)

Curate negatives that directly attack heuristic failure modes observed on instructional and broadcast-like domains:

- Hand posting (`negative.hand_posting`)
- Grip fighting (`negative.grip_fighting`)
- Normal scramble (`negative.normal_scramble`)
- Foot movement during escape (often logged under `negative.normal_scramble` with `notes`, or extend schema later—see Hard negatives table above)
- Instructional pause (static demonstration holds—encode with explicit `notes` until a dedicated type exists if needed)
- Demonstration transition (reset between techniques—encode with `notes`, distinct from competition submission finish)
- Fall without KO (`negative.fall_without_ko`)
- Celebration slap (`negative.celebration_slap`)

---

## Positive classes

Planning intent → recommended `event_type` (schema v1.0). Extend schema only after QA on `notes` conventions.

### Hand tap on mat

| Item | Detail |
|------|--------|
| **Visual cue** | Repeated palm/finger contact with the mat or cage floor in a tapping rhythm; often adjacent to referee or grounded opponent; wrists travel vertically or patting motion distinct from posting. |
| **Required annotation fields** | Full baseline; `event_type`: `submission_signal.hand_tap`; `notes`: include `tap_surface:mat` (or equivalent controlled vocabulary); `target_id` if tapping opponent body vs bare mat is ambiguous. |
| **Minimum clips desired** | 120 |
| **Likely sources** | Grappling promotions (gi/no-gi), MMA events with ground phases, instructional sparring with clear surrender conventions (consent documented). |
| **Licensing / ethics** | Same as global policy; prefer footage with athlete consent for research use where possible. |
| **Audio required** | No (`requires_audio: false`), unless verbal overlap forces review. |

### Hand tap on opponent

| Item | Detail |
|------|--------|
| **Visual cue** | Hand taps on opponent’s body or limbs (not the mat); multi-hit rhythm similar to mat tap but contact target is clearly the other athlete. |
| **Required annotation fields** | `submission_signal.hand_tap`; `notes`: `tap_surface:opponent`; `target_id`: opponent id when identifiable. |
| **Minimum clips desired** | 120 |
| **Likely sources** | Same as mat tap; prioritize angles showing both athletes’ torsos. |
| **Licensing / ethics** | Same as global policy. |
| **Audio required** | No. |

### Foot tap when hands are trapped

| Item | Detail |
|------|--------|
| **Visual cue** | Foot/heel taps mat or opponent while upper limbs appear trapped or pinned; may coincide with leg entanglements; distinct from steady posting or walking on knees. |
| **Required annotation fields** | `submission_signal.foot_tap`; `notes`: indicate trapped-arm proxy (e.g. `context:hands_trapped`) and visibility caveats. |
| **Minimum clips desired** | 80 |
| **Likely sources** | Submission grappling, MMA ground fights. |
| **Licensing / ethics** | Same as global policy. |
| **Audio required** | No. |

### Verbal tap

| Item | Detail |
|------|--------|
| **Visual cue** | Mouth movement, corner reaction, or referee step-in consistent with verbal surrender; often **not** sufficient alone without audio in broadcast mixes. |
| **Required annotation fields** | `submission_signal.verbal_tap`; `requires_audio`: **true** when label depends on hearing “tap” / corner yelling / referee confirmation; `notes`: broadcast vs cage mic uncertainty. |
| **Minimum clips desired** | 60 |
| **Likely sources** | MMA/grappling broadcasts with multi-track or isolated audio if legally obtainable; fewer clips acceptable if quality high. |
| **Licensing / ethics** | Audio may have separate rights; document in `source_uri` provenance; do not redistribute raw audio without permission. |
| **Audio required** | **Yes** for high-confidence rows; if video-only guess, lower `confidence` and say so in `notes`. |

### Technical submission / choke unconsciousness candidate

| Item | Detail |
|------|--------|
| **Visual cue** | Two related intents: (a) **technical stoppage proxy**—referee positioning, athlete stopped fighting, limb goes limp without clear tap; (b) **choke-related compromise proxy**— sustained neck control + reduced defensive framing (schema separates choke candidate explicitly—see next row if splitting). |
| **Required annotation fields** | Use `submission_signal.technical_submission_candidate` for stoppage/medical-adjacent **visual** proxies; use `extreme_vulnerability.choke_unconsciousness_candidate` for choke-focused neck proximity + inactivity cues **without** claiming unconsciousness. `notes`: disambiguate which proxy; never assert clinical unconsciousness. |
| **Minimum clips desired** | 70 each intent (140 total if split evenly). |
| **Likely sources** | MMA, grappling rule sets with visible referee intervention. |
| **Licensing / ethics** | High sensitivity: conservative labels; suitable for research disclaimers in publications. |
| **Audio required** | Optional; set `requires_audio` true only if referee verbal stop is part of evidence. |

### KO collapse

| Item | Detail |
|------|--------|
| **Visual cue** | Sudden loss of posture after strike or impact: head snap, drop to canvas, legs buckling; distinguish from fatigue slump when possible via `notes`. |
| **Required annotation fields** | `extreme_vulnerability.ko_collapse`; `visibility`/`occlusion_level` honest; `notes`: strike visible vs obscured. |
| **Minimum clips desired** | 100 |
| **Likely sources** | MMA, boxing, kickboxing events; knockdown reels (rights permitting). |
| **Licensing / ethics** | Same as global policy; avoid sensational reuse outside research context. |
| **Audio required** | No. |

### Post-impact inactivity

| Item | Detail |
|------|--------|
| **Visual cue** | After identifiable impact or submission entry, athlete shows prolonged low motion, disorganized posture, or non-defensive stillness beyond normal recovery rhythm. |
| **Required annotation fields** | `extreme_vulnerability.post_impact_inactivity`; `notes`: anchor “impact” or prior event if subjective (timestamp reference). |
| **Minimum clips desired** | 90 |
| **Likely sources** | MMA, boxing knockdowns, grappling heavy shots to takedown sequences. |
| **Licensing / ethics** | Same as global policy. |
| **Audio required** | No. |

### No intelligent defense

| Item | Detail |
|------|--------|
| **Visual cue** | Extended periods where guard structure is absent while under sustained striking or control—high ambiguity; use sparingly and cross-check in QA. |
| **Required annotation fields** | `extreme_vulnerability.no_intelligent_defense`; `confidence` often mid-range; `notes`: why defense looks absent (angle, fatigue vs impairment). |
| **Minimum clips desired** | 50 (quality over quantity). |
| **Likely sources** | MMA ground-and-pound sequences, shelled-up positions on cage. |
| **Licensing / ethics** | Label as visual proxy only; avoid medical claims. |
| **Audio required** | No. |

### Ground-and-pound vulnerability

| Item | Detail |
|------|--------|
| **Visual cue** | Top athlete in ride/mount/side control delivering repeated strikes while bottom athlete shows minimal framing—overlap with “no intelligent defense” but **emphasizes striking context from top position**. |
| **Required annotation fields** | **Schema v1.0:** no dedicated enum. Options: (a) annotate as `extreme_vulnerability.no_intelligent_defense` with `notes`: `context:ground_and_pound`; or (b) extend schema in a future version with an explicit type—until then, lock the convention in `notes` for dataset exports. |
| **Minimum clips desired** | 70 |
| **Likely sources** | MMA promotions with clear overhead or cage-side angles. |
| **Licensing / ethics** | Same as global policy; graphic content—review institutional IRB or equivalent if human-subjects framing applies. |
| **Audio required** | No. |

---

## Negative classes

Use existing `negative.*` types where they exist; encode extra nuance in `notes` until enums expand.

### Hand posting

| Item | Detail |
|------|--------|
| **Visual cue** | Hand/arm placed on mat or opponent for base or frames without rhythmic tap burst; often during scrambles or guard retention. |
| **Required annotation fields** | `negative.hand_posting`; `notes`: distinguish single-post vs switching hands. |
| **Minimum clips desired** | 150 |
| **Likely sources** | Grappling scrambles, wrestling entries. |
| **Licensing / ethics** | Same as global policy. |
| **Audio required** | No. |

### Grip fighting

| Item | Detail |
|------|--------|
| **Visual cue** | Collar/elbow/wrist ties without cessation tap-like bursts; continuous establishment of grips. |
| **Required annotation fields** | `negative.grip_fighting`. |
| **Minimum clips desired** | 120 |
| **Likely sources** | Gi and no-gi grappling. |
| **Licensing / ethics** | Same as global policy. |
| **Audio required** | No. |

### Normal scramble

| Item | Detail |
|------|--------|
| **Visual cue** | Fast limb movement, hip heists, framing recovery—looks busy like tap but intent is positional fighting. |
| **Required annotation fields** | `negative.normal_scramble`. |
| **Minimum clips desired** | 200 |
| **Likely sources** | Any grappling-heavy footage. |
| **Licensing / ethics** | Same as global policy. |
| **Audio required** | No. |

### Celebration slap

| Item | Detail |
|------|--------|
| **Visual cue** | Corner/friend slaps, post-fight ritual contact—not match-ending submission. |
| **Required annotation fields** | `negative.celebration_slap`. |
| **Minimum clips desired** | 40 |
| **Likely sources** | Post-bout footage, weigh-in rituals (use ethically). |
| **Licensing / ethics** | Personality rights may apply; restrict redistribution. |
| **Audio required** | No. |

### Mat slap unrelated to submission

| Item | Detail |
|------|--------|
| **Visual cue** | Audible-looking or visible mat strikes from adjusting stance, taunting, or coach signaling—not surrender tapping rhythm. |
| **Required annotation fields** | **Schema v1.0:** no dedicated type—use `negative.hand_posting` or `negative.normal_scramble` plus `notes`: `reason:mat_slap_non_submission` until a new enum is approved. |
| **Minimum clips desired** | 80 |
| **Likely sources** | MMA walk-away moments, rule explanations, corner drills. |
| **Licensing / ethics** | Same as global policy. |
| **Audio required** | Rarely; optional. |

### Normal takedown

| Item | Detail |
|------|--------|
| **Visual cue** | Clean level change, entries, mat returns **without** KO-like collapse or submission surrender signals. |
| **Required annotation fields** | **Schema v1.0:** use `negative.fall_without_ko` when impact-like landing happens without KO proxy; otherwise encode “clean takedown” as `notes` on `negative.normal_scramble` or extend schema—recommended `notes`: `scenario:normal_takedown`. |
| **Minimum clips desired** | 150 |
| **Likely sources** | Wrestling, MMA, judo footage. |
| **Licensing / ethics** | Same as global policy. |
| **Audio required** | No. |

### Fall without KO

| Item | Detail |
|------|--------|
| **Visual cue** | Fighter goes down from slip, trip, fatigue, or throw without neurological compromise cues. |
| **Required annotation fields** | `negative.fall_without_ko`; `notes`: trip vs impact when visible. |
| **Minimum clips desired** | 120 |
| **Likely sources** | Same as takedowns / grappling. |
| **Licensing / ethics** | Same as global policy. |
| **Audio required** | No. |

### Guard recovery

| Item | Detail |
|------|--------|
| **Visual cue** | Closed/open guard recovery, knee shields, hand fights back to feet—**active** defensive structure after danger moments. |
| **Required annotation fields** | **Schema v1.0:** no dedicated enum—use `negative.normal_scramble` with `notes`: `scenario:guard_recovery` (or `negative.grip_fighting` if grip-dominant). |
| **Minimum clips desired** | 120 |
| **Likely sources** | Grappling, MMA bottom-game footage. |
| **Licensing / ethics** | Same as global policy. |
| **Audio required** | No. |

---

## Summary targets (order-of-magnitude)

| Bucket | Approx. minimum clips |
|--------|------------------------|
| Positive (sum of rows above) | ~900+ |
| Negative (sum of rows above) | ~980+ |

Adjust upward for cross-promotion diversity (camera angle, gi/no-gi, gender, rule sets) before locking dataset v1.

---

## Versioning

When `tapko_schema.py` gains new `TapkoEventType` values (e.g. explicit ground-and-pound or mat-slap negative), migrate `notes` keywords into enums and bump `format_version` per `docs/tapko_annotation.md`.
