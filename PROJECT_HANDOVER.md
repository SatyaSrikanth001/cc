# Behavioural Biometrics Project — Complete Handover

**Prepared:** 2026-09-11. Self-contained; assumes no knowledge of prior conversation.
**Status:** No pipeline frozen as deployable. Phase 4 is the only real test evidence and it failed the target.
**Source of numbers:** project artifacts, read directly. Anything not verifiable from files on this machine is marked **VERIFY**.

---

# PART 1 — PROJECT CONTEXT

## 1.1 Objective

Build a mobile behavioural biometric authentication system that decides, from a
user's interaction with a payments app (touch, keystroke, accelerometer,
gyroscope, orientation), whether the current session is the enrolled owner.

**Target: TAR ≥ 0.75 and FAR ≤ 0.05, ideally per user rather than on a cohort
average.**

**Deployment constraint, non-negotiable:** each user's model must be trainable
from that user's genuine enrolment sessions alone. No impostor data at
enrolment, no other users present at runtime.

## 1.2 The central scientific problem

In the training corpus **each operator used exactly one device**. Person identity
and device identity are therefore perfectly confounded there, so a feature that
separates people may simply be separating handsets.

This is not hypothetical. Phase 5 measured that a discriminative
owner-versus-background objective drives selection into the **89th–99th
percentile of device-shaped features**. Phase 4 then measured 24.4% real FAR
against a surrogate FAR of 0.3%, a **~76x gap**.

## 1.3 Current dataset — `features/v3_v2` (canonical)

All figures read from artifacts.

| Quantity | Value |
|---|---|
| Session CSVs | 18 (9 users x {training, testing}) |
| Pooled sessions | **1308** |
| Distinct operators (pooled) | **35** |
| Distinct devices | **9** |
| Enrolment sessions (`session_type='enrollment'`) | **727** |
| Attack sessions (`session_type='attack'`) | **581** |
| Raw CSV columns | **5233** |
| Common numeric features | **5226** |
| Frozen Stage A/B pool | **2011** |

**Cohort.** `users.txt` lists 9 canonical users: Parthish, Teja, akshaya,
pritam, ram, realme, reddy, rohity, surya. `cohort_active.txt` lists the **8**
used in all experiments; **akshaya is excluded** (5175 columns vs 5233, raw JSON
unavailable to regenerate). Data retained, not deleted.

**File structure.** `<user>_training_sessions.csv` holds that owner's genuine
enrolment. `<user>_testing_sessions.csv` holds genuine test sessions plus
impostor sessions performed **on that owner's device**.

**Person x device crossing.** Within training alone the design is a **9x9
diagonal** — 0 devices with >1 operator, 0 operators with >1 device — so person
and device are *statistically unidentifiable* there.

Pooling training + testing and relabelling by `operator_id` breaks the diagonal:

| Quantity | Pooled | Attack-phase estimable subset |
|---|---|---|
| Sessions | 1308 | **526** |
| Operators | 35 | **25** |
| Devices | 9 | **9** |
| Filled cells | 103 of 315 (density 0.327) | **85** |
| Operators on ≥2 devices | **27 of 35** | 25 |
| Connected components | **1** | **1** |
| Sessions per cell | — | median **5**, range 3–10 |

The single connected component is what makes person/device separable at all.

---

# PART 2 — PHASE-BY-PHASE HISTORY

Full append-only log: `phase2/EXPERIMENT_LOG.md` (~600 lines, includes all void
runs and bugs).

## Phase 1 / 1A — Project audit and protocol-parity audit
- **Objective:** understand the pipeline; check for a protocol artifact.
- **Result:** an earlier "oracle" result (FAR ≈ 0%) was invalid — genuine
  sessions had 5 form fields and attack sessions 6, making protocol features
  perfect separators. **In v3_v2 that artifact is GONE** (field-presence flags
  identical across groups, separation AUC 0.500).
- **New artifact found:** for 5 of 8 owners, genuine and impostor test blocks are
  **temporally disjoint** (`session_start_ts` separation 1.000). Recording phase
  is a live confound.
- **Frozen:** no.

## Phase 2 — Genuine-only filter stack and pilot
- **Data:** enrolment CSVs only, behind an audit-hook guard that raises on any
  attempt to open a testing CSV.
- **Result:** 5226 -> 2011 features. Duration-coupling veto (absent from v5,
  restored) removed 935 features.
- **Key finding:** Isolation Forest won on genuine acceptance at every subset
  size **while accepting 100% of shuffled sessions**. Rejected. This produced the
  negative-control gates (outlier ≤ 0.05, shuffled ≤ 0.60).
- **Frozen:** the pool — `phase2/experiments/pool_stageAB_frozen.json`,
  sha256 `cf9929c876eb2277`, verified deterministic across 3 processes.

## Phase 3 — Ablation, stability, finalist freeze
- **Result:** removing whole device-suspect families *hurt*. Temporal subsample
  stability was 0.03–0.13; consensus selection improved it to 0.20–0.33.
- **Frozen:** F1_PRIMARY (k=15, Mahalanobis), F2_CONSERVATIVE (k=20),
  F3_OCSVM_REF (k=30) in `phase3/frozen/`.

## Phase 4 — LOCKED FINAL EVALUATION (the only real test evidence)

| Finalist | macro TAR | macro FAR | pooled TAR | pooled FAR | met both |
|---|---|---|---|---|---|
| F1_PRIMARY | **0.903** | **0.244** | 0.903 | 0.245 | **1/8** |
| F2_CONSERVATIVE | 0.929 | 0.478 | 0.931 | 0.434 | 1/8 |
| F3_OCSVM_REF | 0.846 | 0.198 | 0.834 | 0.187 | 0/8 |

- **Conclusion:** TAR was never the problem. FAR is ~5x the ceiling.
- Mean per-user AUC 0.927 — discrimination is real, the operating point is wrong.
- **ROC diagnostic:** only **2 of 8** users have a curve entering the target box
  at *any* threshold. **Threshold recalibration alone cannot fix 6 of 8 users.**
- **Frozen and spent.** This test set can no longer certify anything.

## Phase 5 — Surrogate negatives; global+personal search
- **Decisive measurement:** using F1's frozen features, other users' enrolment is
  rejected at **0.003** while real test FAR was **0.244** (~76x).
  **The surrogate task was already saturated**, so optimising it adds nothing.
- **Headroom:** max cross-person AUC anywhere in the 2011-pool is **0.787**;
  **zero** features above 0.80.
- **Two void runs, both my own bugs:** (a) self-fulfilling TAR — threshold placed
  on the same held-out scores TAR was measured on, giving TAR = 0.744 for all 112
  configs; (b) architecture collapse — `total = gk + pk` drawn from one ranking,
  so arms A/B/C were identical computations.
- **Frozen:** no.

## Phase 6 — Multi-level stability
- 2011 features -> 1064 clusters at |r|≥0.80; **66% have a swappable twin**.
- Phase 5's stability verdict was an artifact of the metric: consensus frequency
  0.300 vs pairwise Jaccard 0.733 for the *same* procedure.
- **Data sufficiency: stability is FLAT from 20 sessions onward** while TAR rises
  steeply. More sessions improve the model, not selection stability.
- **Frozen:** no.

## Phase 7 — Pre-registered stability framework
- Nulls measured *before* choosing thresholds: random feature sets score **0.783**
  raw decision agreement, so raw agreement was **disqualified**; kappa-corrected
  agreement (null 0.176, real 0.600) became primary at threshold 0.40.
- 3 of 78 configs passed on cohort means; winner G5, 5 global + 8 personal.
- **One void run:** backbone built once and reused, making stability trivially 1.000.
- **Frozen:** no.

## Phase 8 — Per-owner robustness
- **Best of all 78 Phase 7 configs satisfies only 2 of 8 owners individually.**
  Three owners pass in **0 of 78**.
- **Structural finding:** spearman(hardware ratio, shuffled acceptance) = **-0.74**.
  A device-shaped feature is near-constant within an owner, so permuting it
  changes little and the shuffled session is still rejected — **a device-heavy set
  passes the shuffled control more easily.** Only **1 of 8** owners passes both.
- Family-wise generation (2011 -> 135, all 21 families retained) **did not** improve
  per-owner robustness; best was 3 of 8 against a pre-declared bar of 6 of 8.
- **NO WINNER. No constraint relaxed. Frozen: no.**

## Phase 9 — PERSON vs DEVICE MEASUREMENT STUDY

**What it attempted to measure.** For each feature, how much of its variance is
attributable to the **person**, the **device**, and residual session noise.

**Data used.** **Attack-phase sessions only** — 526 sessions, 25 operators,
9 devices, 85 cells. Restricted because owner-on-own-device cells are 100%
enrolment and off-diagonal cells are 100% attack; a full-pool decomposition would
attribute **recording phase** to **device**.

**Were genuine/impostor labels used?** **No.** Everything was relabelled by
`operator_id`. The genuine/impostor distinction was discarded entirely. This is
a measurement study, not authentication.

**Role of the IDs.** `operator_id` = the person effect; `device_id` = the device
effect. Their crossing is the entire basis of identifiability. Operators on only
one device were reported as `not_estimable` (9 of them), never silently dropped.

**Features analysed:** all **2011**.

**Method.** Henderson method-of-moments on the crossed unbalanced design;
(operator, device) **cell** bootstrap B=2000; two directional permutation nulls
(person: permute operator within device; device: permute device within operator),
200 each; seed 42.

### Results

| Classification | Count |
|---|---|
| Mixed | 995 |
| Confidently person-dominant | **707** |
| Confidently device-dominant | **112** |
| Uncertain | 102 |
| Weak/uninformative | 95 |

**Experiment 1 verdict: PARTIAL.** 819 confidently classified (bar ≥100: PASS),
median log-ratio CI width **2.357** (bar <2.0: FAIL). Both were required.

**Experiment 2 — validating the frozen Phase 2 hardware proxy:**
Spearman **0.565**, CI [0.510, 0.613], n=819. **PARTIAL** (needed ≥0.60).
Error profile: **63 agreements, 27 false alarms, only 1 missed detection.**

### What succeeded
- Person and device effects **are** separable in this corpus.
- Classifications are **mechanistically coherent**: strongest device-dominant are
  touch contact-geometry (`touch_dsize_*`, `touch_size_*`, `touch_ny_min`) —
  digitiser properties. Strongest person-dominant are gravity/tilt orientation and
  tap-triggered gyroscope energy — how a person holds and moves the phone.
- Family ordering: person end **key 6.16, grav 5.34, corr 4.36, grv 4.28, tap 3.60**;
  device end **tapdamp 0.33, dtw 0.82, place 0.82, coh 0.92, acc 0.97**.

### What only partially succeeded
Individual-feature precision. Only **32%** of features have an interval below the
bar; p90 width is 18.6.

### Why individual-feature conclusions remain uncertain
Two distinct causes, measured:
1. **The ratio statistic is intrinsically unstable.** R = person/(device+ε) puts
   device in the denominator. 108 features (5%) have device variance truncated to
   0; their median width is **18.82** vs **2.31** for the rest. Even excluding
   them the median is 2.31 — still over the bar.
2. **Design resolution.** Width correlates with device share at **-0.768** and
   residual share at **+0.664**.

The bounded **device share** is far better behaved than the ratio. Had the
pre-registration chosen share as primary, the verdict might have differed. That
is recorded as a limitation, **not** grounds to re-score anything.

### Two critical interpretation warnings

**Person-dominant does NOT mean authentication-useful.** The decomposition
measures variance attribution among 25 operators on 9 devices in attack-phase
sessions. It says nothing about whether a feature helps *reject an impostor on
the owner's own phone*. A feature can vary strongly between people and still
overlap heavily at the individual decision boundary.

**Device-dominant does NOT mean useless.** It means the feature primarily tracks
device variation *in this dataset*. Phase 3 measured that removing whole
device-suspect families made performance **substantially worse**, and Phase 9
shows why: device loading is heterogeneous *within* families (`acc` has 56
person-dominant and 47 device-dominant features). Blanket removal discards both.

**Frozen:** no. Experiment 3 was **not authorised** and was **not run**.

## Phase 10 — POWER AND EXPERIMENTAL-DESIGN ANALYSIS

**What it investigated.** Using the real design's index structure as a simulation
substrate with synthetic signal: how many operators, devices, crossed cells and
sessions per cell are needed for precise person-vs-device estimates.

**Not used:** any real feature value, any authentication metric, any Phase 9
classification. Pure design analysis. 408 designs x 5 signal conditions
(strong person / moderate person / balanced / moderate device / strong device),
2040 rows, seed 42, runtime 6190 s. Estimator unchanged.

### Why crossed cells matter
A person effect is estimable only for operators appearing on ≥2 devices. Cells
are the unit that carries person-device contrast. **The design must also be a
single connected component**, or effects are identifiable only within components.

### Why cells beat sessions — spearman(median_width, axis)

| Signal | **Cells** | Sessions/cell | Operators | Devices |
|---|---|---|---|---|
| strong_person | **-0.961** | +0.048 | -0.379 | -0.547 |
| moderate_person | **-0.970** | +0.090 | -0.401 | -0.529 |
| balanced | **-0.972** | +0.103 | -0.409 | -0.520 |
| moderate_device | **-0.973** | +0.099 | -0.420 | -0.512 |
| strong_device | **-0.971** | +0.075 | -0.438 | -0.494 |

Identical in all five conditions, so **not** an artifact of one synthetic ratio.
Sessions per cell is slightly **positive** — extra sessions in existing cells do
not help once cell count is fixed.

**Budget-matched** (same total sessions, restructured toward more cells):
71%, 64%, 63%, 75%, 75% improvement across five budget bands.

**Marginal value per 1000 extra sessions**, from the current-like baseline:
density 0.38->0.85 **+0.968**; devices 9->25 +0.747; operators 25->70 +0.660;
sessions/cell 5->20 **+0.171**. Raising crossing density is **~5.7x more
efficient per session** than adding sessions.

### Recommended future collection

| Design | Operators | Devices | Density | Cells | Sess/cell | Total | Worst width |
|---|---|---|---|---|---|---|---|
| **Minimum viable** | 15 | 9 | 0.60 | 81 | 3 | **243** | 1.267 |
| **Recommended** | 70 | 15 | 0.85 | 892 | 3 | **2676** | 0.339 |
| **High confidence** | 70 | 25 | 0.60 | 1050 | 3 | **3150** | 0.315 |

**The 70-operator question: 62 designs simulated, 23 pass, 39 FAIL.** 70 operators
is neither sufficient nor insufficient on its own — **crossing decides it**.
Fails: 70 ops x 5 devices at density 0.15–0.38 (140 cells) -> width 1.53.
Passes cheaply: 70 ops x 5 devices at density 0.85 (298 cells, 3 sess) -> 0.648
with only 894 sessions.

Explicit structure: **each operator uses ≥4 devices; each device is used by ≥30
operators; target ≥300 crossed cells; only 3 sessions per cell.**

### Estimator bias limitation (important)
Median |bias| is 0.057 (person) / 0.065 (device), **but** it reaches **0.90** for
the person component under strong-device signals and **0.53** for device under
strong-person signals. Only **73–74%** of design-signal cells meet bias <0.15.
The moment estimator systematically mis-attributes the **smaller** component when
the two are very unequal. This is a property of the estimator, not of any design.
**Extreme ratios are estimated less reliably than balanced ones.**

### What Phase 10 did NOT do
No feature selection, no Experiment 3, no authentication metric, no use of real
feature values, no new frozen artifact. Simulation results are design-based
estimates under stated assumptions, **not predictions** about a future dataset.

---

# PART 3 — EXACT CURRENT SCIENTIFIC POSITION

### 1. Can we say for every individual feature whether it is behavioural or device?
**No.** 819 of 2011 classify confidently, but only 32% have intervals tight enough
to order reliably. 995 are "mixed" and 102 "uncertain".

### 2. STRONGLY SUPPORTED
- Person and device effects are separable in the pooled corpus (single connected
  component, 27 of 35 operators on ≥2 devices).
- Classifications are mechanistically coherent (touch geometry -> device;
  gravity/tilt orientation -> person).
- The frozen hardware proxy carries genuine device information (ρ 0.565, CI lower
  bound 0.510, decisively above zero).
- **Crossed cells, not sessions per cell, are the binding design constraint** —
  holds in all five signal conditions.
- TAR is not the problem; FAR is (Phase 4: TAR 0.903, FAR 0.244).
- Threshold recalibration alone cannot fix 6 of 8 users.

### 3. PARTIALLY SUPPORTED (family/group level only)
- Family orderings (key/grav/corr person-leaning; tapdamp/dtw device-leaning).
  Families aggregate many features so medians are better determined than rows.
- The hardware proxy as a **conservative screen** — high recall (1 missed
  detection), low precision (27 false alarms).

### 4. NOT SUPPORTED
- That any Phase 5–10 configuration achieves the target. None has been tested.
- That removing device-loaded features improves authentication (Phase 3 measured
  the opposite for family-level removal).
- That the surrogate background FAR predicts real FAR (~76x gap).

### 5. CANNOT BE CONCLUDED
- Reliable ordering of individual features by person/device ratio.
- Whether any person-dominant feature is authentication-useful — never tested.
- Any FAR claim below ~18% for a single user: with ~15 attackers, zero breaches
  bounds the true rate at 18%, not 5%. **Success is currently unmeasurable even
  if achieved.**

### 6. Exact limitation of the current dataset
Two, jointly:
- **Structural:** only 85 crossed cells and 9 devices. Precision is bounded by
  cell count.
- **Statistical power:** ~5–16 attackers per user cannot certify FAR < 5%.

### 7. Why more sessions from the same person on the same device is NOT the fix
Three independent measurements:
- Phase 6: selection stability is **flat from 20 sessions onward**.
- Phase 10: sessions/cell correlates **+0.05 to +0.10** with interval width —
  slightly *harmful* once cells are fixed.
- Phase 10 budget-matched: restructuring the same budget toward cells improves
  precision **63–75%**.

More sessions improve TAR and covariance conditioning. They do **not** improve
selection stability or person/device resolution.

### 8. What future structure would solve it
70 operators x 15 devices at density 0.85 -> ~892 cells, 3 sessions each,
~2676 sessions. Each operator on ≥4 devices; each device used by ≥30 operators.

---

# PART 4 — THE OLDER SAME-DEVICE DATASET

## 4.1 Status: NOT PRESENT ON THIS MACHINE

Only `features/v3` (superseded, 12 CSVs) and `features/v3_v2` (canonical, 18 CSVs)
exist here. **Every structural claim below is the user's description and must be
VERIFIED against the actual files.**

## 4.2 Described structure — ALL ITEMS **VERIFY**

| Item | Described | Status |
|---|---|---|
| Profiles/owners | ~30 | **VERIFY** |
| Genuine sessions per owner | ~75 | **VERIFY** |
| Impostor sessions per profile | ~45 | **VERIFY** |
| Distinct impostors per profile | ~9 | **VERIFY** |
| Sessions per impostor | ~5 | **VERIFY** |
| Same phone for owner + impostors | yes | **VERIFY — the critical claim** |
| Total people | ~45 | **VERIFY** |
| Features | ~2000+ | **VERIFY** |
| Extraction pipeline | different | **VERIFY** |

Implied scale if accurate: ~2250 genuine + ~1350 impostor ≈ **3600 sessions**,
roughly 2.75x the current corpus, with ~30 same-device owner/impostor contrasts.

## 4.3 Scientific value

**Same-device genuine-versus-impostor is the contrast the current corpus lacks
in its training half.** If owner and impostors used the same phone, then device
is held constant within a profile, so any genuine/impostor separation **cannot**
be a device effect. That is exactly the confound that has blocked Phases 2–10.

Statistically it is also far stronger: ~30 profiles x ~9 impostors ≈ 270
owner-impostor pairs, against ~8 x ~15 now. **This may be enough to certify
FAR < 5%**, which the current corpus cannot do at any feature quality.

## 4.4 What it CAN investigate
- Whether a genuine-only selection methodology yields features that reject
  **same-device** impostors.
- Whether per-user FAR < 5% is attainable at all under this protocol.
- Whether the negative controls (shuffled, outlier) behave sensibly with real
  same-device negatives — directly testing the Phase 8 hardware/shuffled conflict.
- Whether a selection methodology generalises to held-out **people**.

## 4.5 What it CANNOT solve
- **It does not resolve person-vs-device attribution.** If each profile has one
  phone, the design is again a diagonal at profile level. Same-device
  genuine/impostor proves a feature *is not purely device* **within that profile**;
  it does not decompose variance. **VERIFY whether any person appears on ≥2 phones.**
- It cannot validate current-dataset feature *values* — different pipeline.
- It cannot substitute for the crossed collection Phase 10 specifies.
- Its age may mean different app version, sampling rate or handset generation. **VERIFY.**

## 4.6 Transfer: METHODOLOGY yes, FEATURES no

**THE METHODOLOGY CAN TRANSFER; THE FEATURE NAMES AND VALUES DO NOT NEED TO.**

What transfers: the protocol — train/test separation, genuine-only selection,
pre-registration, negative controls, leakage guards, stability definitions,
per-owner rather than cohort-mean aggregation, FAR confidence intervals.
These are properties of the *experimental design*, independent of feature names.

What must NOT transfer: any selected feature list, threshold, hyperparameter or
model fitted on the old data. Different extraction pipelines make names
meaningless across corpora, and a value-level transfer would be unfounded.

**Analogy:** we are validating the measuring procedure, not the measurements.

---

# PART 5 — HIGHEST-VALUE WORK BEFORE NEW DATA

## A. CURRENT DATASET

**A1. Re-express Phase 9 on the bounded device-share scale.** *(highest value here)*
- **Question:** does share, being bounded [0,1], classify more features confidently
  than the unstable ratio?
- **Data:** existing Phase 9 outputs. No new feature reads.
- **Labels:** none.
- **Type:** re-analysis, not selection.
- **Positive:** more usable family-level guidance. **Does NOT** overturn the Phase 9
  PARTIAL verdict — that stands under its own pre-registration.

**A2. Repair the shuffled control.** *(high value, independent)*
- **Question:** can a control be built that detects marginal-only models without
  correlating with hardware ratio (currently -0.74)?
- **Data:** enrolment only.
- **Labels:** none.
- **Type:** validation.
- **Positive:** removes a structural conflict where only 1 of 8 owners passes both.
  **Does NOT** improve FAR by itself.

**A3. Do NOT run more feature selection.** Five phases moved per-owner passes from
2 of 8 to 3 of 8. Headroom is capped (max cross-person AUC 0.787, none above 0.80).

## B. OLDER SAME-DEVICE DATASET — **the higher-value track**

**B1. Structural audit.** *(must be first)*
- **Question:** does it actually provide same-device genuine/impostor contrasts?
- **Data:** metadata columns only — IDs, labels, device, timestamps. **No feature values.**
- **Labels:** read but not used for any selection.
- **Positive:** unlocks B2–B4. **Negative:** the dataset cannot serve this purpose;
  say so and stop.

**B2. Protocol-parity audit.**
- **Question:** are genuine and impostor sessions comparable, or separated by a
  form-field / recording-phase artifact as v3 was?
- **Data:** metadata + protocol descriptors.
- **Positive:** results will be interpretable. **Negative:** parity controls needed
  before any authentication claim.

**B3. Methodology replication under pre-registration.**
- **Question:** does the genuine-only methodology reject **same-device** impostors?
- **Data:** old dataset, strict train/test split by **person**.
- **Type:** validation of a methodology, not feature discovery.
- **Positive:** the methodology is sound and the current corpus was the limitation.
  **Negative:** the methodology itself is inadequate — which would be the most
  important finding in the project.

**B4. FAR power analysis.**
- **Question:** how many impostors are needed to certify FAR < 5%?
- **Positive:** a certification specification. Cheap; do alongside B1.

**Explicitly NOT recommended:** uncontrolled feature-combination search on either
corpus.

---

# PART 6 — RECOMMENDED METHODOLOGY DEVELOPMENT PLAN (old dataset)

The user's 9-point list is close to correct. Below is the version I would run,
with **four corrections** derived from measured failures in Phases 2–10.

### Corrections to the proposed list

**C1. Split by PERSON, not by session.** Impostors recur across profiles. A
session-level split lets the same impostor appear in train and test. **Hold out
whole people, in both owner and impostor roles.**

**C2. Per-owner aggregation, never cohort means.** Phase 7 passed 3 of 78 configs
on cohort means; Phase 8 showed the best satisfied only **2 of 8 owners**. Declare
`n_owners_pass` as primary before searching.

**C3. Pre-register thresholds against measured NULLS, before results.** Phase 7's
null measurement disqualified raw decision agreement (random sets scored 0.783).
Any stability or control threshold must be justified against its own null.

**C4. Report FAR with exact binomial intervals and the attacker count.** FAR = 0
from 15 attackers is not FAR = 0 from 200. Cluster intervals on **attackers**, not
sessions.

### The plan

**Stage 0 — Audit (no modelling).** Structure, IDs, labels, same-device claim,
protocol parity, leakage candidates. Metadata only.

**Stage 1 — Pre-registration, locked before any search.** Cohort; person-level
split; feature eligibility rules; model set; threshold rule; negative controls
with null-derived thresholds; primary metric `n_owners_pass`; success/failure/stop
conditions; seeds.

**Stage 2 — Eligibility filtering.** Degenerate, duplicate, missing, duration-coupled,
clock-coupled, resolution-gated. Training data only. Report counts removed per reason.

**Stage 3 — Genuine-only selection inside nested CV.** Selection recomputed inside
every fold. Scaler fitted on the fit block only. Held-out people never influence
ranking, scaling, selection or fitting.

**Stage 4 — Freeze.** Feature list, preprocessing, model, hyperparameters,
threshold rule, seeds, cohort. Hash it.

**Stage 5 — Single evaluation on held-out people.** Per-owner TAR and FAR with
exact intervals, plus `n_owners_pass`. **One shot. No re-tuning afterwards.**

**Stage 6 — Generalisation analysis.** Does the *procedure* transfer to held-out
people? That is the transferable finding, not the feature list.

### Anti-optimisation guarantees
- Testing data never enters selection — enforced by an **audit-hook guard** that
  raises on any attempt to open a test file, plus a recorded list of files read.
- Distinguish **validation** (one shot, pre-registered) from **optimisation**
  (repeated, on development folds only).
- Every void run and bug preserved in the append-only log.
- After the single evaluation, **no re-selection**. A second question needs a
  second held-out set.

---

# PART 8 — FINAL DECISION TREE

```
WHAT WE SHOULD DO NOW
  Stop feature selection on the current dataset. Headroom is capped
  (max cross-person AUC 0.787) and 5 phases moved per-owner passes 2/8 -> 3/8.
        |
        v
WHAT THE OFFICE-LAPTOP CLAUDE SHOULD DO FIRST
  Audit the OLD dataset's structure from metadata only. Verify the
  same-device genuine/impostor claim. Propose a methodology. STOP for approval.
        |
        v
WHAT CAN BE LEARNED FROM THE OLD DATASET
  Whether a genuine-only methodology rejects SAME-DEVICE impostors;
  whether per-user FAR < 5% is attainable; whether the negative controls
  behave with real same-device negatives. Possibly enough attackers to
  CERTIFY FAR < 5%, which the current corpus cannot do at any feature quality.
        |
        v
WHAT CAN THEN BE APPLIED TO THE CURRENT DATASET
  THE METHODOLOGY ONLY — splits, controls, pre-registration, per-owner
  aggregation, FAR intervals. NEVER feature names, values, thresholds or models.
        |
        v
WHAT REQUIRES FUTURE NEW CROSSED PERSON-DEVICE DATA
  Reliable per-FEATURE person-vs-device attribution. Needs ~70 operators x
  ~15 devices at density 0.85 (~892 cells, 3 sessions each, ~2676 sessions),
  each operator on >=4 devices, each device used by >=30 operators.
  Neither the current nor the old dataset can substitute.
```

**Conservative bottom line.** Nothing is deployable. The only real test evidence
(Phase 4) failed at 1 of 8 users. The old dataset is the highest-value next step
because it may make success *measurable*, which it currently is not — but its
structure is unverified and the first task is to check it, not to use it.
