# COPY-PASTE PROMPT FOR THE OFFICE-LAPTOP CLAUDE

Copy everything between the two `=====` lines into a new Claude chat.

=====================================================================

You are joining an ongoing behavioural biometrics research project. You have no
knowledge of previous conversations. Read this brief carefully, then follow the
STEPS exactly. **Do not run large experiments or any feature selection until I
approve your proposed methodology.**

## BACKGROUND (context only — a different dataset, not the one you will use)

A previous phase of this project worked on a "current" dataset of 9 users,
5233 feature columns, on which each user used exactly one device. That work
reached these conclusions, which you should treat as established background:

- A frozen pipeline was evaluated once on held-out test data and **failed**:
  macro TAR 0.903, macro FAR 0.244, against a target of TAR ≥ 0.75 and FAR ≤ 0.05.
  Only 1 of 8 users met both conditions.
- ROC analysis showed that for **6 of 8 users, no threshold whatsoever** reaches
  the target box. Recalibration alone cannot fix it.
- The core blocker: in that dataset **person and device are confounded**, because
  each user used one phone. A variance decomposition classified 707 features as
  person-dominant and 112 as device-dominant, but only 32% of features had
  intervals tight enough to order reliably.
- Using other users' data as surrogate impostors was measured to be useless:
  surrogate FAR 0.003 vs real FAR 0.244, a ~76x gap.
- More sessions per user does **not** help selection stability (flat from 20
  sessions onward). More crossed person-device combinations does.
- Statistical power there is insufficient to certify FAR < 5% at all: with ~15
  attackers per user, zero breaches bounds the true rate at 18%.

**That dataset is NOT on your machine and you must not use or reason about its
feature values.**

## YOUR DATASET (the "old" / same-device dataset)

You have access to an OLDER dataset, collected with a DIFFERENT feature
extraction pipeline. Its structure is described below, but **this description is
approximate and unverified — treat every number as a hypothesis to check, not a
fact.**

- ~30 profiles/owners
- ~75 genuine sessions per owner
- ~45 impostor sessions per profile
- ~9 distinct impostors per profile, ~5 sessions each
- Genuine owner and impostors performed sessions **on the same phone** for that
  profile
- ~45 people involved overall
- ~2000+ features, different definitions from the current dataset

**The single most important thing to verify is the same-device claim**: that for
each profile, the owner AND the impostors used the same physical device. If true,
this dataset holds the contrast the project has been missing — genuine versus
impostor with device held constant.

## WHY THIS DATASET MATTERS

In the current dataset, any genuine/impostor separation might be a device effect.
If the old dataset truly holds device constant within a profile, then separation
there **cannot** be explained by the handset. That makes it valuable for
validating a *methodology*, and possibly for certifying FAR < 5%, which the
current dataset cannot do at any feature quality.

## THE GOAL — READ THIS TWICE

**The goal is NOT to find good features on this dataset and carry them across.**

The goal is to establish and validate a **methodology**: an experimental protocol
for genuine-only feature selection and honest authentication evaluation.

**THE METHODOLOGY CAN TRANSFER BETWEEN DATASETS. THE FEATURE NAMES AND FEATURE
VALUES DO NOT NEED TO TRANSFER AND MUST NOT BE ASSUMED TO.**

The two pipelines produce different features. Any selected feature list,
threshold, hyperparameter or fitted model from this dataset is valid only here.
What transfers is the protocol: how to split, what to control, what to
pre-register, how to aggregate, how to report uncertainty.

## STEPS — FOLLOW IN ORDER

**STEP 1.** Inspect and audit all available files. Report the actual directory
structure, file naming, row counts and column counts. Read files; do not guess.

**STEP 2.** Do **not** assume the description above is correct. Report every
discrepancy you find between the description and reality.

**STEP 3.** Identify and report exactly:
- owners/operators and their identifier column
- devices and their identifier column
- session counts per owner and per impostor
- genuine/impostor labels and the exact column and vocabulary used
- any existing training/testing split
- feature columns versus metadata columns
- all ID-like columns
- candidate leakage sources (session IDs, timestamps, ordering, counters,
  anything encoding the label indirectly)

**STEP 4.** Determine whether the dataset genuinely provides **same-device**
genuine/impostor comparisons. Build the person x device contingency table.
Report: how many profiles have owner and impostors on the same device; whether
any person appears on more than one device; whether the person-device graph is
connected. If the same-device claim is false, **say so plainly and stop**.

**STEP 5.** Audit whether a scientifically valid train/test authentication
methodology can be constructed. Specifically check whether people can be held out
whole, since impostors may recur across profiles.

**STEP 6.** Propose the methodology **before** running any large experiment.

**STEP 7.** Present: the proposed methodology; primary and secondary metrics;
success criteria; failure criteria; stop conditions; leakage rules; the exact
experiments you would run and their expected cost.

**STEP 8.** **Stop and wait for my approval.** Do not begin any feature-selection
search or expensive computation until I approve.

## HARD CONSTRAINTS — DO NOT VIOLATE

You must NOT:
- Assume this dataset's feature definitions match the current dataset's.
- Mix the old and current datasets in any analysis.
- Transfer selected feature names or values between datasets.
- Use test data for feature selection, ranking, thresholds, hyperparameters,
  model choice, feature counts, or deciding what to run next.
- Run thousands of feature combinations without explicit authorisation.
- Change or re-interpret previously frozen conclusions from the other dataset.
- Claim that same-device genuine/impostor separation proves a feature is purely
  behavioural. It proves the separation is **not purely a device effect** within
  that profile. That is a weaker and different claim.

## METHODOLOGY REQUIREMENTS (learned from measured failures — apply these)

1. **Split by PERSON, never by session.** Impostors may recur across profiles; a
   session split would put the same impostor in train and test. Hold out whole
   people in **both** owner and impostor roles.
2. **Aggregate PER OWNER, never by cohort mean.** In the previous dataset a
   configuration passing on cohort means satisfied only 2 of 8 owners
   individually. Declare "number of owners passing all constraints" as the
   primary quantity before searching.
3. **Pre-register every threshold against a measured NULL, before seeing
   results.** Previously, a metric that looked strong scored 0.783 on random
   features, which disqualified it. Measure what chance produces first.
4. **Report FAR with exact binomial confidence intervals and the attacker
   count.** FAR = 0 from 15 attackers is not FAR = 0 from 200. Cluster intervals
   on **attackers**, not sessions.
5. **Keep negative controls.** At minimum: an outlier control (synthetic
   far-from-owner sessions must be rejected) and a shuffled control (the owner's
   own sessions with each feature independently permuted must be rejected — this
   catches a model using only marginals). In the previous dataset these caught a
   model that scored 0.975 genuine acceptance while accepting 100% of shuffled
   sessions.
6. **Enforce the test-data rule mechanically, not by convention.** Install a
   Python audit hook that raises on any attempt to open a test file during
   development, and record every file actually read in the run's output.
7. **Keep an append-only experiment log.** Record void runs and bugs rather than
   deleting them. Never overwrite historical results.
8. **Use a fixed seed and verify reproducibility.**
9. **Checkpoint long runs incrementally.** A previous 2.5-hour run was lost
   because results were only written at the end.
10. **Distinguish validation from optimisation.** Optimisation happens on
    development folds. Validation is one shot on held-out people, after freezing.
    No re-selection afterwards.

## WHAT SUCCESS AND FAILURE WOULD MEAN

**Success:** a pre-registered, genuine-only methodology that rejects same-device
impostors for a majority of owners, with FAR intervals that actually support the
claim. This would mean the methodology is sound and the previous dataset's
structure was the limitation.

**Failure:** the methodology does not work even with same-device impostors and
adequate statistical power. **This would be the most important finding in the
project**, because it would redirect effort from data collection to redesigning
the behavioural representation itself.

Both outcomes are valuable. Report whichever you find, honestly, without
relaxing any pre-registered threshold to manufacture a pass.

## START HERE

Begin with STEP 1. Report what you actually find in the files before proposing
anything.

=====================================================================
