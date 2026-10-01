# TrackTheTrackers — workspace summary

Rank the 355 trackers for each of the 50k domains in `target.tsv`; the metric is
Recall@10 per domain, averaged over domains. All scores below are CV Recall@10
on the modelled sample S. They are comparable to each other, but **not** a
forecast of the target score. For that, see *Stratified recall and headline
range*.

## 1. Target shift: the sample was chosen after checking it

`target.tsv` has no labels, so the property the metric divides by (true tracker
count) cannot be seen directly. It was checked indirectly
(`data_exploration_2/3/4.py`, observables from `shift_observables.py`):

| design of the corpus comparison draw | adversarial AUC (target vs draw) | mean count | share of 1-tracker domains |
|---|---|---|---|
| U: uniform over tracked domains | **0.612** | 1.96 | 0.54 |
| size-biased, count^0.5 / ^1 / ^1.5 | 0.571 / 0.547 / 0.569 | 2.5 / 3.3 / 4.3 | 0.41 / 0.28 / 0.17 |
| T3: uniform over count >= 3 | 0.590 | — | 0 |
| T2 + size bias | 0.548 | — | 0 |
| 50% U + 50% T2 | 0.545 | — | 0.27 |
| **T2: uniform over count >= 2** | **0.503** (seeds 0.504 / 0.502) | 3.08 | 0 |
| target vs untracked domains | 0.811 | — | — |

- **Count model** (XGBoost Poisson on label-free observables, trained on the
  corpus, Spearman 0.45): the target's predicted counts sit far above the
  corpus. The covariate-shift read of this is a mean of 2.27 vs 1.96.
- **Label-shift inversion (BBSE)** puts 0–2% of the target on single-tracker
  domains. For the other buckets it matches the corpus's own ≥2 mix within
  bootstrap bands, with fit residual 0.0058 (T2) vs 0.0794 (U).
  Sanity check: BBSE recovers 0.2% singles on a known-truncated draw and 55.7%
  on a known-uniform one.
- **Decision:** S = a uniform draw over tracked domains with **≥2 trackers**:
  1.62M domains / 5.0M pairs (`common.SAMPLE_MIN_COUNT = 2`), picked by a hash of
  the domain_id.
- **Coverage parity after the choice** (`data_exploration_5.py`): every feature
  block matches between S and the target to about 0.003. For example,
  out-neighbour coverage is 0.673 vs 0.670 and mass is 16.53 vs 16.50. The
  uniform shadow set E, which includes singletons, is the one that differs.

## 2. Leakage design

- **Label pool P = corpus − S − E.** Every neighbour/sibling feature reads labels
  from P only. S and P are disjoint by construction, so no walk (including the
  return walk `d -> u -> d`) can hand a row its own labels. The target is
  outside the corpus and reads the same P, so parity is exact.
- **E (shadow evaluation set)** is a uniform draw over *all* tracked domains
  (206k), disjoint from S and removed from P. The harness never uses it; it
  exists so recall can be measured on buckets S does not contain.
- **Audit** (`data_exploration_5.py`, re-run for every new block): the
  standalone score is the same on S and on the disjoint E (gap at most 0.0015
  for every block, never near-perfect), and target coverage/mass parity holds.
  Fold std is small (3e-5 to 1.4e-4, from 3 folds of about 540k domains each).
  That is below a naive i.i.d. estimate, so fold std alone does **not** clear
  the pipelines. The check that does is **shadow E**: E's labels are never read
  by any feature or any training step, and the deployed model scores *as well
  or better* on E than the CV folds do, in every count bucket (section 6). A leak
  through S's own labels would show up as CV >> E.
- CV: `KFold(3, shuffle, 42)` over S domains. The metric is a plan scorer
  (`common.recall_at_10`, locked as `plan:recall_at_10`).

## 3. Leaderboard (CV Recall@10 on S)

| # | score | Δ vs parent (paired folds) | what |
|---|---|---|---|
| 01 | 0.68905 | — | popularity top-10 (no features) |
| 02 | 0.85866 | +0.170 | MLP 2x1024 softmax on 1-hop out/in labelled-neighbour tracker shares + host/TLD |
| 03 | 0.88466 | +0.0260 (+.0258/.0259/.0263) | + 2-hop co-citation (in>out) and coupling (out>in) |
| 04 | 0.88734 | +0.0027 (+.0028/.0027/.0026) | + direct links to each tracker's own domain |
| 05 | 0.88776 | ablation | drop-one-block grid (below) |
| 06 | 0.88776 | = 05 winner | + out>out, in>in 2-hop (promoted) |
| 07 | 0.86346 | −0.024 | pairwise XGBoost rank:ndcg over top-40 candidates — **rejected** |
| 08 | 0.88457 | −0.0032 vs 06 | RRF blend MLP + GBM — **rejected** |
| 09 | 0.88880 | +0.0010 (+.0011/.0011/.0010) | 5-seed MLP ensemble |
| 10 | 0.88985 | ablation | epochs x width x dropout x loss grid (below) |
| 11 | 0.89144 | +0.0026 vs 09 | 5-seed ensemble of ablation-2 winner (softmax, 16 ep, 2x2048, drop 0.3) |
| 12 | 0.89116 | −0.0003 | capacity push (24 ep, 2x3072, drop 0.4) — **rejected** |
| 13 | 0.89008 | −0.0014 | + wide linear skip path — **rejected** |
| 14 | 0.89128 | −0.0002 | + 1-hop per-tracker log-count channels — flat alone |
| 15 | 0.89176 | +0.0003 (+.0004/.0003/.0003) | 10-member mixed-config ensemble (5x 2048/0.3 + 5x 1024/0.1) |
| **16** | **0.89197** | +0.0002 (+.0002/.0002/.0002) | **FINAL**: 15's ten members + 5 members on 14's (+log-count) feature set |

**Ablation 1 (05), drop one block from 04's set:** −outin −0.0116, −inout −0.0068,
−host −0.0048, −out −0.0028, −direct −0.0027, −in −0.0015; adding out>out + in>in
gives +0.0004.

**Ablation 2 (10):** softmax (per-domain listwise) beats BCE in every matched
cell, by 0.002–0.01. 16 epochs beat 8 everywhere, and the wider net needs more
dropout. The best cell, 16/2048/0.3, scores 0.88985. The base cell (8/1024/0.1)
reproduces 06 exactly.

## 4. Measured and rejected (outside the leaderboard)

- **Hostname char 3-grams** (`data_exploration_7`): 0.7125 alone, but they
  *hurt* on top of graph+host (0.8541 vs 0.8581).
- **Hub-damped (IDF) 2-hop walks**: audit-clean, standalone slightly below the
  plain means (0.8361 vs 0.8384).
- **Full-TLD label prior + url-classification category**: audit-clean, but
  redundant. The prior alone scores 0.7425, about the same as the top-60 TLD
  one-hots (0.748); only 1.3% of rows are outside the top-60 TLDs, and url
  categories cover about 2%.
- **Company-level links** (`data_exploration_6`): precision 0.02. Tracker-level
  direct links (0.55) were kept.

## 5. Residual analysis (`data_exploration_9`)

- 48% of the misses are near misses, ranked 11–20.
- 12% are structurally unavoidable: domains with more than 10 trackers.
- 62% fall on mid-tail trackers (popularity rank 11–100, e.g. feedburner,
  sharethis, quantserve, and the paired Commission Junction beacons).
- Almost every missed pair has *some* 2-hop evidence. The limit is ranking
  sharpness, not coverage, which is why only ensembling and training moved the
  score after 06.

## 6. Stratified recall and headline range

Produced by `stratified_report.py pipeline_16`, which is analysis only and
never writes results.json. It uses two routes:

- **CV on S:** the harness's exact folds. It reproduces the harness score
  exactly (0.89197; folds 0.89203 / 0.89197 / 0.89191).
- **Shadow E:** the deployed model, fit once on all of S, scored on the
  disjoint uniform shadow set. This is the only route that covers
  single-tracker domains.

**Recall@10 by number of true trackers per domain (pipeline_16):**

| true trackers | n in S | recall, CV on S | n in E | recall, shadow E (±1 SE) |
|---|---|---|---|---|
| 1 | 0 | — | 111,413 | 0.9354 ± 0.0007 |
| 2 | 811,215 | 0.9204 | 47,685 | 0.9226 ± 0.0009 |
| 3 | 408,588 | 0.9005 | 24,025 | 0.9035 ± 0.0012 |
| 4 | 184,379 | 0.8709 | 10,640 | 0.8734 ± 0.0019 |
| 5 | 99,549 | 0.8512 | 5,813 | 0.8510 ± 0.0026 |
| 6–10 | 106,852 | 0.7564 | 6,232 | 0.7602 ± 0.0026 |
| >10 | 11,078 | 0.5196 | 656 | 0.5346 ± 0.0064 (ceiling 10/k) |

Shadow E runs about 0.002–0.003 above CV in every bucket. The deployed model
trains on all of S (1.62M) rather than 2/3 of it; a submission model would be
fit the same way, so E is the relevant number.

**Headline under named assumptions about the target's count mix**
(headline = Σ_k q_k · r_k, with r_k from shadow E; ±2 SE covers bucket sampling
only):

| assumption about the target's mix | evidence for / against it | share of 1-tracker | headline |
|---|---|---|---|
| **T2: uniform over count ≥ 2** | adversarial AUC 0.503; BBSE residual as good as the free fit | 0 | **0.8944 ± 0.0013** |
| BBSE point estimate (label shift) | fits the predicted-count histogram by construction | 0.008 | 0.8948 ± 0.0013 |
| COV: covariate shift (observables reweighted) | matches observables by construction; contradicts BBSE (it would keep 45% singles) | 0.447 | 0.9100 ± 0.0010 |
| U: uniform over tracked corpus | rejected: AUC 0.612 | 0.538 | 0.9165 ± 0.0010 |
| SB^1: size-biased (∝ count) | rejected: AUC 0.547, BBSE residual 0.043 | 0.277 | 0.8865 ± 0.0011 |

- **Range across all named assumptions: 0.8865 .. 0.9165.**
- **Range across the assumptions the evidence supports (T2, BBSE): 0.894 .. 0.895.**
  The BBSE 95% band allows up to 2.1% singletons, which moves the headline by
  at most about +0.001, so the supported range is **about 0.893 .. 0.896**.
- **What these numbers assume:**
  - recall conditional on the count bucket transfers from E to the target.
    This is supported by the S/target coverage parity in section 1, not proven.
  - The +0.0008 from reading the full label pool at prediction time (section
    7) is **not** included.
- **What the CV number alone would have said:** 0.8920. That's close to T2
  here, but only because S was built to match T2. A CV on a naive uniform
  sample would have reported about the U-mix headline, 0.916. That's
  approximate, since that model would also have been trained on a different
  mix. It would have overstated the target score by about 0.02.

## 7. Deployment note (for a later ml-submit run)

`data_exploration_8`: letting *prediction-time* features also read S's labels
(the target is disjoint from S) lifts shadow-E recall on domains with ≥2
trackers from 0.8904 to 0.8913 (+0.0008), with singletons unchanged. It is a
deliberate, measured break of train/predict parity. A submission should build
the target's features from the full corpus pool (P + S + E). No submission has
been generated.

## 8. Held-out result

`submission.tsv` was written by `final_pipeline.py`: pipeline_16 refit on all of
S, with target features built from the full-corpus label pool. It was scored
externally at **Recall@10 = 0.8963** on the 50,000 target domains.

- **Forecast:** T2 headline 0.8944 ± 0.0013, plus about 0.0008 from the full
  pool, gives about 0.8952. The supported range was about 0.893–0.896.
- **Actual vs forecast:** 0.8963 sits at the top of the supported range. That
  fits the BBSE band, which allowed up to 2% single-tracker domains in the
  target and would add up to about +0.001.
- **The rejected mixes were further off:** the uniform-over-tracked forecast
  (0.9165) would have been about 0.02 too high, and the size-biased one (0.8865)
  about 0.01 too low.
