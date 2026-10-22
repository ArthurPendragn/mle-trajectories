# Experiment report

Controller: llm; policy: draft-greedy; memory: window; model: openai/gpt-6.1-sol (reasoning medium); state: complete; elapsed: 459 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_4cbe04701631_variant_1 | expansion_915dc3f137ff_variant_1 | ok | -4.021321967308652 | {} |
| expansion_1f868c628b8f_variant_1 | expansion_4cbe04701631_variant_1 | ok | -4.022662078840738 | {} |
| expansion_2580aa386477_variant_1 | expansion_4cbe04701631_variant_1 | ok | -4.022979878641709 | {} |
| expansion_915dc3f137ff_variant_1 | root | ok | -4.024716244896186 | {} |
| expansion_73b6983c5a03_variant_1 | root | ok | -4.085957510543336 | {} |
| expansion_05dc2cd1de17_variant_1 | expansion_4cbe04701631_variant_1 | ok | -4.087285473220766 | {} |
| expansion_9e4598dda11b_variant_1 | expansion_4cbe04701631_variant_1 | ok | -4.12699429045594 | {} |
| expansion_abb87964f60b_variant_1 | root | ok | -4.2104272399999125 | {'training_objective': 'RMSE'} |
| expansion_abb87964f60b_variant_2 | root | ok | -4.379162570200491 | {'training_objective': 'Huber:delta=10'} |
| expansion_71a0d6ea49aa_variant_1 | root | ok | -4.526562999616703 | {} |
| expansion_9dc44d9ec695_variant_1 | expansion_4cbe04701631_variant_1 | ok | -4.705854266839018 | {} |
| expansion_be9081879c99_variant_1 | root | ok | -4.916248549032444 | {} |
| expansion_a9e6b46cc092_variant_1 | expansion_4cbe04701631_variant_1 | ok | -6.379132444010704 | {} |

## Evaluation audit

{
  "rows": 16090000,
  "folds": 3,
  "labels_missing": 0,
  "row_keys": "unique, nonmissing, aligned",
  "fold_sizes": [
    {
      "train": 15999388,
      "test": 30151
    },
    {
      "train": 15999388,
      "test": 30262
    },
    {
      "train": 15999388,
      "test": 30199
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Probes (out-of-fold predictions, not scored candidates)

- probe_05c0c8e56758 (expansion_915dc3f137ff_variant_1): Using retained out-of-fold predictions without refitting, what explains fold RMSEs of 3.79, 4.71 and 3.58? Report error concentration and signed bias by fare range, ride distance, coordinate validity, airport proximity and calendar period, including the largest residuals and each group's contribution to squared error. score -4.02472
- probe_a6645f2f63ba (expansion_4cbe04701631_variant_1): Without refitting, report the selected candidate's fold RMSE, MAE, signed bias, absolute-error quantiles and squared-error concentration in the largest 1% of residuals. Compare available summaries with the baseline probe to assess whether the tied RMSE conceals materially different typical-ride or tail errors. score -4.02132
- probe_f7a37e5c9644 (expansion_05dc2cd1de17_variant_1): Without refitting, report retained OOF fold scores, signed bias, absolute-error quantiles and per-fold MAE for the two-regime candidate. Compare these with the selected candidate's existing summaries to determine whether their tied RMSE conceals different typical-ride errors. score -4.08729
- probe_7fea3472c091 (expansion_2580aa386477_variant_1): Without refitting, report retained OOF fold scores, signed bias, absolute-error quantiles and per-fold MAE for the route-context candidate. Compare with the selected candidate's existing summaries: does their tied RMSE conceal better typical-ride accuracy? score -4.02298

## Findings

- finding_38d6a9a6e958: The sampled training data contain missing dropoff coordinates, impossible coordinates, and passenger counts absent from test; all test rows pass the reported observable-validity check. (250,000 sampled training rows and all 9,914 test rows; not a full-source missingness audit.)
- finding_a354f99d6d57: Source-block summaries show mixed years and broadly stable fare means, rather than obvious chronological ordering, but do not establish that a prefix is representative. (Reported source-block summaries.)
- finding_b088ee2a3623: Test timestamps are much more concentrated than sampled training timestamps, providing a plausible source of adversarial separability. (Timestamp distributions in the inspected train sample and complete test set.)
- finding_138e885e1434: Fares are strongly right-skewed, with rare nonpositive values and a small high-fare tail. (250,000 sampled training targets.)
- finding_9a77399e8254: Every tested population remains meaningfully distinguishable from test; observable-validity filtering has the lowest mean AUC in both raw and rounded comparisons, but does not resolve the shift. (Reported three-fold adversarial comparisons; source counts below are estimates except the unfiltered total.)
- finding_21e55fd61349: The retention counts quantify exactly which sampled fare ranges each reported rule removes. (Sample-level fare retention for all reported rules; source extrapolation assumes representative sampling.)
- finding_bee71c4288de: Reported coordinate rounding does not materially reduce adversarial separation; removable recording differences have not been comprehensively isolated. (Raw-versus-rounded comparisons and coordinate residual summaries.)
- finding_32027f15ef6e: Use a documented, broad observable-validity population as a provisional baseline, retaining the ordinary fare range and testing target exclusions through RMSE rather than AUC alone. (Proposed population choice; not a validated predictive improvement.)
- finding_a3ea6d3398a0: Timestamp coarsening substantially reduces reported adversarial separation, but does not eliminate it. (Reported three-fold ablations on 250,000 sampled training rows and 9,914 test rows.)
- finding_63d56ccb3a0d: Reported separation is concentrated in temporal features rather than locations or passenger counts. (Feature groups named in the returned adversarial ablations; exact feature definitions and classifier configuration are unavailable.)
- finding_c0c709670143: Test timestamps exhibit pronounced concentration beyond a simple difference in the fraction recorded at second zero. (Current 250,000-row training sample and complete test set.)
- finding_91968773cef9: Temporal population selection or timestamp clustering is a plausible residual explanation; recording resolution alone is insufficient, but genuine ride-population selection is not established. (Interpretation of the reported timestamp distributions and feature ablations.)
- finding_88434266740c: After normalization, no tested fare or validity filter resolves the shift, and the apparent benefit of validity-only filtering is no longer consistent. (Current sample population comparisons; estimated full-source counts assume representative sampling.)
- finding_a06ca1e7300e: The returned fare-band counts identify precisely which fare ranges the tested filters remove in the current sample. (Current sample only; these counts differ from the earlier exploration and should not be merged with it.)
- finding_3844e3bc0b1d: Retaining the unfiltered population is a defensible provisional choice for shift matching; target exclusions or validity cleaning should be justified separately through RMSE and data-quality checks. (Population recommendation, not a demonstrated predictive improvement or formal equivalence test.)
- finding_5228963931c3: The result does not provide the requested reproducibility documentation or an implemented representative sampling strategy. (Returned exploration outputs.)
- finding_a71f02b54c66: Validation RMSE improves at every measured training size on all three fixed validation subsets. (Unfiltered-population CUDA XGBoost learning curve; 30,000 validation rows per subset.)
- finding_27b635f17c84: Adjacent paired gains diminish but remain positive; the final measured gain provides no evidence of a plateau. (Paired comparisons across the three fixed validation subsets, conditional on the sampled training pool.)
- finding_e563d9846667: The documented sampling procedure uses the full source and separates all validation rows from nested training samples. (Procedure reported in this exploration; no persisted pool artifact supplied.)
- finding_ee1001105306: The curve uses an explicitly GPU-enabled learned regressor with location and calendar features, but preprocessing is only partially specified. (Model and feature documentation for this curve.)
- finding_b4a48e559176: The reported runtime ceiling is optimistic and does not establish the largest affordable end-to-end training pool. (Runtime extrapolation for twelve experiments within approximately 28,000 seconds.)
- finding_9dd787d8173c: Two million rows is the strongest measured starting pool; larger pools warrant investigation rather than choosing a smaller pool on plateau grounds. (Provisional pool recommendation for this model, not a demonstrated optimal or maximum affordable size.)
- finding_4a5d400ed1a8: A reproducible sampling recipe is supplied, but the requested pool-materialization strategy and feature-inclusive size timings remain incomplete. (Deliverables returned by this exploration.)
- finding_f387fecc7ca3: RMSE decreases on every validation subset at both measured size increases; 8 million rows is the best measured size. (Current unfiltered learning curve with three 30,000-row validation subsets; not directly interchangeable with the earlier curve.)
- finding_477e6112a3fd: Both adjacent paired gains are positive, but the final gain has substantial uncertainty and does not establish either a plateau or a statistically conclusive improvement. (Three paired validation-subset comparisons conditional on the sampled training pool.)
- finding_140da3daf2c2: The reported sampling recipe retains all labels and excludes every validation row from all nested training samples, but exact reuse of earlier validation identities is unverified. (Sampling documentation for this exploration.)
- finding_0729c2364972: Feature-inclusive recurring timings are supplied, but separate component timings and a measured one-time preparation cost are absent. (Returned timing artifacts.)
- finding_ffefcc7a1806: The approximately 17.21-million-row affordability ceiling is a conditional extrapolation, not a measured runtime or memory guarantee. (Twelve-experiment budget projection using 28,000 seconds.)
- finding_3b3b155ea51b: Eight million rows is a defensible measured starting pool; larger pools require runtime and memory verification rather than assuming the extrapolated ceiling is affordable. (Provisional pool recommendation for the current implementation.)
- finding_9f174ac14bf3: Exact model reproducibility and durable pool reuse remain incomplete in the returned deliverables. (Current exploration documentation.)
- finding_d7675864d068: Holding out entire date-hours sharply reduces adversarial separation compared with ordinary cross-validation. (Unfiltered 250,000-row training sample versus test; three folds per splitting method.)
- finding_5f57f147eb3d: The results support timestamp-cluster memorization as a major source of ordinary adversarial separation, rather than establishing generalizable temporal population selection. (Interpretation of normalized adversarial comparisons, not proof about the test-generation mechanism.)
- finding_2bad583113ba: Temporal restrictions reduce ordinary AUC only modestly relative to their substantial training-row cost; supported years remove no sampled rows. (Current sample; source counts are extrapolations, not exact full-source retention counts.)
- finding_bf619a7c3c26: Fare and validity filters do not consistently improve ordinary separation, while their grouped AUCs are generally near chance. (Current sample comparisons; no predictive RMSE evaluation.)
- finding_7bc9fe5c7a36: Temporal filters remove observations across the fare distribution, not just anomalous targets. (Current sample target-range retention.)
- finding_67690d0cd9f9: Sampling, validity and normalized features are now documented, but exact classifier reproduction and paired cross-rule uncertainty remain unresolved. (Returned methodology documentation.)
- finding_5e5a2d54cf88: Retain all rows provisionally and use date-hour-grouped adversarial validation as a sensitivity check; temporal filtering needs predictive evidence before adoption. (Population and validation recommendation, not a formally established near-best equivalence result.)
- finding_43d497962450: The requested OOF ensemble diagnostics were not computed; the exploration reports that access to retained prediction files was blocked by its source-only contract. (This exploration's returned diagnostics, despite its overall status of ok.)
- finding_23206cdaa0a4: XGBoost has lower reported RMSE on every fold than the other three models, but aggregate scores cannot establish ensemble complementarity. (Previously reported locked-fold candidate summaries; not newly recomputed or row-aligned.)
- finding_d58a8248641f: Additive three-fold fit-time estimates exceed the 1800-second attempt limit for both proposed ensembles. (Runtime extrapolation from reported mean base-model fit times, not measured ensemble execution.)
- finding_5d3a7d059676: Retaining the selected single model is provisional; no deployable blend is established. A future blend could estimate constrained weights on internal date-hour holdouts without using outer validation labels. (Future ensemble design, requiring authorized diagnostics and a feasible runtime budget.)

All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
