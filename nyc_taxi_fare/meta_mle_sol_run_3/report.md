# Experiment report

Controller: llm; policy: draft-greedy; memory: window; model: openai/gpt-6.1-sol (reasoning medium); state: complete; elapsed: 1403 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_563dce85752f_variant_1 | expansion_ead10ec30182_variant_2 | ok | -3.9879203224413105 | {} |
| expansion_ead10ec30182_variant_2 | expansion_96322339ac3b_variant_1 | ok | -4.05279455937931 | {'objective_delta': 25} |
| expansion_96322339ac3b_variant_1 | expansion_c42110b389c6_variant_1 | ok | -4.05626531683786 | {} |
| expansion_7590e192fc19_variant_1 | expansion_96322339ac3b_variant_1 | ok | -4.058046487839249 | {} |
| expansion_aaea24a5d87f_variant_1 | expansion_96322339ac3b_variant_1 | ok | -4.062799905562255 | {} |
| expansion_dfcc28a60a5a_variant_1 | expansion_96322339ac3b_variant_1 | ok | -4.064013001008999 | {} |
| expansion_7e85057f11f8_variant_1 | expansion_96322339ac3b_variant_1 | ok | -4.072032568456261 | {} |
| expansion_c42110b389c6_variant_1 | expansion_378c44d400f7_variant_1 | ok | -4.074917733356847 | {} |
| expansion_af42d6d6dc5d_variant_1 | expansion_c42110b389c6_variant_1 | ok | -4.075904448749134 | {} |
| expansion_ead10ec30182_variant_1 | expansion_96322339ac3b_variant_1 | ok | -4.076702757955406 | {'objective_delta': 10} |
| expansion_b311e0bd10ff_variant_1 | expansion_96322339ac3b_variant_1 | ok | -4.098753515108967 | {} |
| expansion_afd6ca1cec8c_variant_2 | expansion_c42110b389c6_variant_1 | ok | -4.099189708102672 | {'robust_delta_usd': 10} |
| expansion_afd6ca1cec8c_variant_1 | expansion_c42110b389c6_variant_1 | ok | -4.1138812145274715 | {'robust_delta_usd': 5} |
| expansion_378c44d400f7_variant_1 | expansion_6cd5a5131664_variant_1 | ok | -4.1159388050578345 | {} |
| expansion_6cd5a5131664_variant_1 | root | ok | -4.214992910669326 | {} |
| expansion_c17b312b660b_variant_1 | root | ok | -4.312825457373585 | {} |
| expansion_bdd46429c253_variant_1 | root | ok | -4.654114242806983 | {} |
| expansion_dc4e7add9cb1_variant_1 | root | ok | -4.924339422005201 | {} |

## Evaluation audit

{
  "rows": 26000000,
  "folds": 3,
  "labels_missing": 0,
  "row_keys": "unique, nonmissing, aligned",
  "fold_sizes": [
    {
      "train": 25000000,
      "test": 333333
    },
    {
      "train": 25000000,
      "test": 333333
    },
    {
      "train": 25000000,
      "test": 333334
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Probes (out-of-fold predictions, not scored candidates)

- probe_0c3442bc5eb4 (expansion_c17b312b660b_variant_1): Using retained out-of-fold predictions without refitting, report fold RMSE and error contributions by fare range, trip distance, coordinate validity, year and passenger count. Identify the largest systematic biases and extreme-error cases. Also report exact test-date-hour-weighted RMSE with coverage, effective sample size and weight concentration. score -4.31283
- probe_eb5ca07d9fc6 (expansion_c42110b389c6_variant_1): Without refitting, report retained OOF fold RMSE, mean error, per-fold MAE, absolute-error quantiles and the largest squared-error cases with observed and predicted fares. Quantify how much total squared error comes from the largest 1% of errors and summarize underprediction versus overprediction. score -4.07492
- probe_5e679ad8b0fd (expansion_96322339ac3b_variant_1): Report the retained OOF fold scores and harness-computed regression error summary, including mean error, per-fold MAE, absolute-error quantiles and prediction preview, without refitting. Compare these summaries with the earlier ensemble probe to assess whether typical errors improved despite tied RMSE. score -4.05627
- probe_aeada8bdbb31 (expansion_563dce85752f_variant_1): Without refitting, report retained OOF fold scores, mean error, per-fold MAE, absolute-error quantiles and prediction preview. Compare with the previous probe to determine whether the residual correction improved typical errors as well as RMSE. score -3.98792

## Findings

- finding_5bce1df0d438: A reproducible full-source sample was reported, but the requested reusable training sample was not materialized and streaming was not documented. (Sampling and downstream reuse)
- finding_a2dc79337313: The sampled training data contain missing dropoff coordinates, geographically invalid coordinates, and passenger counts absent from the prediction set. (500,000 sampled training rows versus 9,914 test rows)
- finding_83f704a8f493: Both populations cover 2009–2015, but their temporal frequencies differ; inspected source segments mix years rather than following chronological order. (Timestamp coverage and source ordering)
- finding_8fec4827782b: Every tested population remains distinguishable from test inputs; the reported validity and fare filters do not resolve the observed distribution shift. (Reported three-fold raw-input adversarial comparisons; spread is maximum minus minimum fold AUC)
- finding_2eba2bdb7d72: Rounding coordinates to four decimal places does not remove adversarial separability. (Reported precision-normalization experiment)
- finding_37faa14c2bd6: Most fare filters alter very few rows, whereas the $5 lower cutoff removes 13.71% and materially changes the fare distribution. (Fare-selection effects within the sampled training population)
- finding_176e2319b0b9: No fare-based exclusion rule has demonstrated a fare-prediction benefit, particularly the substantial exclusion of fares below $5. (Evidence available for training-population selection)
- finding_2e68b551917b: Random CV spanning the observed date range is a reasonable provisional baseline rather than a future-only temporal holdout, but prediction-like validation and population selection remain unresolved. (Provisional validation strategy, not an experimentally established choice)
- finding_b81da03d9c5f: Timestamp inputs dominate the reported train-versus-test separability; coordinates and passenger count alone are nearly uninformative for this adversary. (Three-fold adversarial comparison using 9,914 candidate training rows and all 9,914 test rows)
- finding_81603fa8b691: Clock and calendar patterns both contribute, with seconds contributing mainly through interactions rather than strong marginal separation. (Reported timestamp feature ablations)
- finding_df153046b199: Observed date and clock frequencies differ substantially despite shared year coverage; a simple seconds-rounding explanation is insufficient. (500,000 sampled training rows versus 9,914 test rows)
- finding_070d16048157: Coordinate rounding does not resolve separation, and no coordinate recording artifact was demonstrated. (Four-decimal coordinate normalization)
- finding_618d901e2bd2: None of the tested validity or fare selections materially removes adversarial separability. (Balanced adversarial comparisons drawn from each filtered 500,000-row source sample; listed counts are retained population sizes)
- finding_b19713b381e3: Keeping all rows is a defensible least-restrictive provisional baseline for population studies, not a validated optimal fare-training population. (Provisional population choice pending common-validation fare RMSE)
- finding_c14c05cf2eb9: The follow-up documents a nested full-source sampling recipe, but does not materialize a reusable training sample or fully document row-level reproducibility of the adversarial comparisons. (Sampling and reproducibility deliverables)
- finding_619a6d1a90ac: One million rows achieved the lowest mean validation RMSE among the three measured training sizes. (Unfiltered nested learning curve using three fixed 20,000-row validation subsets)
- finding_0e6e747c8d3a: The final increase produced a positive but uncertain mean gain; neither saturation nor a reliably positive final gain is established. (Paired RMSE differences across three fixed validation subsets)
- finding_c0ce21aac724: The experiment specifies reproducible nested training and disjoint validation sampling, but does not supply a materialized reusable pool or complete preprocessing specification. (Learning-curve sampling and model documentation)
- finding_15259d68271e: Reported timings demonstrate inexpensive GPU fitting, but do not establish complete per-size experiment costs or the largest affordable pool. (Timing and throughput evidence for the measured learning curve)
- finding_4c1f617e7ca8: One million rows is a reasonable provisional measured baseline, not a demonstrated optimal or largest affordable training pool. (Pool-size choice limited to this model and random-source validation design)
- finding_47e6797125ca: Ten million training rows achieved the lowest mean RMSE in the expanded learning curve. (Unfiltered nested training pools evaluated on three fixed 20,000-row validation subsets in this study)
- finding_b406eb7ee674: The final paired gain is positive but uncertain under the study's two-standard-error rule; saturation is not established. (Adjacent-size paired RMSE improvements across three fixed validation subsets)
- finding_8a6faca31010: The study finished within 1,500 seconds, but reported per-size times still exclude shared timestamp and geographic feature construction. (Measured runtime and timing completeness)
- finding_14cee24d7e5b: Approximately 27.36 million rows are projected affordable, but that projection is not a measured runtime or memory guarantee. (Same model and hardware, assuming conservative linear runtime scaling)
- finding_5c333cf4061b: Ten million rows is the updated provisional pool recommendation, rather than a demonstrated optimum. (This CUDA regressor and random-source validation design)
- finding_1aba51183658: Sampling, feature families and model seed are documented, but no reusable sampled-pool artifact or fully exact feature implementation is supplied. (Reproducibility of the expanded study)
- finding_8850987dbaa3: The curve artifact has an anomalous reported shape and visible empty rows. (Output artifact quality)
- finding_1ef0e24d0aa6: Test pickup timestamps are highly repeated and concentrated in particular date-hour cells; this establishes clustering, not its causal origin or duplicate trips. (Uniform 2,000,000-row training sample versus all 9,914 test rows)
- finding_9029ac605baf: Timestamp support restriction and date-hour matching leave substantial adversarial separability. (Three stratified shuffled adversarial folds, seed 714; 180-tree LightGBM; 9,914 sampled candidate rows per rule versus test, excluding identifiers and fare)
- finding_3e6ddb243763: The additional fare-threshold comparisons again fail to remove adversarial separation. (Same balanced adversarial design, with retained counts referring only to the 2,000,000-row training sample)
- finding_873bb1034008: Date-hour-supported training worsened fare RMSE on every common validation subset; date-supported training had mixed effects. (CUDA squared-error boosting on identical disjoint validation rows supported by test date-hours)
- finding_72c94ec5bf65: The common validation comparison has limited documented coverage and does not establish representative test performance. (Coverage and reproducibility limitations of the timestamp-restriction study)
- finding_9c4e5bd39617: Unrestricted training with ordinary and test-date-hour-weighted validation RMSE is a provisional strategy, not a validated solution to timestamp shift. (Validation design preserving the large training pool and all fare values)
- finding_3d265317b553: The tested fare-per-distance selections do not explain away the train–test shift; several restrictive rules make populations much more distinguishable. (300,000-row full-source random study sample; balanced adversarial comparisons against all 9,914 test rows)
- finding_f17b50bb1753: Cross-fitted fare-residual restrictions also leave substantial separability; small AUC differences do not establish improvement. (Residual-based population diagnostics within the same 300,000-row sample)
- finding_e746caab2a9a: Fare-only and timestamp-support baselines remain distinguishable from test inputs in this study. (Same balanced adversarial design; retained counts refer to the 300,000-row sample, not the full source)
- finding_db1840955a13: No population restriction tested here has demonstrated a fare-prediction benefit. (Predictive justification for fare, ratio, residual and timestamp exclusions in this result)
- finding_9deaead16d2e: A million-row validation sample nearly covers test date-hour mass, but weighting sharply reduces effective sample size. (Independent 1,000,000-row coverage sample with exact UTC date-hour weighting)
- finding_d3f576680424: Preserving unrestricted training and reporting both ordinary and date-hour-weighted RMSE remains a defensible provisional validation strategy, not a validated correction for the shift. (Population and validation recipe supported by this diagnostic study)
- finding_0de46192abb3: The study documents comparable label-free adversarial experiments and finishes within the requested bound, but exact model reproduction remains incomplete. (Experimental comparability, reproducibility and runtime)
- finding_985d0562c73b: The requested OOF ensemble analysis was not performed because candidate prediction artifacts were inaccessible under the plan contract. (Artifact-only analysis of the parent, randomized forest, CUDA XGBoost and standalone neural network)
- finding_75c8aa959626: The parent has lower reported RMSE on every fold than each of the other three families, but these scores do not establish ensemble complementarity. (Previously recorded locked-fold scores, reproduced rather than recomputed)
- finding_bd9a209ca8fd: All three two-family ensembles are projected below 7,200 seconds, but complete runtime compliance is unverified. (Three-fold runtime extrapolation from recorded mean fit and score times)
- finding_429f0388f895: A predetermined equal-weight parent–forest blend is a provisional next experiment, not an established improvement. (Ensemble selection on the existing locked folds)

All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
