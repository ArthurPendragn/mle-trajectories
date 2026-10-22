# Experiment report

Controller: llm; policy: draft-greedy; memory: window; model: openai/gpt-6.1-sol (reasoning medium); state: complete; elapsed: 477 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_8e1aa38eee57_variant_1 | expansion_24982fb3665b_variant_1 | ok | -3.7450967376457744 | {} |
| expansion_24982fb3665b_variant_1 | expansion_3e75d0c37db2_variant_1 | ok | -3.7629798395513157 | {} |
| expansion_3e75d0c37db2_variant_1 | expansion_eb14f5dd4390_variant_1 | ok | -3.7850616793490452 | {} |
| expansion_eb14f5dd4390_variant_1 | expansion_47e92d568194_variant_1 | ok | -3.8040879621735804 | {} |
| expansion_47e92d568194_variant_1 | root | ok | -3.805607332312004 | {} |
| expansion_5973910a700b_variant_1 | root | ok | -3.8100183329199617 | {} |
| expansion_bbcf99bb7215_variant_2 | root | ok | -3.836474918226321 | {'route_regime_features': 'augmented'} |
| expansion_6141edcc2283_variant_1 | expansion_24982fb3665b_variant_1 | ok | -3.8834076197382363 | {} |
| expansion_bbcf99bb7215_variant_1 | root | ok | -3.896224405307279 | {'route_regime_features': 'basic'} |
| expansion_1976545741a1_variant_1 | root | ok | -3.925783810585461 | {} |
| expansion_603f447a707d_variant_1 | root | ok | -4.284099376461287 | {} |
| expansion_f87b23d7f09f_variant_1 | expansion_24982fb3665b_variant_1 | failed | None | {} |
| expansion_ff11da935d6e_variant_1 | expansion_8e1aa38eee57_variant_1 | failed | None | {} |
| expansion_f9fd29b02c94_variant_1 | expansion_8e1aa38eee57_variant_1 | failed | None | {} |
| expansion_2d9bae9e3f2e_variant_1 | expansion_8e1aa38eee57_variant_1 | failed | None | {} |

## Evaluation audit

{
  "rows": 8360000,
  "folds": 3,
  "labels_missing": 0,
  "row_keys": "not declared",
  "fold_sizes": [
    {
      "train": 8300000,
      "test": 20000
    },
    {
      "train": 8300000,
      "test": 20000
    },
    {
      "train": 8300000,
      "test": 20000
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Probes (out-of-fold predictions, not scored candidates)

- probe_11f567e447fc (expansion_5973910a700b_variant_1): Using retained out-of-fold predictions without refitting, report fold RMSE and error contributions by fare range, trip distance, pickup year, airport versus other trips, and coordinate validity. Show signed residual summaries and the largest squared-error cases to distinguish systematic pricing errors from rare anomalies. score -3.81002
- probe_42a4f4065068 (expansion_8e1aa38eee57_variant_1): Without refitting, report the leading candidate's retained fold RMSE, mean signed error, absolute-error quantiles and per-fold mean absolute error. Compare these summaries with the baseline probe to check whether typical errors improve alongside RMSE. score -3.74510

## Findings

- finding_34cc2b734c52: Sampling is reproducible and covers the full source, but sampled rows were not cached as requested. (Exploration sampling and reuse)
- finding_dce0839bc5f2: The inspected sample and test have no reported missing values, but training contains severe coordinate and passenger-count anomalies absent from test. (400,000 sampled training rows and 9,914 test rows; not a full-source missingness audit)
- finding_f90b7e960ee1: Fares are right-skewed with rare negative values and a substantial legitimate-looking upper tail; magnitude alone does not establish recording error. (400,000-row training sample)
- finding_cb721f9d208d: Test rows are strongly clustered by exact pickup timestamp, despite having no exact duplicate inputs. (Timestamp and duplicate structure in inspected data)
- finding_f86da86d76ce: The source is not strictly chronological, but broader source-order effects were not quantified. (Displayed source head and tail only)
- finding_6d2d56841a5a: All tested populations remain distinguishable from test; filtering does not eliminate the observed distribution difference. (Reported three-fold adversarial comparisons; fractions relative to 350,000 fitting rows)
- finding_3daca73497fe: Input-validity filtering and removal of negative fares improved RMSE on the shared validation subset; aggressive upper-tail truncation offered little consistent additional benefit. (Single shared input-valid holdout from the final 50,000 sampled rows; exploratory model results only)
- finding_b8623254f6ef: The reported input-valid rule cannot yet be reproduced exactly, and the adversarial results do not isolate removable recording differences from genuine population selection. (Limitations of returned exploration artifacts)
- finding_8a7ee0c91251: A reproducible replacement input-valid rule is now defined, but the previous undocumented rule was not recovered. (Current 400,000-row full-source sample, seed 1971; first 300,000 fitting rows)
- finding_411a92e282d9: Timestamp-grouped adversarial evaluation finds little distinguishability even before normalization; these results do not establish recording normalization as the explanation for the earlier high AUC. (Balanced 9,914 rows per domain; LightGBM with 140 trees, 15 leaves, learning rate 0.06, minimum leaf 60; three stratified exact-timestamp-grouped folds)
- finding_f2c13f43f9de: No tested population has a compelling normalized adversarial advantage over retaining all rows. (Current normalized adversarial comparisons; counts relative to 300,000 fitting rows)
- finding_01a1450526a6: Two paired holdouts do not reproduce a consistent benefit from input-validity filtering; removing fares above $100 consistently worsens RMSE. (Two disjoint shared input-valid validation subsets; current exploratory fare models)
- finding_16dbdb713462: Precision summaries show broadly similar recording characteristics, while substantial timestamp clustering remains a genuine structural difference. (400,000 sampled training rows and all 9,914 test rows)
- finding_ad5a05ea4b46: Retaining the unfiltered population is the least restrictive provisional choice; stronger exclusions are not supported by these comparisons. (Population choice for subsequent modeling, not a locked evaluation decision)
- finding_8f9ad824933c: The learning curve uses nested full-source random samples and fixed timestamp-disjoint validation sets without population filtering. (Reported 100k, 300k and 1m training-row experiments)
- finding_fc6282818eec: Increasing training volume to one million rows improves RMSE on all three fixed validation subsets; the measured curve does not show saturation. (CUDA XGBoost squared-error model: 600 trees, depth 8, learning rate 0.06, minimum child weight 20, subsample 0.85, column subsample 0.9, lambda 10)
- finding_bb160fd23b6c: Reported experiment timings omit shared feature construction and source decoding, so they are not complete per-size runtime measurements. (Runtime evidence from this learning-curve exploration)
- finding_fd5436eb3194: The requested largest affordable training-pool recommendation remains unresolved. (Completeness of the learning-curve result and subsequent experiment budgeting)
- finding_5dd35434c789: One million training rows is a supported measured baseline, but not an established budget-optimal ceiling. (Provisional size choice for subsequent modeling)
- finding_d23b5c385652: The updated learning curve improves every holdout at both size increases and shows no measured saturation through one million training rows. (Unfiltered CUDA fare-model study with three fixed 20,000-row holdouts)
- finding_8dc46b45013b: Feature-inclusive experiment timings are now available, although sampling and timestamp filtering remain outside the measured per-size runtime. (Runtime measurements for 100k, 300k and 1m training rows)
- finding_84351478215b: The returned protocol specifies reproducible geographic and calendar features and preserves timestamp-disjoint validation. (Updated learning-curve sampling, feature construction and validation design)
- finding_d35d26f0d41b: Memory measurements do not establish size-specific host peaks or GPU feasibility at the extrapolated training size. (Memory evidence and limitations of the size recommendation)
- finding_1ec9fdceb8bb: An 8.3-million-row shared training pool is a numerical budget-supported candidate, not a demonstrated feasible or optimal ceiling. (Extrapolated subsequent-experiment sizing under the stated runtime and memory budgets)
- finding_b982040ab20b: The requested residual audit was not completed; no stratum RMSE, signed bias, squared-error contributions or error-supported modeling recommendation can be inferred from these outputs. (Current no-refit exploration of the leading candidate)
- finding_910d8f10d7f9: Coordinate-invalid rows occur consistently in validation but are absent from test under the reported geographic rule. (Three 20,000-row validation holdouts and 9,914 test rows)
- finding_c14fd8490df7: Near-zero and over-40-km recorded distances are more prevalent in validation than test; distance strata include coordinate anomalies. (Displayed distance coverage summaries)
- finding_bfc1f5df2564: Airport strata have recurring validation support and roughly comparable test coverage, although EWR support is sparse. (Airport involvement defined as either endpoint within 3 km of an airport center)
- finding_fae722e4a1af: Calendar coverage differs between validation and test, but the outputs do not establish conditional prediction bias. (Reported year summaries across three folds and displayed fold-0 hour summaries)
- finding_f06efaa44c25: Extreme fares include both plausible long trips and inconsistent-looking coordinate/fare combinations; they are not identified largest prediction errors or confirmed label errors. (Displayed extreme-label cases from fixed validation rows)

All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
