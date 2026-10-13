# Experiment report

Policy: greedy; model: gemini/gemini-3.8-flash (reasoning high); state: complete; elapsed: 213 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_54b35629f692_variant_2 | expansion_89fba5c987f2_variant_2 | ok | 0.9623969999339327 | {'regularization_and_schedule': 'l1_regularized_500'} |
| expansion_c4cbf7f8e47c_variant_2 | expansion_54b35629f692_variant_2 | ok | 0.9623787499293703 | {'tree_structural_regularization': 'path_smoothed_leaf100'} |
| expansion_89fba5c987f2_variant_2 | expansion_ca1dc827521d_variant_1 | ok | 0.9623784999306203 | {'regularized_capacity': 'extended_schedule_500'} |
| expansion_54b35629f692_variant_3 | expansion_89fba5c987f2_variant_2 | ok | 0.9623747499488079 | {'regularization_and_schedule': 'extended_schedule_750'} |
| expansion_54b35629f692_variant_1 | expansion_89fba5c987f2_variant_2 | ok | 0.9623627499296828 | {'regularization_and_schedule': 'feature_bagging_500'} |
| expansion_051fe74c819f_variant_3 | expansion_54b35629f692_variant_2 | ok | 0.9623502499154952 | {'feature_set': 'full_topographic_interactions'} |
| expansion_89fba5c987f2_variant_1 | expansion_ca1dc827521d_variant_1 | ok | 0.9623204999178078 | {'regularized_capacity': 'reg_300'} |
| expansion_051fe74c819f_variant_2 | expansion_54b35629f692_variant_2 | ok | 0.9623174999442453 | {'feature_set': 'hydrology_elevation_and_counts'} |
| expansion_c4cbf7f8e47c_variant_1 | expansion_54b35629f692_variant_2 | ok | 0.9623139999617454 | {'tree_structural_regularization': 'depth_constrained_14'} |
| expansion_89fba5c987f2_variant_3 | expansion_ca1dc827521d_variant_1 | ok | 0.9623104999186828 | {'regularized_capacity': 'scaled_capacity_384'} |
| expansion_051fe74c819f_variant_1 | expansion_54b35629f692_variant_2 | ok | 0.9617479999505578 | {'feature_set': 'benchmark_raw'} |
| expansion_c4cbf7f8e47c_variant_3 | expansion_54b35629f692_variant_2 | ok | 0.9615509999508077 | {'tree_structural_regularization': 'extra_trees_bagged'} |
| expansion_ca1dc827521d_variant_1 | expansion_300b701aa2d4_variant_3 | ok | 0.9560417490921198 | {'feature_ablation': 'parent_all'} |
| expansion_300b701aa2d4_variant_3 | expansion_672d8160cf2e_variant_3 | ok | 0.9547187488575571 | {'model_capacity': 'capacity_255_300'} |
| expansion_300b701aa2d4_variant_1 | expansion_672d8160cf2e_variant_3 | ok | 0.9546347490173072 | {'model_capacity': 'capacity_127_200'} |
| expansion_ca1dc827521d_variant_2 | expansion_300b701aa2d4_variant_3 | ok | 0.954070998953807 | {'feature_ablation': 'categorical_and_solar'} |
| expansion_ca1dc827521d_variant_3 | expansion_300b701aa2d4_variant_3 | ok | 0.953471748880307 | {'feature_ablation': 'full_engineered'} |
| expansion_300b701aa2d4_variant_2 | expansion_672d8160cf2e_variant_3 | ok | 0.9528919988671819 | {'model_capacity': 'capacity_255_200'} |
| expansion_672d8160cf2e_variant_3 | expansion_a3b56014eced_variant_3 | ok | 0.9497449987537441 | {'feature_set': 'all'} |
| expansion_a3b56014eced_variant_3 | expansion_74e721d8fb49_variant_1 | ok | 0.9434589982301809 | {'subsample_fraction': 'subsample_1.0'} |
| expansion_672d8160cf2e_variant_2 | expansion_a3b56014eced_variant_3 | ok | 0.9432467480078058 | {'feature_set': 'physical'} |
| expansion_a3b56014eced_variant_1 | expansion_74e721d8fb49_variant_1 | ok | 0.9432319994060556 | {'subsample_fraction': 'subsample_0.2'} |
| expansion_672d8160cf2e_variant_1 | expansion_a3b56014eced_variant_3 | ok | 0.938853997380618 | {'feature_set': 'baseline'} |
| expansion_a3b56014eced_variant_2 | expansion_74e721d8fb49_variant_1 | ok | 0.9379012477823677 | {'subsample_fraction': 'subsample_0.5'} |
| expansion_74e721d8fb49_variant_1 | root | ok | 0.9357339951574307 | {} |

## Evaluation audit

{
  "rows": 4000000,
  "folds": 3,
  "labels_missing": 0,
  "row_keys": "unique, nonmissing, aligned",
  "fold_sizes": [
    {
      "train": 2666666,
      "test": 1333334
    },
    {
      "train": 2666667,
      "test": 1333333
    },
    {
      "train": 2666667,
      "test": 1333333
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Probes (out-of-fold predictions, not scored candidates)

- probe_63c007508ddf (expansion_89fba5c987f2_variant_2): What is the out-of-fold error distribution across the 7 Cover_Type classes for top candidate expansion_89fba5c987f2_variant_2, specifically quantifying mutual misclassification between dominant classes 1 and 2 versus error rates on minority classes? score 0.96238

## Findings

- finding_3834d7146b62: Cover_Type is heavily imbalanced across the 7 classes, dominated by classes 1 and 2 (~93.2% combined), while class 4 has only 377 rows and class 5 is a singleton with exactly 1 row. (train)
- finding_a10115c9930b: Standard 3-fold StratifiedKFold cannot allocate class 5 across all 3 folds due to the single observation; CV splitting will require dropping or reassigning class 5, or using non-stratified/custom splitting. (train)
- finding_ab41c08a1e89: Soil_Type7 and Soil_Type15 are uninformative constant zero columns in both the training and test sets and can be safely dropped. (dataset)
- finding_a2e0d45b0ced: There are no missing values in train (4M rows) or test (1M rows), and all features are integer-encoded (int64). (dataset)
- finding_ff6a6231983f: Scikit-learn's 3-fold StratifiedKFold executes successfully on the raw 4,000,000-row train set despite Cover_Type=5 having only 1 sample, placing it into fold 0 (fold sizes 1,333,334, 1,333,333, and 1,333,333), while dropping the single class 5 observation yields exactly equal folds of 1,333,333. (train)
- finding_d990edbeb672: Both train and test feature distributions exhibit synthetic data artifacts including negative distances, negative angles, and hillshade indices outside the standard [0, 255] range. (dataset)
- finding_93f5b597a2bc: There are zero duplicate feature rows in train (4M rows) or test (1M rows). (dataset)
- finding_086237b4dcf2: A 1,000,000-row deterministic subsample of train accurately mirrors the full 4,000,000-row dataset across numerical features, wilderness areas, and target class proportions. (train)
- finding_355562972d2a: There is a moderate covariate shift between train and test in Wilderness Area distributions, notably with Wilderness_Area3 dropping from 65.4% in train to 47.3% in test, and Wilderness_Area4 rising from 2.2% to 5.7%. (dataset)

All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
