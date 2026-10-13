# Experiment report

Policy: greedy; model: gemini/gemini-3.8-flash (reasoning high); state: complete; elapsed: 226 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_0f878ed4778a_variant_1 | expansion_01b30a5e21e5_variant_1 | ok | 0.9625839999176828 | {'model': 'lgbm_1000_leaves31_lr005'} |
| expansion_d0dcdd20699b_variant_1 | expansion_0f878ed4778a_variant_1 | ok | 0.9625639999329328 | {'model': 'lgbm_1100_leaves35_lr0045'} |
| expansion_d0dcdd20699b_variant_2 | expansion_0f878ed4778a_variant_1 | ok | 0.9625622499581828 | {'model': 'lgbm_1100_leaves45_lr0045'} |
| expansion_74bfaf9be2f3_variant_1 | expansion_0f878ed4778a_variant_1 | ok | 0.9625549999443704 | {'model': 'lgbm_1250_leaves35_lr004'} |
| expansion_74bfaf9be2f3_variant_2 | expansion_0f878ed4778a_variant_1 | ok | 0.9625519999342455 | {'model': 'lgbm_1250_leaves42_lr004'} |
| expansion_0f878ed4778a_variant_2 | expansion_01b30a5e21e5_variant_1 | ok | 0.9625109999472453 | {'model': 'lgbm_850_leaves45_depth9_lr005'} |
| expansion_01b30a5e21e5_variant_1 | expansion_082c745e99b5_variant_2 | ok | 0.9622684999554328 | {'model': 'lgbm_750_mc100_l2_5_l1_05'} |
| expansion_b2f70299a092_variant_3 | expansion_01b30a5e21e5_variant_1 | ok | 0.9622684999554328 | {'subsample_fraction': 'train_frac_1.00'} |
| expansion_217f1b633e8f_variant_1 | expansion_01b30a5e21e5_variant_1 | ok | 0.9622504999479329 | {'model': 'lgbm_750_leaves63_mc100'} |
| expansion_01b30a5e21e5_variant_2 | expansion_082c745e99b5_variant_2 | ok | 0.9622309999496202 | {'model': 'lgbm_750_mc150_l2_10_l1_1'} |
| expansion_082c745e99b5_variant_2 | expansion_bc51974c040a_variant_1 | ok | 0.9622177499509954 | {'model': 'lgbm_500_mc100_l2_5'} |
| expansion_217f1b633e8f_variant_2 | expansion_01b30a5e21e5_variant_1 | ok | 0.9621502499545578 | {'model': 'lgbm_650_leaves127_mc150'} |
| expansion_082c745e99b5_variant_1 | expansion_bc51974c040a_variant_1 | ok | 0.9620614999456828 | {'model': 'lgbm_500_mc50_l2_1'} |
| expansion_b2f70299a092_variant_2 | expansion_01b30a5e21e5_variant_1 | ok | 0.9619567499091829 | {'subsample_fraction': 'train_frac_0.50'} |
| expansion_b2f70299a092_variant_1 | expansion_01b30a5e21e5_variant_1 | ok | 0.9612467499435576 | {'subsample_fraction': 'train_frac_0.25'} |
| expansion_bc51974c040a_variant_1 | expansion_c6c5e8a4847a_variant_2 | ok | 0.949563998874619 | {'model': 'lgbm_500_reg'} |
| expansion_c6c5e8a4847a_variant_2 | root | ok | 0.9465352481384312 | {'model': 'lgbm_300'} |
| expansion_bc51974c040a_variant_2 | expansion_c6c5e8a4847a_variant_2 | ok | 0.9459562476081814 | {'model': 'lgbm_700_reg'} |
| expansion_5d0a96ad9d22_variant_1 | expansion_c6c5e8a4847a_variant_2 | ok | 0.9434682465588687 | {'model': 'lgbm_500_leaves63'} |
| expansion_5d0a96ad9d22_variant_2 | expansion_c6c5e8a4847a_variant_2 | ok | 0.9431554967923063 | {'model': 'lgbm_500_leaves127'} |
| expansion_c6c5e8a4847a_variant_1 | root | ok | 0.9395449991152427 | {'model': 'lgbm_150'} |

## Evaluation audit

{
  "rows": 4000000,
  "folds": 3,
  "labels_missing": 0,
  "row_keys": "not declared",
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

- probe_2d2ab749b31f (expansion_01b30a5e21e5_variant_1): What is the out-of-fold error distribution across target cover types for top candidate expansion_01b30a5e21e5_variant_1? Specifically, how much error is mutual confusion between dominant classes 1 and 2 versus classification errors in minority classes (3, 4, 6, 7)? score 0.96227

## Findings

- finding_cb9935eb9db2: The target Cover_Type is heavily imbalanced, dominated by classes 1 and 2 (~93.3%), with extreme rarity in class 4 (377 rows) and class 5 (1 row). (target_distribution)
- finding_10a806f7999e: There are zero missing values across train and test datasets, and Soil_Type7 and Soil_Type15 are zero-variance constant columns in both splits. (data_cleanliness)
- finding_428816c937e1: Continuous features show synthetic generation artifacts with out-of-bound values (negative distances, out-of-range angles/hillshades), and train and test exhibit distribution drift in Wilderness_Area designations. (feature_distributions)
- finding_ad02b848fa45: Validation must handle single-instance class 5 (e.g. dropping row or custom stratification), and models should rarely or never predict class 5. (validation_and_modeling)
- finding_aed1e1eb6782: The dataset is non-stationary along the Id index: both feature distributions and target class frequencies drift systematically across the 1M-row blocks, with the final train quartile closely transitioning into test. (feature_and_target_drift)
- finding_5ebc6314afdd: All 4,000,000 rows in the training set possess unique feature vectors, confirming there are zero duplicate rows and zero conflicting labels. (data_integrity)
- finding_c2f3d8e6e3fb: Standard scikit-learn StratifiedKFold runs successfully on the training target Cover_Type without throwing exceptions, despite the single-instance class 5. (validation_feasibility)
- finding_ef6611299f44: LightGBM exhibits rapid training throughput (>500k rows/second) and low memory overhead (~420 MB/million rows), making multi-fold training on the 4M dataset computationally fast and lightweight. (compute_feasibility)

All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
