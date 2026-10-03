# Experiment report

Policy: greedy; model: gemini/gemini-3.8-flash (reasoning high); state: complete; elapsed: 221 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_674c0e08f0ab_variant_2 | expansion_6bd179b8a8bd_variant_2 | ok | 0.9624754999490578 | {'colsample_bytree': 0.85} |
| expansion_6bd179b8a8bd_variant_2 | expansion_6f62aa67c818_variant_1 | ok | 0.9624639999321203 | {'num_leaves': 90} |
| expansion_6bd179b8a8bd_variant_1 | expansion_6f62aa67c818_variant_1 | ok | 0.9624484999233704 | {'num_leaves': 63} |
| expansion_674c0e08f0ab_variant_1 | expansion_6bd179b8a8bd_variant_2 | ok | 0.9624482499358703 | {'colsample_bytree': 0.7} |
| expansion_674c0e08f0ab_variant_3 | expansion_6bd179b8a8bd_variant_2 | ok | 0.9624464999504329 | {'colsample_bytree': 1.0} |
| expansion_6bd179b8a8bd_variant_3 | expansion_6f62aa67c818_variant_1 | ok | 0.9624217499314954 | {'num_leaves': 120} |
| expansion_6f62aa67c818_variant_1 | expansion_4c23919c0840_variant_3 | ok | 0.9620174999569953 | {'reg_lambda': 15.0} |
| expansion_6f62aa67c818_variant_2 | expansion_4c23919c0840_variant_3 | ok | 0.9620134999484953 | {'reg_lambda': 30.0} |
| expansion_6f62aa67c818_variant_3 | expansion_4c23919c0840_variant_3 | ok | 0.9619609999355578 | {'reg_lambda': 50.0} |
| expansion_4c23919c0840_variant_3 | expansion_46ec68da413d_variant_1 | ok | 0.9618697499466827 | {'reg_lambda': 10.0} |
| expansion_4c23919c0840_variant_2 | expansion_46ec68da413d_variant_1 | ok | 0.9618654999572452 | {'reg_lambda': 5.0} |
| expansion_4c23919c0840_variant_1 | expansion_46ec68da413d_variant_1 | ok | 0.9618552499509327 | {'reg_lambda': 1.0} |
| expansion_46ec68da413d_variant_1 | expansion_93b422d43300_variant_1 | ok | 0.9610299999428076 | {'feature_set': 'hydrology_spatial'} |
| expansion_93b422d43300_variant_1 | expansion_cd8aa00c1878_variant_3 | ok | 0.96062174991987 | {'num_leaves': 63} |
| expansion_46ec68da413d_variant_3 | expansion_93b422d43300_variant_1 | ok | 0.9605044997375577 | {'feature_set': 'full_terrain_indicators'} |
| expansion_93b422d43300_variant_3 | expansion_cd8aa00c1878_variant_3 | ok | 0.960460749904745 | {'num_leaves': 127} |
| expansion_46ec68da413d_variant_2 | expansion_93b422d43300_variant_1 | ok | 0.9602734997407452 | {'feature_set': 'hydro_solar_aspect'} |
| expansion_93b422d43300_variant_2 | expansion_cd8aa00c1878_variant_3 | ok | 0.9600489998456826 | {'num_leaves': 95} |
| expansion_cd8aa00c1878_variant_3 | expansion_bdc8175b1f27_variant_1 | ok | 0.9575839992238073 | {'reg_schedule': 'SubsampledClassifier(...)'} |
| expansion_cd8aa00c1878_variant_2 | expansion_bdc8175b1f27_variant_1 | ok | 0.9575822493004323 | {'reg_schedule': 'SubsampledClassifier(...)'} |
| expansion_cd8aa00c1878_variant_1 | expansion_bdc8175b1f27_variant_1 | ok | 0.953011248798182 | {'reg_schedule': 'SubsampledClassifier(...)'} |
| expansion_bdc8175b1f27_variant_1 | expansion_811f3f25422f_variant_2 | ok | 0.9476392482708063 | {'num_leaves': 63} |
| expansion_811f3f25422f_variant_2 | root | ok | 0.9458507476384314 | {'train_fraction': 1.0} |
| expansion_bdc8175b1f27_variant_2 | expansion_811f3f25422f_variant_2 | ok | 0.9433217472563061 | {'num_leaves': 127} |
| expansion_bdc8175b1f27_variant_3 | expansion_811f3f25422f_variant_2 | ok | 0.941968247222431 | {'num_leaves': 255} |
| expansion_6aa8d579e8bf_variant_1 | expansion_811f3f25422f_variant_2 | ok | 0.9413777476952432 | {'feature_set': 'baseline'} |
| expansion_6aa8d579e8bf_variant_2 | expansion_811f3f25422f_variant_2 | ok | 0.9396659959908059 | {'feature_set': 'physical'} |
| expansion_6aa8d579e8bf_variant_3 | expansion_811f3f25422f_variant_2 | ok | 0.9391987454389935 | {'feature_set': 'all_domain'} |
| expansion_811f3f25422f_variant_1 | root | ok | 0.9346222463800552 | {'train_fraction': 0.25} |

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

- probe_1f9e765b8628 (expansion_811f3f25422f_variant_2): What are the primary error patterns and per-class confusions across the 7 Cover_Type classes in the out-of-fold predictions of the top baseline model? score 0.94475

## Findings

- finding_149476bb6088: Cover_Type is severely imbalanced across the 4M train rows: classes 1 and 2 account for ~93.25% of all samples, while class 5 contains only a single sample and class 4 contains 377 samples. (target)
- finding_e908dc884901: There are zero missing values across both the 4,000,000-row training set and the 1,000,000-row test set. (data_hygiene)
- finding_2836d9fa92dd: Soil_Type7 and Soil_Type15 are strictly constant with value 0 in both train and test datasets and can be dropped. (feature_selection)
- finding_e6eb38172f5c: The full dataset loads as 64-bit integers taking ~1.79 GB for train and ~440 MB for test; downcasting binary indicators and small-range features will significantly decrease memory footprint. (memory)
- finding_afc715a6db99: Standard StratifiedKFold cross-validation will fail or require special handling because Cover_Type 5 has only a single observation. (validation)
- finding_58edb716a6fb: Multiple numeric features exhibit out-of-bounds values, including negative distances, negative slopes, Aspect outside [0, 360], and Hillshades outside [0, 255]. (feature_engineering)
- finding_9574bdeabb40: StratifiedKFold (3-fold) partitions the 4M dataset without raising errors despite singleton Cover_Type 5, assigning the singleton to fold 0 test (and folds 1 and 2 train). (validation)
- finding_ff3d0b4bab2d: LightGBM fits and evaluates without runtime errors when class 5 is omitted from train or test, but predict_proba column counts vary (6 vs 7) depending on whether class 5 appears in training. (modeling)
- finding_40d70c05c26d: LightGBM demonstrates fast training throughput (~100k samples/sec) on this tabular dataset, scaling to 93.8% validation accuracy on 200k samples in ~2.1 seconds. (modeling)

All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
