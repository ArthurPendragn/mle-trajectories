# Experiment report

Controller: auto; policy: islands; memory: window; model: gemini/gemini-3.8-flash; state: complete; elapsed: 291 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_1c778324d0d8_variant_1 | expansion_3f23a0e5cfb6_variant_2 | ok | -3.9578830027302656 | {'ensemble_weights': 'tri_1800_equal'} |
| expansion_1c778324d0d8_variant_2 | expansion_3f23a0e5cfb6_variant_2 | ok | -3.958000960103791 | {'ensemble_weights': 'tri_1800_skewed_xgb10'} |
| expansion_3f23a0e5cfb6_variant_2 | expansion_f812d00f9845_variant_1 | ok | -3.9585878862669652 | {'feature_config': 'tlc_tariffs_and_topography'} |
| expansion_79fc9908d81c_variant_2 | expansion_365fb4a3cee0_variant_2 | ok | -3.9589194075229543 | {'boosting_schedule': 'schedule_1800_lr022'} |
| expansion_3f23a0e5cfb6_variant_1 | expansion_f812d00f9845_variant_1 | ok | -3.959198910156828 | {'feature_config': 'tlc_enhanced_tariffs'} |
| expansion_f4233b5dc87b_variant_1 | expansion_56eac13d2941_variant_2 | ok | -3.9595451959828445 | {'topography_features': 'topography_and_tariffs'} |
| expansion_56eac13d2941_variant_2 | expansion_f812d00f9845_variant_1 | ok | -3.9607546541009255 | {'regulatory_features': 'tlc_hike_jfk_and_elapsed_time'} |
| expansion_f812d00f9845_variant_1 | expansion_db840584f901_variant_1 | ok | -3.961060777963076 | {'boosting_schedule': 'schedule_1600_lr025'} |
| expansion_f4233b5dc87b_variant_2 | expansion_56eac13d2941_variant_2 | ok | -3.9612721231132584 | {'topography_features': 'topography_tariffs_and_hudson'} |
| expansion_703d18ba0d8d_variant_1 | expansion_79fc9908d81c_variant_2 | ok | -3.9616740778992963 | {'traffic_features': 'traffic_distance_interactions'} |
| expansion_db840584f901_variant_1 | expansion_54a623e9fac5_variant_2 | ok | -3.962732274222811 | {'ensemble_weights': 'tri_ensemble_equal'} |
| expansion_f812d00f9845_variant_2 | expansion_db840584f901_variant_1 | ok | -3.962732372429464 | {'boosting_schedule': 'schedule_2000_lr020'} |
| expansion_db840584f901_variant_2 | expansion_54a623e9fac5_variant_2 | ok | -3.96288986352466 | {'ensemble_weights': 'tri_ensemble_skewed'} |
| expansion_365fb4a3cee0_variant_2 | expansion_9aa4885bde95_variant_2 | ok | -3.963364328478908 | {'boosting_schedule': 'schedule_1800_lr022'} |
| expansion_9aa4885bde95_variant_2 | expansion_022f906df412_variant_1 | ok | -3.9637022810221465 | {'ensemble_weights': 'skewed_xgb_40_60'} |
| expansion_703d18ba0d8d_variant_2 | expansion_79fc9908d81c_variant_2 | ok | -3.963734911020544 | {'traffic_features': 'traffic_and_intra_manhattan'} |
| expansion_54a623e9fac5_variant_2 | expansion_c385be20edee_variant_2 | ok | -3.963977253880864 | {'feature_variant': 'rotated_coords_and_tlc'} |
| expansion_56eac13d2941_variant_1 | expansion_f812d00f9845_variant_1 | ok | -3.964207988588595 | {'regulatory_features': 'tlc_post_hike_and_jfk_flat'} |
| expansion_9aa4885bde95_variant_1 | expansion_022f906df412_variant_1 | ok | -3.964479060474698 | {'ensemble_weights': 'balanced_50_50'} |
| expansion_79fc9908d81c_variant_1 | expansion_365fb4a3cee0_variant_2 | ok | -3.9644987881131315 | {'boosting_schedule': 'schedule_1600_lr025'} |
| expansion_365fb4a3cee0_variant_1 | expansion_9aa4885bde95_variant_2 | ok | -3.9660718348676967 | {'boosting_schedule': 'schedule_1600_lr025'} |
| expansion_ee183318067e_variant_1 | expansion_54a623e9fac5_variant_2 | ok | -3.9668884372221562 | {'feature_variant': 'rot_spatial_enhancements'} |
| expansion_41a0afbd955e_variant_1 | expansion_ee183318067e_variant_1 | ok | -3.970709857443782 | {'feature_variant': 'clean_spatial_and_fare_hike'} |
| expansion_41a0afbd955e_variant_2 | expansion_ee183318067e_variant_1 | ok | -3.970716323106931 | {'feature_variant': 'spatial_fare_hike_and_manhattan_bounds'} |
| expansion_ee183318067e_variant_2 | expansion_54a623e9fac5_variant_2 | ok | -3.9713540138080545 | {'feature_variant': 'rot_spatial_and_fare_hike'} |
| expansion_d9733f1a38b3_variant_2 | expansion_14aa4c6fabf7_variant_2 | ok | -3.9848299793448545 | {'feature_variant': 'rotated_grid_and_tlc_rules'} |
| expansion_54a623e9fac5_variant_1 | expansion_c385be20edee_variant_2 | ok | -3.9869281328768738 | {'feature_variant': 'rotated_delta_and_tlc'} |
| expansion_022f906df412_variant_1 | expansion_e5729711f04d_variant_1 | ok | -3.987217034785504 | {'feature_variant': 'rotated_and_tlc'} |
| expansion_d9733f1a38b3_variant_1 | expansion_14aa4c6fabf7_variant_2 | ok | -3.9873126495369227 | {'feature_variant': 'rotated_grid'} |
| expansion_022f906df412_variant_2 | expansion_e5729711f04d_variant_1 | ok | -3.9915769435563875 | {'feature_variant': 'rotated_tlc_and_fare_hike'} |
| expansion_14aa4c6fabf7_variant_2 | expansion_920b7b727d49_variant_1 | ok | -3.9915889016708945 | {'xgb_depth': 'xgb_depth_10'} |
| expansion_c385be20edee_variant_1 | expansion_dd10dc9800b8_variant_2 | ok | -3.993184085364716 | {'ensemble_type': 'tri_ensemble_equal'} |
| expansion_e5729711f04d_variant_1 | expansion_dd10dc9800b8_variant_2 | ok | -3.9938933463287176 | {'xgb_depth': 'xgb_depth_10'} |
| expansion_c385be20edee_variant_2 | expansion_dd10dc9800b8_variant_2 | ok | -3.9939129109386755 | {'ensemble_type': 'skewed_blend_lgb_xgb10'} |
| expansion_e5729711f04d_variant_2 | expansion_dd10dc9800b8_variant_2 | ok | -3.995247212765565 | {'xgb_depth': 'xgb_depth_11'} |
| expansion_14aa4c6fabf7_variant_1 | expansion_920b7b727d49_variant_1 | ok | -3.9967047845187347 | {'xgb_depth': 'xgb_depth_9'} |
| expansion_dd10dc9800b8_variant_2 | expansion_79243fe54cb9_variant_2 | ok | -3.9986005672698575 | {'xgb_depth': 'xgb_depth_9'} |
| expansion_920b7b727d49_variant_1 | expansion_a85dda868c7f_variant_2 | ok | -4.000225483282224 | {'blend_weights': 'equal_blend'} |
| expansion_920b7b727d49_variant_2 | expansion_a85dda868c7f_variant_2 | ok | -4.000486788720071 | {'blend_weights': 'xgb_skewed_blend'} |
| expansion_dd10dc9800b8_variant_1 | expansion_79243fe54cb9_variant_2 | ok | -4.000923974527268 | {'xgb_depth': 'xgb_depth_8'} |
| expansion_01438a0516e9_variant_2 | expansion_f6738b3523ba_variant_1 | ok | -4.012016388184904 | {'ensemble_capacity': 'leaves95_depth8'} |
| expansion_6a42d25b0b49_variant_2 | expansion_322c8b9729d7_variant_2 | ok | -4.012803521994091 | {'directional_variant': 'signed_plus_bearing'} |
| expansion_01438a0516e9_variant_1 | expansion_f6738b3523ba_variant_1 | ok | -4.015796489694099 | {'ensemble_capacity': 'leaves63_depth8'} |
| expansion_a85dda868c7f_variant_2 | expansion_d2ad1aa42536_variant_1 | ok | -4.021401756110403 | {'directional_variant': 'signed_plus_bearing'} |
| expansion_79243fe54cb9_variant_2 | expansion_d2ad1aa42536_variant_1 | ok | -4.029360061367568 | {'num_leaves': 'leaves95_trees1200_lr003'} |
| expansion_79243fe54cb9_variant_1 | expansion_d2ad1aa42536_variant_1 | ok | -4.029512807161046 | {'num_leaves': 'leaves63_trees1200_lr003'} |
| expansion_d2ad1aa42536_variant_1 | expansion_b6e3f19ce43e_variant_1 | ok | -4.0306955075544035 | {'num_leaves': 'leaves63_trees1200_lr003'} |
| expansion_d2ad1aa42536_variant_2 | expansion_b6e3f19ce43e_variant_1 | ok | -4.031004719210404 | {'num_leaves': 'leaves95_trees1200_lr003'} |
| expansion_a85dda868c7f_variant_1 | expansion_d2ad1aa42536_variant_1 | ok | -4.034058240810998 | {'directional_variant': 'signed_displacement'} |
| expansion_322c8b9729d7_variant_2 | expansion_b6e3f19ce43e_variant_1 | ok | -4.037145749912972 | {'max_depth': 'depth8_trees1200_lr003'} |
| expansion_f6738b3523ba_variant_1 | expansion_63c700ab6ca0_variant_1 | ok | -4.037444646732373 | {'model_capacity': 'leaves95_trees1200_lr003'} |
| expansion_6a42d25b0b49_variant_1 | expansion_322c8b9729d7_variant_2 | ok | -4.037575590370386 | {'directional_variant': 'signed_displacement'} |
| expansion_3729ea47ff1f_variant_1 | expansion_b6e3f19ce43e_variant_1 | ok | -4.0393088942107465 | {'model_capacity': 'leaves127_trees1200_lr003'} |
| expansion_7cf2e09e9ac8_variant_2 | expansion_63c700ab6ca0_variant_1 | ok | -4.041710008282012 | {'feature_variant': 'directional_pruned'} |
| expansion_f6738b3523ba_variant_2 | expansion_63c700ab6ca0_variant_1 | ok | -4.047889610810394 | {'model_capacity': 'leaves127_trees1200_lr003'} |
| expansion_3729ea47ff1f_variant_2 | expansion_b6e3f19ce43e_variant_1 | ok | -4.050668524367546 | {'model_capacity': 'leaves255_trees1200_lr003'} |
| expansion_b6e3f19ce43e_variant_1 | root | ok | -4.051485311942981 | {} |
| expansion_322c8b9729d7_variant_1 | expansion_b6e3f19ce43e_variant_1 | ok | -4.054465698155438 | {'max_depth': 'depth7_trees1200_lr003'} |
| expansion_7cf2e09e9ac8_variant_1 | expansion_63c700ab6ca0_variant_1 | ok | -4.059207517439797 | {'feature_variant': 'directional_all'} |
| expansion_94656b35ff61_variant_1 | expansion_b6e3f19ce43e_variant_1 | ok | -4.061848721150104 | {} |
| expansion_61003b15d3c7_variant_1 | expansion_b6e3f19ce43e_variant_1 | ok | -4.063812068320002 | {} |
| expansion_63c700ab6ca0_variant_1 | expansion_e0b35bed6418_variant_1 | ok | -4.063812068320002 | {} |
| expansion_14b06406b260_variant_1 | expansion_9cc0fb1bf38c_variant_1 | ok | -4.114373999098975 | {} |
| expansion_c9b19d52a385_variant_1 | expansion_9cc0fb1bf38c_variant_1 | ok | -4.16683702629706 | {} |
| expansion_9cc0fb1bf38c_variant_1 | root | ok | -4.178899354087719 | {} |
| expansion_7654da8eafb9_variant_1 | expansion_9cc0fb1bf38c_variant_1 | ok | -4.180254663419172 | {} |
| expansion_e0b35bed6418_variant_1 | expansion_d47b360fd3e7_variant_1 | ok | -4.711958019478803 | {} |
| expansion_d47b360fd3e7_variant_1 | root | ok | -5.289580683073235 | {} |

## Evaluation audit

{
  "rows": 487784,
  "folds": 5,
  "labels_missing": 0,
  "row_keys": "not declared",
  "fold_sizes": [
    {
      "train": 390227,
      "test": 97557
    },
    {
      "train": 390227,
      "test": 97557
    },
    {
      "train": 390227,
      "test": 97557
    },
    {
      "train": 390227,
      "test": 97557
    },
    {
      "train": 390228,
      "test": 97556
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Findings


All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
