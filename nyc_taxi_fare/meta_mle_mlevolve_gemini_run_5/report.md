# Experiment report

Controller: auto; policy: mlevolve; memory: experience; model: gemini/gemini-3.8-flash; state: complete; elapsed: 464 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_5e4b486ef55a_variant_2 | expansion_ff8b90a73410_variant_2 | ok | -3.8947244212257957 | {'model_variant': 'wide_resnet_weight_30'} |
| expansion_504db42f8907_variant_2 | expansion_764a4514a5de_variant_1 | ok | -3.894796521784457 | {'model_variant': 'bridge_detour_weight_35'} |
| expansion_504db42f8907_variant_1 | expansion_764a4514a5de_variant_1 | ok | -3.8958340016879 | {'model_variant': 'bridge_detour_weight_30'} |
| expansion_5e4b486ef55a_variant_1 | expansion_ff8b90a73410_variant_2 | ok | -3.8968016796114773 | {'model_variant': 'wide_resnet_weight_25'} |
| expansion_764a4514a5de_variant_1 | expansion_5e4b486ef55a_variant_2 | ok | -3.8973514028554526 | {'feature_variant': 'lga_toll_and_morning_congestion'} |
| expansion_ff8b90a73410_variant_2 | expansion_fdac56b10832_variant_2 | ok | -3.8977982342607347 | {'model_variant': 'silu_weight_25'} |
| expansion_aca5e4ded6cd_variant_2 | expansion_504db42f8907_variant_2 | ok | -3.8996753077588457 | {'feature_variant': 'grid_bridge_detour_traffic_and_2015_surcharge'} |
| expansion_ff8b90a73410_variant_1 | expansion_fdac56b10832_variant_2 | ok | -3.901162680237595 | {'model_variant': 'silu_weight_20'} |
| expansion_fdac56b10832_variant_2 | expansion_ffadee794e66_variant_2 | ok | -3.9028643751569256 | {'model_variant': 'resnet_25ep_weight_20'} |
| expansion_aca5e4ded6cd_variant_1 | expansion_504db42f8907_variant_2 | ok | -3.904138864889334 | {'feature_variant': 'grid_bridge_detour_and_traffic'} |
| expansion_ffadee794e66_variant_2 | expansion_64b8334e7884_variant_1 | ok | -3.905880749585843 | {'model_variant': 'cosine_lr_leaves_256_weight_15'} |
| expansion_ce76b96ff816_variant_1 | expansion_aca5e4ded6cd_variant_2 | ok | -3.906714795551212 | {} |
| expansion_fdac56b10832_variant_1 | expansion_ffadee794e66_variant_2 | ok | -3.9075884565554246 | {'model_variant': 'resnet_25ep_weight_15'} |
| expansion_764a4514a5de_variant_2 | expansion_5e4b486ef55a_variant_2 | ok | -3.9093363756495934 | {'feature_variant': 'full_borough_and_toll_features'} |
| expansion_0471a8ecf805_variant_1 | expansion_88b161142a8e_variant_2 | ok | -3.9121599154402498 | {'model_variant': 'scaled_capacity_leaves_192'} |
| expansion_ffadee794e66_variant_1 | expansion_64b8334e7884_variant_1 | ok | -3.9127804662734675 | {'model_variant': 'cosine_lr_leaves_256_weight_10'} |
| expansion_64b8334e7884_variant_1 | expansion_0471a8ecf805_variant_1 | ok | -3.913051664907549 | {'feature_variant': 'toll_and_rate_interaction_features'} |
| expansion_0471a8ecf805_variant_2 | expansion_88b161142a8e_variant_2 | ok | -3.914376684195406 | {'model_variant': 'deep_capacity_leaves_224'} |
| expansion_88b161142a8e_variant_2 | expansion_49e426165a3c_variant_1 | ok | -3.917734795778961 | {'feature_variant': 'composite_statutory_crossings_and_boroughs'} |
| expansion_88b161142a8e_variant_1 | expansion_49e426165a3c_variant_1 | ok | -3.9184751945997163 | {'feature_variant': 'composite_statutory_and_crossings'} |
| expansion_64b8334e7884_variant_2 | expansion_0471a8ecf805_variant_1 | ok | -3.918483449807927 | {'feature_variant': 'comprehensive_toll_rate_and_chokepoint_features'} |
| expansion_49e426165a3c_variant_1 | expansion_b662534310aa_variant_1 | ok | -3.920900873735492 | {'model_variant': 'high_capacity_balanced_ensemble'} |
| expansion_b662534310aa_variant_1 | expansion_b6b1b101b0c8_variant_1 | ok | -3.9235741344389026 | {'feature_variant': 'directional_grid_features'} |
| expansion_aeb2b5283f67_variant_1 | expansion_bfea2f4575fc_variant_2 | ok | -3.925608885893999 | {'feature_variant': 'rotated_grid_features'} |
| expansion_49e426165a3c_variant_2 | expansion_b662534310aa_variant_1 | ok | -3.92573259966355 | {'model_variant': 'high_capacity_tree_dominant_ensemble'} |
| expansion_aeb2b5283f67_variant_2 | expansion_bfea2f4575fc_variant_2 | ok | -3.926318443956864 | {'feature_variant': 'rotated_grid_and_transverse_features'} |
| expansion_bfea2f4575fc_variant_2 | expansion_d4802a80275c_variant_1 | ok | -3.9266207923238667 | {'feature_variant': 'full_temporal_spatial'} |
| expansion_b662534310aa_variant_2 | expansion_b6b1b101b0c8_variant_1 | ok | -3.926743742080376 | {'feature_variant': 'directional_grid_and_peak_asymmetry'} |
| expansion_bfea2f4575fc_variant_1 | expansion_d4802a80275c_variant_1 | ok | -3.92705369931747 | {'feature_variant': 'congestion_and_bearing'} |
| expansion_d4802a80275c_variant_1 | expansion_9edce451aa09_variant_1 | ok | -3.9275498826743656 | {'model_variant': 'balanced_deep_ensemble'} |
| expansion_b6b1b101b0c8_variant_1 | expansion_aeb2b5283f67_variant_1 | ok | -3.9279139587118217 | {'feature_variant': 'statutory_toll_and_morning_congestion'} |
| expansion_d4802a80275c_variant_2 | expansion_9edce451aa09_variant_1 | ok | -3.928164048818202 | {'model_variant': 'lgb_dominant_deep_ensemble'} |
| expansion_b6b1b101b0c8_variant_2 | expansion_aeb2b5283f67_variant_1 | ok | -3.9306326182796 | {'feature_variant': 'statutory_toll_congestion_and_aspect_ratios'} |
| expansion_9edce451aa09_variant_1 | expansion_ea7cb030cd9f_variant_2 | ok | -3.933246346438542 | {'feature_variant': 'statutory_tariff_features'} |
| expansion_9edce451aa09_variant_2 | expansion_ea7cb030cd9f_variant_2 | ok | -3.93659140661793 | {'feature_variant': 'statutory_and_borough_crossing_features'} |
| expansion_718178b9f93d_variant_1 | expansion_9a10ef205af3_variant_2 | ok | -3.9428279150726055 | {'feature_variant': 'manhattan_traffic_features'} |
| expansion_ea7cb030cd9f_variant_2 | expansion_6a854333068f_variant_1 | ok | -3.9460704879764292 | {'model_variant': 'tri_ensemble_lgb_xgb_resnet'} |
| expansion_718178b9f93d_variant_2 | expansion_9a10ef205af3_variant_2 | ok | -3.9509459032135568 | {'feature_variant': 'manhattan_and_detour_features'} |
| expansion_9a10ef205af3_variant_2 | expansion_b25e4c02b3d2_variant_1 | ok | -3.9526821765621754 | {'feature_variant': 'full_domain_features'} |
| expansion_6a854333068f_variant_1 | expansion_718178b9f93d_variant_1 | ok | -3.954958881718375 | {'feature_variant': 'tariff_diurnal_interactions'} |
| expansion_6a854333068f_variant_2 | expansion_718178b9f93d_variant_1 | ok | -3.955212643484333 | {'feature_variant': 'full_interactions'} |
| expansion_9a10ef205af3_variant_1 | expansion_b25e4c02b3d2_variant_1 | ok | -3.957955295298439 | {'feature_variant': 'core_domain_features'} |
| expansion_b25e4c02b3d2_variant_1 | expansion_53a6fedd6ee1_variant_2 | ok | -3.963643226816336 | {'model_variant': 'blend_75_25'} |
| expansion_ea7cb030cd9f_variant_1 | expansion_6a854333068f_variant_1 | ok | -3.965086599878961 | {'model_variant': 'lgb_xgb_ensemble'} |
| expansion_b25e4c02b3d2_variant_2 | expansion_53a6fedd6ee1_variant_2 | ok | -3.967623146197905 | {'model_variant': 'blend_85_15'} |
| expansion_7a35a2ba1d47_variant_2 | root | ok | -3.9882749251630196 | {'model_variant': 'lgbm_leaves63'} |
| expansion_7a35a2ba1d47_variant_1 | root | ok | -3.994832785161459 | {'model_variant': 'lgbm_leaves31'} |
| expansion_53a6fedd6ee1_variant_2 | expansion_d53f986de1b2_variant_1 | ok | -4.178711404358836 | {'model_variant': 'resnet_3blocks'} |
| expansion_d53f986de1b2_variant_1 | root | ok | -4.181358382483024 | {'model_variant': 'mlp_hidden256'} |
| expansion_d53f986de1b2_variant_2 | root | ok | -4.185493029552421 | {'model_variant': 'mlp_hidden512'} |
| expansion_53a6fedd6ee1_variant_1 | expansion_d53f986de1b2_variant_1 | ok | -4.196986650229377 | {'model_variant': 'resnet_2blocks'} |

## Evaluation audit

{
  "rows": 487692,
  "folds": 5,
  "labels_missing": 0,
  "row_keys": "not declared",
  "fold_sizes": [
    {
      "train": 390153,
      "test": 97539
    },
    {
      "train": 390153,
      "test": 97539
    },
    {
      "train": 390154,
      "test": 97538
    },
    {
      "train": 390154,
      "test": 97538
    },
    {
      "train": 390154,
      "test": 97538
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Findings


All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
