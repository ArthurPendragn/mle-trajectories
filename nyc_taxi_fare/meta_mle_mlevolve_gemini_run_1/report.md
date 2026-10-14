# Experiment report

Controller: auto; policy: mlevolve; memory: experience; model: gemini/gemini-3.8-flash; state: complete; elapsed: 289 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_2d4d1e5df9d4_variant_2 | expansion_67fc139d1bd7_variant_1 | ok | -3.9100839063699397 | {'ensemble_weights': 'trees_slight_edge'} |
| expansion_2d4d1e5df9d4_variant_1 | expansion_67fc139d1bd7_variant_1 | ok | -3.9101409534713136 | {'ensemble_weights': 'tri_equal'} |
| expansion_b702fd9ab6d1_variant_2 | expansion_841bcf2f620f_variant_1 | ok | -3.9102040587153426 | {'ensemble_weights': 'resnet_equal_33'} |
| expansion_b702fd9ab6d1_variant_1 | expansion_841bcf2f620f_variant_1 | ok | -3.9105062538862327 | {'ensemble_weights': 'resnet_boosted_30'} |
| expansion_841bcf2f620f_variant_1 | expansion_923a36b4c7c4_variant_1 | ok | -3.9111702693219357 | {'ensemble_weights': 'resnet_boosted'} |
| expansion_516a7f1a3ca4_variant_2 | expansion_2d4d1e5df9d4_variant_2 | ok | -3.9129057900247752 | {'tree_feature_partition': 'all_features_for_trees'} |
| expansion_516a7f1a3ca4_variant_1 | expansion_2d4d1e5df9d4_variant_2 | ok | -3.912914291816576 | {'tree_feature_partition': 'filter_fourier_for_trees'} |
| expansion_841bcf2f620f_variant_2 | expansion_923a36b4c7c4_variant_1 | ok | -3.9140139656711 | {'ensemble_weights': 'balanced_refined'} |
| expansion_67fc139d1bd7_variant_1 | expansion_b702fd9ab6d1_variant_2 | ok | -3.915539297690917 | {'ensemble_weights': 'tri_balanced'} |
| expansion_923a36b4c7c4_variant_1 | expansion_b9b8ea415afb_variant_1 | ok | -3.917495516642395 | {'ensemble_weights': 'balanced_deep'} |
| expansion_da8bd268c4ff_variant_1 | expansion_516a7f1a3ca4_variant_2 | ok | -3.9183285286909126 | {'ensemble_weights': 'balanced_trees_resnet'} |
| expansion_da8bd268c4ff_variant_2 | expansion_516a7f1a3ca4_variant_2 | ok | -3.918738620930572 | {'ensemble_weights': 'equal_tri_ensemble'} |
| expansion_67fc139d1bd7_variant_2 | expansion_b702fd9ab6d1_variant_2 | ok | -3.9189804618239537 | {'ensemble_weights': 'resnet_heavy_40'} |
| expansion_65a94b508b2f_variant_2 | expansion_40b5b2d7b8e6_variant_2 | ok | -3.9267403679145985 | {'routing_variant': 'domain_routing_log'} |
| expansion_923a36b4c7c4_variant_2 | expansion_b9b8ea415afb_variant_1 | ok | -3.927784164843333 | {'ensemble_weights': 'tree_dominant_deep'} |
| expansion_b9b8ea415afb_variant_1 | expansion_65a94b508b2f_variant_2 | ok | -3.9280665743320697 | {'pricing_variant': 'traffic_pricing'} |
| expansion_65a94b508b2f_variant_1 | expansion_40b5b2d7b8e6_variant_2 | ok | -3.928386214765048 | {'routing_variant': 'domain_routing'} |
| expansion_b9b8ea415afb_variant_2 | expansion_65a94b508b2f_variant_2 | ok | -3.9314129246864753 | {'pricing_variant': 'traffic_pricing_boroughs'} |
| expansion_40b5b2d7b8e6_variant_2 | expansion_bd53f4a76da7_variant_1 | ok | -3.9417727438872348 | {'weights': 'equal_trees'} |
| expansion_40b5b2d7b8e6_variant_1 | expansion_bd53f4a76da7_variant_1 | ok | -3.942046726733433 | {'weights': 'balanced_trees'} |
| expansion_08dfbeb6536e_variant_2 | expansion_fb0012bfb8e7_variant_2 | ok | -3.951925713363937 | {'ensemble_weight': 'blend_70_30'} |
| expansion_bd53f4a76da7_variant_1 | expansion_08dfbeb6536e_variant_2 | ok | -3.956395864604006 | {} |
| expansion_08dfbeb6536e_variant_1 | expansion_fb0012bfb8e7_variant_2 | ok | -3.967582625791386 | {'ensemble_weight': 'blend_50_50'} |
| expansion_1de0906b1362_variant_2 | root | ok | -4.022962890625574 | {'lgbm_capacity': 'lgbm_leaves63'} |
| expansion_1de0906b1362_variant_1 | root | ok | -4.035742305403416 | {'lgbm_capacity': 'lgbm_leaves31'} |
| expansion_fb0012bfb8e7_variant_2 | expansion_204ae0c54982_variant_2 | ok | -4.122110351142571 | {'linear_bypass': 'regularized'} |
| expansion_204ae0c54982_variant_2 | expansion_461e43538681_variant_1 | ok | -4.129558000863215 | {'spatial_fourier_octaves': 'octaves_6'} |
| expansion_fb0012bfb8e7_variant_1 | expansion_204ae0c54982_variant_2 | ok | -4.132548694771392 | {'linear_bypass': 'direct'} |
| expansion_204ae0c54982_variant_1 | expansion_461e43538681_variant_1 | ok | -4.139304232184673 | {'spatial_fourier_octaves': 'octaves_4'} |
| expansion_461e43538681_variant_1 | expansion_6bf432a25112_variant_2 | ok | -4.1961592938040235 | {} |
| expansion_6bf432a25112_variant_2 | expansion_b3e0fa4ae80e_variant_2 | ok | -4.1999947362967 | {'resnet_depth': 'resnet_3blocks'} |
| expansion_b3e0fa4ae80e_variant_2 | root | ok | -4.203909603756392 | {'mlp_capacity': 'mlp_4layer_256'} |
| expansion_6bf432a25112_variant_1 | expansion_b3e0fa4ae80e_variant_2 | ok | -4.274205507665227 | {'resnet_depth': 'resnet_2blocks'} |
| expansion_b3e0fa4ae80e_variant_1 | root | ok | -4.308854600602061 | {'mlp_capacity': 'mlp_3layer_128'} |

## Evaluation audit

{
  "rows": 487778,
  "folds": 5,
  "labels_missing": 0,
  "row_keys": "not declared",
  "fold_sizes": [
    {
      "train": 390222,
      "test": 97556
    },
    {
      "train": 390222,
      "test": 97556
    },
    {
      "train": 390222,
      "test": 97556
    },
    {
      "train": 390223,
      "test": 97555
    },
    {
      "train": 390223,
      "test": 97555
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Findings


All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
