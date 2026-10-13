# Experiment report

Controller: auto; policy: mlevolve; memory: experience; model: gemini/gemini-3.8-flash; state: complete; elapsed: 117 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_3e685ab931dd_variant_2 | expansion_51251f25b8c5_variant_1 | ok | 0.8155986900398429 | {'ensemble_weights': 'tree_emphasized'} |
| expansion_fb23186268c0_variant_2 | expansion_3e685ab931dd_variant_2 | ok | 0.8152541273737072 | {'feature_expansion': 'spend_shares_only'} |
| expansion_3e685ab931dd_variant_3 | expansion_51251f25b8c5_variant_1 | ok | 0.8149086382859612 | {'ensemble_weights': 'conservative_gam'} |
| expansion_246ebbac5eda_variant_1 | expansion_383250660a55_variant_1 | ok | 0.8146791504184447 | {} |
| expansion_2f8f41dbaafe_variant_1 | expansion_7e8b3ae46375_variant_1 | ok | 0.8146791504184447 | {'regularization': 'default'} |
| expansion_8bc081c33da1_variant_1 | expansion_3e685ab931dd_variant_2 | ok | 0.8146778269590012 | {'meta_regularization': 'C0_1'} |
| expansion_383250660a55_variant_1 | expansion_566808a3af1c_variant_1 | ok | 0.814564141792798 | {} |
| expansion_2ac03900a142_variant_2 | expansion_11d9504e31b9_variant_2 | ok | 0.814333793676643 | {'bagging_diversity': 'triple_lgb_xgb_et'} |
| expansion_8e7654bed66a_variant_1 | expansion_aa3a0506c8ca_variant_1 | ok | 0.814333793676643 | {'amenity_count_routing': 'gam_and_trees'} |
| expansion_11d9504e31b9_variant_2 | expansion_2f8f41dbaafe_variant_1 | ok | 0.8142191820888293 | {'ensemble_diversity': 'triple_lgb_xgb_hgb'} |
| expansion_8e7654bed66a_variant_2 | expansion_aa3a0506c8ca_variant_1 | ok | 0.8142189835699127 | {'amenity_count_routing': 'trees_only'} |
| expansion_11d9504e31b9_variant_1 | expansion_2f8f41dbaafe_variant_1 | ok | 0.8142186527050518 | {'ensemble_diversity': 'dual_lgb_xgb'} |
| expansion_8bc081c33da1_variant_2 | expansion_3e685ab931dd_variant_2 | ok | 0.8142177924564136 | {'meta_regularization': 'C1_0'} |
| expansion_9e15f90eb4a2_variant_2 | expansion_8e7654bed66a_variant_1 | ok | 0.8141035117334606 | {'ensemble_composition': 'balanced_diversity'} |
| expansion_8bc081c33da1_variant_3 | expansion_3e685ab931dd_variant_2 | ok | 0.8141027838307664 | {'meta_regularization': 'C10_0'} |
| expansion_aa3a0506c8ca_variant_1 | expansion_fb23186268c0_variant_2 | ok | 0.8137586182024641 | {'group_coord_routing': 'trees_only'} |
| expansion_aa3a0506c8ca_variant_2 | expansion_fb23186268c0_variant_2 | ok | 0.8136440066146502 | {'group_coord_routing': 'trees_and_gam'} |
| expansion_2ac03900a142_variant_1 | expansion_11d9504e31b9_variant_2 | ok | 0.8136434772308728 | {'bagging_diversity': 'triple_lgb_xgb_rf'} |
| expansion_3e685ab931dd_variant_1 | expansion_51251f25b8c5_variant_1 | ok | 0.8136432125389842 | {'ensemble_weights': 'equal'} |
| expansion_224cdd20eef9_variant_1 | expansion_8bc081c33da1_variant_1 | ok | 0.8136424846362902 | {'feature_set': 'group_and_family'} |
| expansion_9c248db9fb18_variant_1 | expansion_224cdd20eef9_variant_1 | ok | 0.8134131952876903 | {'ensemble_weights': 'tree_emphasized'} |
| expansion_fb23186268c0_variant_3 | expansion_3e685ab931dd_variant_2 | ok | 0.8132979881431268 | {'feature_expansion': 'social_demographic'} |
| expansion_2f8f41dbaafe_variant_2 | expansion_7e8b3ae46375_variant_1 | ok | 0.8131836412472018 | {'regularization': 'regularized'} |
| expansion_224cdd20eef9_variant_2 | expansion_8bc081c33da1_variant_1 | ok | 0.8131828471715355 | {'feature_set': 'full_with_ratios'} |
| expansion_9e15f90eb4a2_variant_1 | expansion_8e7654bed66a_variant_1 | ok | 0.8129531607851026 | {'ensemble_composition': 'quad_heterogeneous'} |
| expansion_9c248db9fb18_variant_3 | expansion_224cdd20eef9_variant_1 | ok | 0.8128380198135113 | {'ensemble_weights': 'tree_dominant'} |
| expansion_f0d7a2ef7fee_variant_1 | expansion_9c248db9fb18_variant_1 | ok | 0.8122625134744714 | {'ensemble_architecture': 'quad_blend'} |
| expansion_fb23186268c0_variant_1 | expansion_3e685ab931dd_variant_2 | ok | 0.811687933557042 | {'feature_expansion': 'full_domain'} |
| expansion_566808a3af1c_variant_1 | expansion_7b11c09be14a_variant_1 | ok | 0.8116876688651533 | {} |
| expansion_2c8181108922_variant_2 | root | ok | 0.8116869409624595 | {'estimator_variant': 'lgb_lr003_500'} |
| expansion_9c248db9fb18_variant_2 | expansion_224cdd20eef9_variant_1 | ok | 0.8113427753341569 | {'ensemble_weights': 'equal'} |
| expansion_f0d7a2ef7fee_variant_2 | expansion_9c248db9fb18_variant_1 | ok | 0.8112271049787882 | {'ensemble_architecture': 'tripartite'} |
| expansion_7b11c09be14a_variant_1 | expansion_295ecf4c72c7_variant_2 | ok | 0.8108817482369866 | {'ensemble_type': 'dual_lgb_xgb'} |
| expansion_7b11c09be14a_variant_2 | expansion_295ecf4c72c7_variant_2 | ok | 0.8107670043032285 | {'ensemble_type': 'triple_lgb_xgb_hgb'} |
| expansion_f0d7a2ef7fee_variant_3 | expansion_9c248db9fb18_variant_1 | ok | 0.8096167195278425 | {'ensemble_architecture': 'equal_quad'} |
| expansion_7e8b3ae46375_variant_1 | expansion_246ebbac5eda_variant_1 | ok | 0.809157809965782 | {} |
| expansion_2c8181108922_variant_1 | root | ok | 0.8088109974185922 | {'estimator_variant': 'lgb_lr005_300'} |
| expansion_ef35b912c334_variant_1 | root | ok | 0.8069702638514921 | {'mlp_config': 'mlp_h128_l3'} |
| expansion_295ecf4c72c7_variant_2 | expansion_cc8d2e249115_variant_1 | ok | 0.8068557184366503 | {'ensemble_weights': 'gbdt_heavy_2_1'} |
| expansion_295ecf4c72c7_variant_1 | expansion_cc8d2e249115_variant_1 | ok | 0.8034051949753538 | {'ensemble_weights': 'equal_1_1'} |
| expansion_c6cbc073a6f2_variant_3 | expansion_ca817ed141c6_variant_2 | ok | 0.8022544469891628 | {'spline_knots': 'knots8'} |
| expansion_51251f25b8c5_variant_1 | expansion_c6cbc073a6f2_variant_3 | ok | 0.8020250914675907 | {'reg_c': 'c0_5'} |
| expansion_51251f25b8c5_variant_3 | expansion_c6cbc073a6f2_variant_3 | ok | 0.8020247606027299 | {'reg_c': 'c2_0'} |
| expansion_51251f25b8c5_variant_2 | expansion_c6cbc073a6f2_variant_3 | ok | 0.8017948756973805 | {'reg_c': 'c1_0'} |
| expansion_ef35b912c334_variant_2 | root | ok | 0.8013339147731822 | {'mlp_config': 'mlp_h256_l2'} |
| expansion_c6cbc073a6f2_variant_2 | expansion_ca817ed141c6_variant_2 | ok | 0.8000687537180939 | {'spline_knots': 'knots6'} |
| expansion_ca817ed141c6_variant_2 | expansion_ceac19dbf271_variant_3 | ok | 0.7997232646303478 | {'reg_c': 'c10'} |
| expansion_ca817ed141c6_variant_3 | expansion_ceac19dbf271_variant_3 | ok | 0.7996081898317287 | {'reg_c': 'c20'} |
| expansion_ca817ed141c6_variant_1 | expansion_ceac19dbf271_variant_3 | ok | 0.7990332128764663 | {'reg_c': 'c05'} |
| expansion_c6cbc073a6f2_variant_1 | expansion_ca817ed141c6_variant_2 | ok | 0.7988029309332838 | {'spline_knots': 'knots4'} |
| expansion_51daba20dc65_variant_1 | expansion_ef35b912c334_variant_1 | ok | 0.7935115415589427 | {} |
| expansion_cc8d2e249115_variant_1 | expansion_51daba20dc65_variant_1 | ok | 0.7875291740091094 | {'resnet_config': 'resnet_2blocks'} |
| expansion_ceac19dbf271_variant_3 | expansion_5c8328badaa0_variant_1 | ok | 0.7853436792569569 | {'reg_c': 'c10'} |
| expansion_cc8d2e249115_variant_2 | expansion_51daba20dc65_variant_1 | ok | 0.7848839756192302 | {'resnet_config': 'resnet_3blocks'} |
| expansion_ceac19dbf271_variant_2 | expansion_5c8328badaa0_variant_1 | ok | 0.7843085354531626 | {'reg_c': 'c02'} |
| expansion_ceac19dbf271_variant_1 | expansion_5c8328badaa0_variant_1 | ok | 0.783618549872253 | {'reg_c': 'c005'} |
| expansion_5c8328badaa0_variant_1 | root | ok | 0.7740701208516991 | {'reg_c': 'c_0_1'} |
| expansion_5c8328badaa0_variant_2 | root | ok | 0.7737250949747583 | {'reg_c': 'c_1_0'} |

## Evaluation audit

{
  "rows": 8693,
  "folds": 5,
  "labels_missing": 0,
  "row_keys": "not declared",
  "fold_sizes": [
    {
      "train": 6954,
      "test": 1739
    },
    {
      "train": 6954,
      "test": 1739
    },
    {
      "train": 6954,
      "test": 1739
    },
    {
      "train": 6955,
      "test": 1738
    },
    {
      "train": 6955,
      "test": 1738
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Findings


All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
