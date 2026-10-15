# Experiment report

Controller: auto; policy: mlevolve; memory: experience; model: gemini/gemini-3.8-flash; state: complete; elapsed: 476 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_6d441e270de6_variant_1 | expansion_339583209699_variant_1 | ok | -3.7086032013498182 | {} |
| expansion_cdcf219995d2_variant_1 | expansion_79e607768d17_variant_1 | ok | -3.7097100013672373 | {} |
| expansion_e54748da9f97_variant_1 | expansion_6d441e270de6_variant_1 | ok | -3.7099943326238014 | {} |
| expansion_79e607768d17_variant_1 | expansion_0b0272d43f2c_variant_1 | ok | -3.7107848555191993 | {} |
| expansion_339583209699_variant_1 | expansion_1ec22b090e49_variant_1 | ok | -3.7111148890525527 | {} |
| expansion_32789fd4b0f7_variant_1 | expansion_cdcf219995d2_variant_1 | ok | -3.7131431005000173 | {} |
| expansion_1ec22b090e49_variant_1 | expansion_32789fd4b0f7_variant_1 | ok | -3.715985081024521 | {} |
| expansion_bffcec8cd316_variant_1 | expansion_e54748da9f97_variant_1 | ok | -3.7166488380660487 | {} |
| expansion_d7522b31df04_variant_1 | expansion_c23ef46729ff_variant_1 | ok | -3.718280209711171 | {} |
| expansion_7df9ed6c69ce_variant_1 | expansion_d7522b31df04_variant_1 | ok | -3.7215966179272613 | {} |
| expansion_c23ef46729ff_variant_1 | expansion_f3929bf8618d_variant_1 | ok | -3.7229701596550577 | {} |
| expansion_0b0272d43f2c_variant_1 | expansion_7df9ed6c69ce_variant_1 | ok | -3.7232585803364344 | {} |
| expansion_814c73cb60c2_variant_1 | expansion_c871315fc259_variant_1 | ok | -3.726575849982063 | {} |
| expansion_9f84affdc518_variant_1 | expansion_814c73cb60c2_variant_1 | ok | -3.7286148696201176 | {} |
| expansion_c871315fc259_variant_1 | expansion_52b56dc7a9ff_variant_1 | ok | -3.730029551468951 | {} |
| expansion_52b56dc7a9ff_variant_1 | expansion_44442cce332c_variant_1 | ok | -3.732984270698817 | {} |
| expansion_55fb5aa30c39_variant_1 | expansion_4abcaef30a20_variant_1 | ok | -3.7415253253021263 | {} |
| expansion_44442cce332c_variant_1 | expansion_55fb5aa30c39_variant_1 | ok | -3.7435950397411433 | {} |
| expansion_f3929bf8618d_variant_1 | expansion_9f84affdc518_variant_1 | ok | -3.74374600298623 | {} |
| expansion_eb3ae4ae555f_variant_1 | expansion_9714f3931858_variant_1 | ok | -3.753924846830581 | {} |
| expansion_9714f3931858_variant_1 | expansion_60d52fc683ef_variant_1 | ok | -3.757329302967706 | {} |
| expansion_4abcaef30a20_variant_1 | expansion_eb3ae4ae555f_variant_1 | ok | -3.7582757891515945 | {} |
| expansion_fa73a7945142_variant_2 | root | ok | -3.8308527565862733 | {'n_estimators': 'trees_600'} |
| expansion_fa73a7945142_variant_1 | root | ok | -3.836439971842254 | {'n_estimators': 'trees_300'} |
| expansion_60d52fc683ef_variant_1 | expansion_29569c52e759_variant_1 | ok | -3.854932178731906 | {} |
| expansion_29569c52e759_variant_1 | expansion_766774700a5e_variant_1 | ok | -3.9014875896671923 | {} |
| expansion_766774700a5e_variant_1 | expansion_cb3d30a5c79f_variant_1 | ok | -3.94174833252413 | {} |
| expansion_cb3d30a5c79f_variant_1 | expansion_c9d6873c30bd_variant_2 | ok | -4.0330156033867315 | {} |
| expansion_c9d6873c30bd_variant_2 | root | ok | -4.042846010949013 | {'arch': 'resnet_4layer'} |
| expansion_c9d6873c30bd_variant_1 | root | ok | -4.113007926798154 | {'arch': 'mlp_3layer'} |

## Evaluation audit

{
  "rows": 975678,
  "folds": 5,
  "labels_missing": 0,
  "row_keys": "not declared",
  "fold_sizes": [
    {
      "train": 780542,
      "test": 195136
    },
    {
      "train": 780542,
      "test": 195136
    },
    {
      "train": 780542,
      "test": 195136
    },
    {
      "train": 780543,
      "test": 195135
    },
    {
      "train": 780543,
      "test": 195135
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Findings


All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
