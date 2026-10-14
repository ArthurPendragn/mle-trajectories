# Experiment report

Controller: auto; policy: mlevolve; memory: experience; model: gemini/gemini-3.8-flash; state: complete; elapsed: 280 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_17fe71b80f64_variant_1 | expansion_a026af80e2e8_variant_1 | ok | -3.7184353173402456 | {} |
| expansion_a026af80e2e8_variant_1 | expansion_e58a1a002cc9_variant_1 | ok | -3.718603179709781 | {} |
| expansion_e58a1a002cc9_variant_1 | expansion_249c2ccb682c_variant_1 | ok | -3.7236934664791184 | {} |
| expansion_ee64f7828a2d_variant_1 | expansion_17fe71b80f64_variant_1 | ok | -3.7297411568878545 | {} |
| expansion_a25527996291_variant_1 | expansion_791e5999f3ac_variant_1 | ok | -3.7300206715144766 | {} |
| expansion_249c2ccb682c_variant_1 | expansion_a25527996291_variant_1 | ok | -3.7314465638349574 | {} |
| expansion_18910ee4bd7a_variant_1 | expansion_ee64f7828a2d_variant_1 | ok | -3.7320532875586423 | {} |
| expansion_791e5999f3ac_variant_1 | expansion_0cdbb5baf033_variant_1 | ok | -3.7356213631976294 | {} |
| expansion_0cdbb5baf033_variant_1 | expansion_15b2e8b227ac_variant_1 | ok | -3.747383346421079 | {} |
| expansion_15b2e8b227ac_variant_1 | expansion_37bc47a58e12_variant_1 | ok | -3.7573297413698734 | {} |
| expansion_c2c5bc2e0783_variant_1 | expansion_a01092fd9ddc_variant_1 | ok | -3.7945539743194936 | {} |
| expansion_37bc47a58e12_variant_1 | expansion_c2c5bc2e0783_variant_1 | ok | -3.7949042487668705 | {} |
| expansion_93fb479263a4_variant_1 | root | ok | -3.810470894626737 | {} |
| expansion_a01092fd9ddc_variant_1 | expansion_fdd25d5e5db8_variant_1 | ok | -3.816225406444578 | {} |
| expansion_26e114be9d45_variant_1 | root | ok | -3.993857211290911 | {} |
| expansion_fdd25d5e5db8_variant_1 | expansion_7c5ed7c9a619_variant_1 | ok | -3.995695717219617 | {} |
| expansion_7c5ed7c9a619_variant_1 | expansion_26e114be9d45_variant_1 | ok | -4.028281135557994 | {} |

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
