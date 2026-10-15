# Experiment report

Controller: auto; policy: mlevolve; memory: experience; model: gemini/gemini-3.8-flash; state: complete; elapsed: 480 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_0c227677b9bc_variant_1 | expansion_08017e586d6a_variant_1 | ok | -3.703295592647187 | {} |
| expansion_08017e586d6a_variant_1 | expansion_c7523c88664c_variant_1 | ok | -3.7060218376214102 | {} |
| expansion_c7523c88664c_variant_1 | expansion_f5f8799f8a79_variant_1 | ok | -3.7066228091915896 | {} |
| expansion_adbd3c17140f_variant_1 | expansion_e7ef10130d00_variant_1 | ok | -3.70804962545989 | {} |
| expansion_345cf988a521_variant_1 | expansion_6745e4fcdf3f_variant_1 | ok | -3.712612899222949 | {} |
| expansion_6745e4fcdf3f_variant_1 | expansion_adbd3c17140f_variant_1 | ok | -3.712723366972706 | {} |
| expansion_da40d76547ce_variant_1 | expansion_c0ee5061d65c_variant_1 | ok | -3.7142274198949465 | {} |
| expansion_1e5782ded162_variant_1 | expansion_a19de95b97f3_variant_1 | ok | -3.715035838871741 | {} |
| expansion_c9213d1bc3b8_variant_1 | expansion_1e5782ded162_variant_1 | ok | -3.717563648071929 | {} |
| expansion_2529424c47b8_variant_1 | expansion_c9213d1bc3b8_variant_1 | ok | -3.717901150505765 | {} |
| expansion_f5f8799f8a79_variant_1 | expansion_2529424c47b8_variant_1 | ok | -3.7211240862259816 | {} |
| expansion_e7ef10130d00_variant_1 | expansion_aea63afb6a43_variant_1 | ok | -3.724815525699462 | {} |
| expansion_aea63afb6a43_variant_1 | expansion_ba0fb8c92eea_variant_1 | ok | -3.7309250389467854 | {} |
| expansion_ba0fb8c92eea_variant_1 | expansion_1bb2585da3a9_variant_1 | ok | -3.732892291922242 | {} |
| expansion_c0ee5061d65c_variant_1 | expansion_345cf988a521_variant_1 | ok | -3.7401527859343764 | {} |
| expansion_1bb2585da3a9_variant_1 | expansion_ad27b86d2df8_variant_1 | ok | -3.7476166074459045 | {} |
| expansion_a19de95b97f3_variant_1 | expansion_c18b67119403_variant_1 | ok | -3.75648193420797 | {} |
| expansion_ad27b86d2df8_variant_1 | expansion_ea70ab83ca05_variant_1 | ok | -3.7590328788762184 | {} |
| expansion_c18b67119403_variant_1 | expansion_8664b1afe9c4_variant_1 | ok | -3.7661344993007804 | {} |
| expansion_8664b1afe9c4_variant_1 | expansion_1c64cebcf762_variant_1 | ok | -3.77806601488464 | {} |
| expansion_053a93a8b8d0_variant_1 | expansion_a01a6c205a13_variant_1 | ok | -3.783395110310083 | {} |
| expansion_ea70ab83ca05_variant_1 | expansion_053a93a8b8d0_variant_1 | ok | -3.7855571174411353 | {} |
| expansion_a01a6c205a13_variant_1 | expansion_d27cafdab917_variant_1 | ok | -3.7896679132019697 | {} |
| expansion_d27cafdab917_variant_1 | expansion_864c4c8b1195_variant_1 | ok | -3.798273421098095 | {} |
| expansion_1c64cebcf762_variant_1 | expansion_22ff9af10ce0_variant_1 | ok | -3.8030211891999075 | {} |
| expansion_b25172d0f878_variant_1 | root | ok | -3.8111489451650646 | {} |
| expansion_864c4c8b1195_variant_1 | expansion_81c396d2004d_variant_1 | ok | -3.994672679963852 | {} |
| expansion_81c396d2004d_variant_1 | expansion_6f7c6da27044_variant_1 | ok | -4.001604451751218 | {} |
| expansion_6f7c6da27044_variant_1 | root | ok | -4.169938773497178 | {} |
| expansion_a63f00b853d9_variant_1 | expansion_fde6899a2ed2_variant_1 | ok | -4.478553530884492 | {'ridge_alpha': 'alpha_1'} |
| expansion_a63f00b853d9_variant_2 | expansion_fde6899a2ed2_variant_1 | ok | -4.478561216232228 | {'ridge_alpha': 'alpha_5'} |
| expansion_a63f00b853d9_variant_3 | expansion_fde6899a2ed2_variant_1 | ok | -4.478578300771963 | {'ridge_alpha': 'alpha_10'} |
| expansion_22ff9af10ce0_variant_1 | expansion_a63f00b853d9_variant_1 | ok | -4.485461375206632 | {} |
| expansion_fde6899a2ed2_variant_1 | expansion_a75815f26569_variant_1 | ok | -4.565322819641681 | {'ridge_alpha': 'alpha_1'} |
| expansion_fde6899a2ed2_variant_2 | expansion_a75815f26569_variant_1 | ok | -4.56532994389783 | {'ridge_alpha': 'alpha_10'} |
| expansion_fde6899a2ed2_variant_3 | expansion_a75815f26569_variant_1 | ok | -4.565662424724124 | {'ridge_alpha': 'alpha_100'} |
| expansion_a75815f26569_variant_1 | expansion_5429ea437ef1_variant_1 | ok | -4.767383890971329 | {} |
| expansion_5429ea437ef1_variant_1 | root | ok | -4.864389836688108 | {} |
| expansion_3f38da571d0b_variant_1 | expansion_0c227677b9bc_variant_1 | failed | None | {} |

## Evaluation audit

{
  "rows": 975671,
  "folds": 5,
  "labels_missing": 0,
  "row_keys": "unique, nonmissing, aligned",
  "fold_sizes": [
    {
      "train": 780536,
      "test": 195135
    },
    {
      "train": 780537,
      "test": 195134
    },
    {
      "train": 780537,
      "test": 195134
    },
    {
      "train": 780537,
      "test": 195134
    },
    {
      "train": 780537,
      "test": 195134
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Findings


All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
