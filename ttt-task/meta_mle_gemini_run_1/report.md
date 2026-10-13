# Experiment report

Controller: llm; policy: draft-greedy; memory: full; model: gemini/gemini-3.8-flash (reasoning high); state: complete; elapsed: 550 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_81168d9ad604_variant_2 | expansion_49aeece00b3b_variant_4 | ok | 0.8475986986406265 | {'model_variant': 'damped_neighbors'} |
| expansion_81168d9ad604_variant_5 | expansion_49aeece00b3b_variant_4 | ok | 0.8427506833234704 | {'model_variant': 'combined_synergy'} |
| expansion_49aeece00b3b_variant_4 | expansion_2037d2a1793a_variant_2 | ok | 0.8408992308297136 | {'model_variant': 'ridge_neighbors_diffuse15'} |
| expansion_81168d9ad604_variant_1 | expansion_49aeece00b3b_variant_4 | ok | 0.8408992308297136 | {'model_variant': 'parent_baseline'} |
| expansion_81168d9ad604_variant_4 | expansion_49aeece00b3b_variant_4 | ok | 0.8407072070201901 | {'model_variant': 'adaptive_calibration'} |
| expansion_49aeece00b3b_variant_5 | expansion_2037d2a1793a_variant_2 | ok | 0.8404918256638443 | {'model_variant': 'ridge_neighbors_diffuse30'} |
| expansion_49aeece00b3b_variant_3 | expansion_2037d2a1793a_variant_2 | ok | 0.8398524720934379 | {'model_variant': 'ridge_combined_neighbors'} |
| expansion_81168d9ad604_variant_3 | expansion_49aeece00b3b_variant_4 | ok | 0.8352170830739608 | {'model_variant': 'thresholded_diffusion'} |
| expansion_49aeece00b3b_variant_2 | expansion_2037d2a1793a_variant_2 | ok | 0.8299659256191007 | {'model_variant': 'ridge_out_neighbors'} |
| expansion_2037d2a1793a_variant_2 | expansion_67b6905cf85a_variant_5 | ok | 0.7826585539610029 | {'model_variant': 'ridge_regional_calib20'} |
| expansion_49aeece00b3b_variant_1 | expansion_2037d2a1793a_variant_2 | ok | 0.7826585539610029 | {'model_variant': 'parent_baseline'} |
| expansion_2037d2a1793a_variant_3 | expansion_67b6905cf85a_variant_5 | ok | 0.7825772500324752 | {'model_variant': 'ridge_regional_calib23'} |
| expansion_2037d2a1793a_variant_5 | expansion_67b6905cf85a_variant_5 | ok | 0.7820362891506498 | {'model_variant': 'ridge_regional_a150_calib23'} |
| expansion_2037d2a1793a_variant_4 | expansion_67b6905cf85a_variant_5 | ok | 0.7817868594034965 | {'model_variant': 'ridge_regional_calib26'} |
| expansion_67b6905cf85a_variant_5 | expansion_3f35d8ec61ba_variant_2 | ok | 0.7808848009239642 | {'model_variant': 'ridge_graph_links_calib20'} |
| expansion_2037d2a1793a_variant_1 | expansion_67b6905cf85a_variant_5 | ok | 0.7808848009239642 | {'model_variant': 'parent_baseline'} |
| expansion_67b6905cf85a_variant_4 | expansion_3f35d8ec61ba_variant_2 | ok | 0.7798007880765605 | {'model_variant': 'ridge_graph_links_calib15'} |
| expansion_67b6905cf85a_variant_2 | expansion_3f35d8ec61ba_variant_2 | ok | 0.7764462687266294 | {'model_variant': 'ridge_graph_links'} |
| expansion_67b6905cf85a_variant_3 | expansion_3f35d8ec61ba_variant_2 | ok | 0.7618236753393666 | {'model_variant': 'ridge_graph_calib20'} |
| expansion_3f35d8ec61ba_variant_2 | expansion_3933bcb33fa5_variant_4 | ok | 0.7582716924713573 | {'model_variant': 'ridge_alpha100_graph'} |
| expansion_67b6905cf85a_variant_1 | expansion_3f35d8ec61ba_variant_2 | ok | 0.7582716924713573 | {'model_variant': 'ridge_graph_baseline'} |
| expansion_3f35d8ec61ba_variant_3 | expansion_3933bcb33fa5_variant_4 | ok | 0.7500124434972492 | {'model_variant': 'ridge_alpha300_graph'} |
| expansion_3f35d8ec61ba_variant_4 | expansion_3933bcb33fa5_variant_4 | ok | 0.7405509118899676 | {'model_variant': 'ridge_alpha1000_graph'} |
| expansion_3933bcb33fa5_variant_4 | root | ok | 0.7369563185395616 | {'model_variant': 'ridge_alpha_100.0'} |
| expansion_3f35d8ec61ba_variant_1 | expansion_3933bcb33fa5_variant_4 | ok | 0.7369563185395616 | {'model_variant': 'ridge_alpha100_nograph'} |
| expansion_3933bcb33fa5_variant_3 | root | ok | 0.735794267727027 | {'model_variant': 'ridge_alpha_10.0'} |
| expansion_9ff014c5b2b5_variant_4 | root | ok | 0.729272022740933 | {'model_variant': 'knn_k60_uniform'} |
| expansion_9ff014c5b2b5_variant_3 | root | ok | 0.7265972841947355 | {'model_variant': 'knn_k60_cosine'} |
| expansion_0335c4003705_variant_4 | root | ok | 0.7213776993751002 | {'model_variant': 'mlp_256_128_drop0.2'} |
| expansion_9ff014c5b2b5_variant_2 | root | ok | 0.7065983241495347 | {'model_variant': 'knn_k30_cosine'} |
| expansion_0335c4003705_variant_3 | root | ok | 0.697516924387532 | {'model_variant': 'mlp_256_128_drop0.1'} |
| expansion_3933bcb33fa5_variant_2 | root | ok | 0.683831751434582 | {'model_variant': 'ridge_alpha_1.0'} |
| expansion_9ff014c5b2b5_variant_1 | root | ok | 0.6701249218236184 | {'model_variant': 'knn_k15_cosine'} |
| expansion_0335c4003705_variant_2 | root | ok | 0.669013849889189 | {'model_variant': 'mlp_512_256_drop0.2'} |
| expansion_0335c4003705_variant_1 | root | ok | 0.6408750977566464 | {'model_variant': 'mlp_512_256_drop0.1'} |
| expansion_3933bcb33fa5_variant_1 | root | ok | 0.6334928276473268 | {'model_variant': 'ridge_alpha_0.1'} |
| expansion_0c652b872b29_variant_1 | expansion_81168d9ad604_variant_2 | failed | None | {} |

## Evaluation audit

{
  "rows": 50000,
  "folds": 5,
  "labels_missing": 0,
  "row_keys": "not declared",
  "fold_sizes": [
    {
      "train": 40000,
      "test": 10000
    },
    {
      "train": 40000,
      "test": 10000
    },
    {
      "train": 40000,
      "test": 10000
    },
    {
      "train": 40000,
      "test": 10000
    },
    {
      "train": 40000,
      "test": 10000
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Probes (out-of-fold predictions, not scored candidates)

- probe_02f94e0dd794 (expansion_3933bcb33fa5_variant_4): How are out-of-fold prediction errors and recall losses distributed across tracker frequency strata, link-graph degree levels, and domain length bins in the leading regularized linear candidate? score 0.73696

## Findings

- finding_3f06f6ef7f66: The tracking graph contains 36,674,685 edges across 18,682,899 unique training domains and 355 trackers, with zero domain overlap with the 50,000 target domains. (dataset_overview & domain_observables)
- finding_6620934bfb09: Target domains closely match the distribution of training domains that have between 2 and 20 trackers (adversarial ROC AUC ~ 0.499), whereas using all training domains shows noticeable shift (AUC 0.573). (adversarial_selection_rules)
- finding_0e6958d4dc76: Tracker distribution is heavily concentrated in a small head, led by Google Analytics at 61.6% coverage and Google Advertising at 23.0% coverage. (tracker_distributions & domain_observables)
- finding_ad54bb45c7af: Target and training domains share nearly identical hostname lengths and high link-graph presence, with target domains exhibiting slightly higher average link degrees. (domain_observables)
- finding_08d5433b38f4: Subsampled training pools (25k-50k domains) yield rapid fold runtimes (~1.7-2.8s) while scoring higher Recall@10 than training on all 18.7M domains under the benchmark MLP setup. (learning_curve)
- finding_79f6cae71cdc: Filtering or sampling candidate training domains based on 2 <= tracker_count <= 20 will provide an unbiased, target-aligned validation set and fast training pipeline. (validation_strategy)
- finding_557e00f73e23: Direct hyperlinks from domains to candidate tracker domains appear on ~13.9% of domains with a high overall precision of 54.6%, reaching over 93% precision for specific widget and beacon trackers. (tracker_hyperlinks_study)
- finding_4abb4402275b: The parent Ridge model exhibits severe head-tracker bias with 98.2% head recall versus 40.1% mid and 3.3% tail, and frequency-calibrated power scaling boosts overall Recall@10 from 0.7582 to 0.7628 (+0.0046). (tracker_strata_and_calibration_study)
- finding_fb3f3d68b786: Content category metadata from url-classification.csv covers only ~2.3% of domains, whereas ccTLDs join 27.6% of domains to press freedom scores and capture concentrated regional tracker ecosystems. (external_metadata_coverage)
- finding_84a5df6dd618: Integrating direct tracker hyperlink indicators into feature representations and applying frequency-calibrated ranking will provide substantial, orthogonal improvements to tracker retrieval on mid-frequency and specialized trackers. (model_pipeline_hypotheses)
- finding_635d29b826b9: Disjoint 1-hop web graph neighbors cover ~91% of domains (including 89.6% of unlinked domains), and out-neighbor tracker profiles boost standalone Recall@10 on unlinked domains from 0.7076 to 0.7743. (one_hop_neighbor_study)
- finding_3436aefc8d46: 2-hop hyperlink paths reach ~57.4% of domains but exhibit low precision (1.72%), and naively ranking candidates by 2-hop paths degrades Recall@10 from 0.6892 to 0.5933. (two_hop_tracker_link_study)
- finding_d02a683a5b64: Candidate trackers exhibit strong modular co-occurrence clustering, with top paired trackers displaying up to 202x lift and conditional co-occurrence probabilities exceeding 90%. (tracker_cooccurrence_study)
- finding_370ffb1fda33: Integrating aggregated 1-hop neighbor tracker frequency distributions into feature representations will significantly boost Recall@10 on the 86% of domains lacking direct tracker hyperlinks while avoiding the noise of multi-hop paths. (model_pipeline_hypotheses)
- finding_0e6c6c1a7fa5: Weighting neighbor tracker profiles by inverse neighbor degree drastically suppresses hub noise, boosting candidate tracker precision from 1.81% on low-authority links to 37.07% on high-authority links. (hub_damped_neighbors_study)
- finding_04424906f7b8: Graph-isolated domains represent ~9.0% of domains, have fewer true trackers on average (2.58 vs 3.12 for connected domains), and are disproportionately concentrated in generic TLDs like .com. (subgroup_recall_breakdown & lexical_knn_isolated_study)
- finding_2d3608e019c6: Tracker co-occurrences feature highly asymmetric, near-deterministic implication rules with conditional probabilities exceeding 90% and lifts exceeding 200x. (cooccurrence_diffusion_study)
- finding_aa1f2676fa90: Incorporating authority-weighted neighbor aggregation (e.g. inverse degree or Adamic-Adar damping) will significantly improve Recall@10 over uniform neighbor counts by filtering non-specific tracker noise propagated through massive web hubs. (model_pipeline_hypotheses)

All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
