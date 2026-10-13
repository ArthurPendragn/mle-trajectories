# Experiment report

Controller: llm; policy: draft-greedy; memory: full; model: openai/gpt-6.1-sol (reasoning medium); state: complete; elapsed: 1388 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_2ad7be9bce45_variant_1 | expansion_a81bf9547289_variant_1 | ok | 0.8576233621345878 | {} |
| expansion_a81bf9547289_variant_1 | expansion_c0fa0d55285c_variant_1 | ok | 0.8574187846133436 | {} |
| expansion_c0fa0d55285c_variant_1 | expansion_691701a593e6_variant_1 | ok | 0.8539052923827676 | {} |
| expansion_ee9dccce2b12_variant_1 | expansion_c0fa0d55285c_variant_1 | ok | 0.8531087994913485 | {} |
| expansion_691701a593e6_variant_1 | expansion_a40066de017c_variant_2 | ok | 0.8438334075633632 | {} |
| expansion_a40066de017c_variant_2 | expansion_f734fb1fba7e_variant_2 | ok | 0.8423420193537154 | {'graph_tree_capacity': 'trees300_depth6'} |
| expansion_5c91999d1a3a_variant_1 | expansion_a81bf9547289_variant_1 | ok | 0.8415400673721191 | {} |
| expansion_a40066de017c_variant_1 | expansion_f734fb1fba7e_variant_2 | ok | 0.8322104990048506 | {'graph_tree_capacity': 'trees200_depth4'} |
| expansion_f734fb1fba7e_variant_2 | expansion_4c259dfc830d_variant_3 | ok | 0.828368750921675 | {'mixture_gate': 'domain'} |
| expansion_f734fb1fba7e_variant_1 | expansion_4c259dfc830d_variant_3 | ok | 0.8235522590129367 | {'mixture_gate': 'global'} |
| expansion_4c259dfc830d_variant_3 | root | ok | 0.8142373967918246 | {'neighborhood_direction': 'both'} |
| expansion_3a1f01247caf_variant_1 | expansion_f734fb1fba7e_variant_2 | ok | 0.8097937539141681 | {'joint_residual_width': 128} |
| expansion_3a1f01247caf_variant_2 | expansion_f734fb1fba7e_variant_2 | ok | 0.7946347422242303 | {'joint_residual_width': 256} |
| expansion_4c259dfc830d_variant_1 | root | ok | 0.7805197224810826 | {'neighborhood_direction': 'outgoing'} |
| expansion_4c259dfc830d_variant_2 | root | ok | 0.7709865186520549 | {'neighborhood_direction': 'incoming'} |
| expansion_b9b332d6c99a_variant_1 | root | ok | 0.7408683359473671 | {'linear_regularization': 'weak_l2'} |
| expansion_f3512dc1d68b_variant_1 | root | ok | 0.7301858113866864 | {} |
| expansion_b9b332d6c99a_variant_2 | root | ok | 0.7252463958058524 | {'linear_regularization': 'moderate_l2'} |
| expansion_57f5c3c665b1_variant_1 | expansion_2ad7be9bce45_variant_1 | failed | None | {} |

## Evaluation audit

{
  "rows": 169000,
  "folds": 3,
  "labels_missing": 0,
  "row_keys": "unique, nonmissing, aligned",
  "fold_sizes": [
    {
      "train": 160000,
      "test": 3000
    },
    {
      "train": 160000,
      "test": 3000
    },
    {
      "train": 160000,
      "test": 3000
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Probes (out-of-fold predictions, not scored candidates)

- probe_390105c476e3 (expansion_4c259dfc830d_variant_3): Report stored fold scores and out-of-fold errors, stratifying domain Recall@10 by hyperlink connectivity, available labelled-neighbor coverage, TLD, and true tracker-set size. Identify trackers contributing most missed recall and compare errors with the hostname-only linear candidate where stored predictions permit. score 0.81424
- probe_5c0742ca4fe7 (expansion_2ad7be9bce45_variant_1): Without refitting, report stored fold scores and summarize multilabel out-of-fold errors using per-domain Recall@10: recall distribution, recall by true tracker-count band, and trackers contributing the most missed domain-normalized recall. Return compact summaries rather than raw label arrays. score 0.85762

## Findings

- finding_8f5c052ae609: The sources contain 46,269,087 domains, 36,674,685 tracking edges and 623,056,313 hyperlink edges. There are 18,682,899 labelled domains and 50,000 targets, with zero target–labelled-domain overlap. (Source inventory and prediction-domain separation)
- finding_f36a66fa1787: Observed label sets are sparse: mean tracker count is 1.963, median 1, 90th percentile 4 and maximum 346; 53.78% of labelled domains are singletons. These summaries do not establish that recorded labels are complete. (Recorded domain-level tracker labels)
- finding_ae7248cd1b4c: Among the visible population comparisons, domains with at least two trackers are least distinguishable from targets: 8,635,505 domains, adversarial AUC 0.4897 ± 0.0053, versus 0.5557 ± 0.0068 for all labelled domains. (Visible three-fold adversarial comparisons)
- finding_26c89dc274db: Increasing lower tracker-count thresholds beyond two makes populations more distinguishable from targets, especially at high thresholds. (Visible lower-threshold population comparisons)
- finding_e6d0c8b04556: None of the visible upper tracker-count thresholds matches the near-chance AUC of the at-least-two population. (Visible upper-threshold population comparisons)
- finding_f70b3714ee8a: Targets may have been selected from domains with at least two trackers, but observable similarity alone does not prove this construction or justify discarding singleton training domains. (Target-selection hypothesis and population choice)
- finding_2958c39c5edd: Hostname and hyperlink-degree observables are available for both populations. Targets have slightly higher typical outbound connectivity and different TLD proportions, despite similar hostname lengths. (Visible label-free distribution summaries)
- finding_93815893debc: The press-freedom file needs tab-separated parsing despite its .csv extension; the displayed parse incorrectly collapses its three fields into one column. (Press-freedom metadata ingestion)
- finding_c4f5959bff2c: In this run, populations with at least two trackers, two to five trackers, and two to ten trackers have near-chance adversarial AUCs whose differences are within fold variability; the at-least-two rule retains the most domains of these three. (Visible population comparisons in the second exploration)
- finding_98488dbcf7ac: Filtering for the presence of incoming or outgoing hyperlinks makes the visible populations more distinguishable from targets than using all labelled domains. (Observable-based population rules in this run)
- finding_9f8ada062e4d: The displayed results do not establish whether excluding singleton domains improves Recall@10, nor support a timed learning-curve pool-size ceiling or a verified leakage-safe graph-feature procedure. (Evidence available in the returned preview, not the undisplayed artifact rows)
- finding_95465700e9c6: Press-freedom metadata is now displayed with its three fields correctly separated, resolving the earlier one-column parsing issue. (Press-freedom metadata ingestion preview)
- finding_c790032c5303: At equal training sizes, restricting training to domains with at least two recorded trackers improves mean Recall@10 on the shared at-least-two held-out population at every measured size and in every fold. This does not establish improvement on singleton domains or actual targets. (Hostname-only MLP; paired population comparison on 9,000 held-out domains with at least two trackers)
- finding_f503d8aadbf2: The learned-model curve shows diminishing returns beyond 40,000 training domains per fold for both populations. (Measured 10k–160k hostname-only learning curve)
- finding_4dc62ddd2a46: Reported timings cover feature fitting/building, model fitting and scoring across three test sets, but exclude source ingestion and graph joins. (Learning-curve timing coverage)
- finding_2e9d0eadff98: The reported evaluation excludes all 9,000 held-out domains from both training pools and uses three disjoint 3,000-domain test sets with mean per-domain Recall@10. (Declared domain-disjoint evaluation design)
- finding_40c028d4a395: The at-least-two population is a supported candidate for further experiments on similarly selected validation domains, but the comparison alone does not justify universally discarding singleton training labels. (Training-population recommendation conditional on validation representativeness)
- finding_2902470a0963: No numeric extrapolated pool-size ceiling or finalized training-size recommendation is visible; the output provides a sizing intention rather than a budget calculation guaranteeing twelve experiments. (Remaining pool-sizing evidence gap)
- finding_dcc38e15addd: This curve uses hostnames only, so it does not test graph features or demonstrate an implemented leakage-safe graph-label aggregation. (Graph-feature scope and declared future safeguards)
- finding_02c9a2755cee: The requested prediction-based error analysis was not computed; the returned tables audit feature availability, not model error concentrations or candidate complementarity. (This exploration's execution and evidence limits)
- finding_b084bc8ada63: Content metadata is available for only 2.304% of targets, closely matching the 2.067–2.600% availability across validation folds. (Joined content-metadata availability on 50,000 targets and three locked validation sets)
- finding_782fc2dbbad5: Most targets have hyperlink connectivity, but 7.078% are disconnected; validation disconnected fractions are 6.467–7.067%. (Incident hyperlink availability)
- finding_c0ac4d42bb9f: Disjoint labelled-neighbor evidence is available for 90.706% of targets; 4,647 targets lack it. Validation coverage is 90.833–91.800%. (Distinct incident neighbors with recorded labels outside the entire 169,000-domain locked pool)
- finding_ef7cbc38276c: Mean incoming connectivity varies substantially across validation folds, so similar binary coverage does not establish matching neighborhood-size distributions. (Target versus locked-validation connectivity summaries)
- finding_b9b2922b2025: Domains lacking labelled-neighbor evidence form a prediction-relevant subgroup worth evaluating, but the audit does not show that their recall is worse or identify a successful remedy. (Conditional priority for subsequent error analysis)

All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
