# Experiment report

Controller: llm; policy: draft-greedy; memory: full; model: openai/gpt-6.1-sol (reasoning medium); state: complete; elapsed: 947 min

| Candidate | Parent | Status | Score | Configuration |
|---|---|---|---:|---|
| expansion_00a937188bb8_variant_2 | expansion_728623f58d1d_variant_2 | ok | 0.8838668893499002 | {'two_hop_evidence': 'shared_destination_and_source'} |
| expansion_de4b02b79e03_variant_1 | expansion_00a937188bb8_variant_2 | ok | 0.8833209513282269 | {} |
| expansion_4365594765e3_variant_1 | expansion_00a937188bb8_variant_2 | ok | 0.8831580844973453 | {} |
| expansion_0536ffa28893_variant_1 | expansion_00a937188bb8_variant_2 | ok | 0.8821205770192999 | {} |
| expansion_00a937188bb8_variant_1 | expansion_728623f58d1d_variant_2 | ok | 0.8794452363025118 | {'two_hop_evidence': 'shared_destination_only'} |
| expansion_728623f58d1d_variant_2 | expansion_03b365ebffa6_variant_1 | ok | 0.8652270902429184 | {'hostname_signal': 'relational_plus_character_ngrams'} |
| expansion_77b388a6fe82_variant_1 | expansion_728623f58d1d_variant_2 | ok | 0.8652270902429184 | {'retrieval_objective': 'inverse_cardinality_bce'} |
| expansion_03b365ebffa6_variant_1 | expansion_4eb94a5a4098_variant_1 | ok | 0.8648372862968644 | {} |
| expansion_728623f58d1d_variant_1 | expansion_03b365ebffa6_variant_1 | ok | 0.8648372862968644 | {'hostname_signal': 'relational_only'} |
| expansion_77b388a6fe82_variant_2 | expansion_728623f58d1d_variant_2 | ok | 0.8646230863764145 | {'retrieval_objective': 'normalized_positive_softmax'} |
| expansion_4eb94a5a4098_variant_1 | root | ok | 0.8452669113430143 | {'linear_l2': 'l2_0_0001'} |
| expansion_4eb94a5a4098_variant_2 | root | ok | 0.8415005882289938 | {'linear_l2': 'l2_0_001'} |
| expansion_5ab9f9b22745_variant_1 | expansion_728623f58d1d_variant_2 | ok | 0.8323667937755399 | {} |
| expansion_452b67ac77ec_variant_1 | expansion_00a937188bb8_variant_2 | ok | 0.8047299763045119 | {} |
| expansion_9a85b3316807_variant_1 | root | ok | 0.7979730002639833 | {} |
| expansion_2e8d84fc50df_variant_1 | root | failed | None | {} |
| expansion_ea3b76dfad30_variant_1 | expansion_00a937188bb8_variant_2 | failed | None | {} |

## Evaluation audit

{
  "rows": 1206000,
  "folds": 3,
  "labels_missing": 0,
  "row_keys": "unique, nonmissing, aligned",
  "fold_sizes": [
    {
      "train": 1200000,
      "test": 2000
    },
    {
      "train": 1200000,
      "test": 2000
    },
    {
      "train": 1200000,
      "test": 2000
    }
  ]
}

Inputs are assumed static/frozen; source contents are not checked.

## Probes (out-of-fold predictions, not scored candidates)

- probe_9c46b8b65931 (expansion_03b365ebffa6_variant_1): Where does the best candidate still miss true trackers? Report locked-fold Recall@10 and out-of-fold error summaries, separating domains by available neighbourhood coverage, known tracker cardinality, geography, and tracker frequency. Identify whether missed trackers fall just below the ten-prediction cutoff or rank far below it, and show representative failures. Use validation labels only for scoring and error analysis. score 0.86484

## Findings

- finding_c685e6991c33: The source data are large: 36,674,685 tracking-edge rows, 46,269,087 domain-lookup rows, 623,056,313 hyperlink rows, and 50,000 target rows. The processed classification source has 1,562,978 rows. (Source-size diagnostics; row counts do not establish unique-domain counts.)
- finding_bcfa4cbfd56f: The reported duplicate counts are zero for tracking labels and targets; consequently, the target file contains 50,000 distinct domain IDs. The exact duplicate-check columns for tracking labels are not shown. (Duplicate diagnostics for the tracking-label source and target.tsv.)
- finding_d3be83e371d9: Known tracker counts are concentrated among domains with few labels, but there is a substantial higher-count tail. There are 10,047,394 domains with one known tracker, 4,322,219 with two, and 2,175,329 with three. These observed counts do not establish that labels are complete. (Known-label cardinality distribution, not the unobserved true tracker distribution.)
- finding_d86fa0fac7d6: The tracker-frequency output contains 355 trackers and shows strong frequency imbalance. Google Analytics is observed on 11,507,671 domains, versus 4,292,906 for googlesyndication.com and 2,126,231 for facebook.com. (Tracker presence in the supplied training graph.)
- finding_a23c377ddeb5: The target preview demonstrates construction of hostname, TLD, metadata-coverage, hyperlink-degree, and classified-neighbourhood observables. It also shows heterogeneous graph coverage and missing metadata. It does not verify which columns were actually used in an adversarial classifier. (The twelve displayed target rows and their feature schema only.)
- finding_c5026b48fc4b: The reported relevant outgoing and incoming hyperlink subsets remain very large. Their sizes alone do not establish a minimal graph slice for subsequent experiments or a leakage-safe supervised-feature construction. (Reported hyperlink subset sizes; filtering definitions and masking implementation are unavailable.)
- finding_992de234886a: The visible result does not support choosing a representative labelled population or a training-pool size that permits twelve learned-model experiments. The required adversarial comparisons, learned-model learning curve, and fold-level timing evidence are not readable in the supplied output. (Limitations of the returned exploration evidence, rather than a claim that the underlying artifact contains no additional results.)
- finding_d24f51fc2d4e: Target overlap with known tracker edges, full labelled-domain uniqueness, label completeness, and the target-selection mechanism remain unresolved by the visible diagnostics. Zero label_count in the displayed target rows cannot establish these properties for all targets. (Population and label-availability conclusions requiring additional diagnostics.)
- finding_88b72b72884e: The training graph covers 18,682,899 unique labelled domains, and no target domain overlaps those known tracker labels. Label completeness and the target-selection mechanism remain unresolved. (Reported full-population diagnostics; absence of supplied labels is not evidence that targets have no trackers.)
- finding_89d1f7d589d9: Among the reported comparisons, selecting domains with at least two distinct known trackers gives the lowest adversarial AUC and retains 8,635,505 domains. Its near-chance AUC supports observable similarity under this classifier, not equivalence of tracker-label distributions. (Sampled adversarial comparison using the reported label-free features and CatBoost classifier.)
- finding_73683f3eb493: The unrestricted labelled population shows meaningful observable shift from targets, and neither upper tracker-count thresholds nor the displayed metadata/graph filters improve on the at-least-two population. (Visible adversarial-comparison rows; two of the 22 artifact rows are not displayed.)
- finding_d32172a4ae32: The adversarial protocol explicitly excludes tracker counts, tracker labels, target indicators, and domain IDs from predictors, but balanced sample sizes are not identical for every rule. (Reported protocol, not an independent audit of execution code.)
- finding_f84246aff172: Exact retained population counts are available for all 16 tracker-count rules, but full-population retained counts for metadata and graph filters are not established by the visible summary. (Completeness of returned comparison diagnostics.)
- finding_0dba513f20e0: A genuinely learned CUDA multilabel MLP improves mean domain-averaged Recall@10 from 0.699393 at 10,000 training domains per fold to 0.754220 at 40,000 and 0.771060 at 120,000. Improvement slows but the measured curve does not establish a plateau. (Three-fold learning curve on the selected population, with 2,000 validation domains per fold; not measured target-set performance.)
- finding_857921d32508: Reported complete three-fold experiment times are approximately 60.16, 59.14, and 73.22 seconds at 10,000, 40,000, and 120,000 training domains per fold, respectively. These include shared source-feature construction rather than fitting time alone. (Timing of this lightweight MLP and sampled feature pipeline; not a timing guarantee for larger pools or supervised graph models.)
- finding_b9ca1dad41c0: A conservative working pool is supported by the largest measured experiment: 120,000 training domains plus three disjoint 2,000-domain validation subsets. The reported 126,000-domain pool is not demonstrated to be the largest feasible pool. (Planning subsequent experiments with the measured MLP; larger sizes require further measurement.)
- finding_e83cd997a2ba: The extended learned-model curve improves mean Recall@10 from 0.770724 at 120,000 training domains to 0.784351 at 400,000 and 0.791325 at 1,200,000: a total gain of 0.020601, or approximately 2.06 percentage points. (Within-exploration three-fold validation of the CUDA character-ngram MLP with 256 hidden units and inverse-cardinality BCE; not target-set performance.)
- finding_a67c09a3b7ff: The extension did not reuse the exact earlier validation subsets. Its re-measured 120,000-domain baseline supports comparisons within this exploration, but the results are not an identical-fold continuation of the previous curve. (Fold reproducibility and comparability across explorations.)
- finding_39ca2725c6ce: Complete reported three-fold experiment costs are 117.90 seconds at 120,000 training domains, 176.02 seconds at 400,000, and 371.91 seconds at 1,200,000, including source construction, fitted features, and fitting/scoring. (Measured costs for this MLP pipeline, not guarantees for other model families.)
- finding_7ee4fcb8a6c3: A 1,206,000-domain working pool—1,200,000 training domains plus three 2,000-domain validation subsets—is a supported planning choice for at least twelve experiments of approximately the measured cost, with substantial allowance for more expensive models. (Experiment planning under the reported runtime assumptions; not the largest feasible pool or a guarantee for arbitrary substantive models.)
- finding_ac16c4f8dd08: Diminishing returns provide a practical reason to stop at 1,200,000 training domains, but the evidence does not establish a statistical plateau. (Interpretation of the measured learning curve and stopping decision.)
- finding_5eb360ae10c4: The recomputed adversarial comparisons again select domains with at least two known trackers. No reported less restrictive rule is within the selected rule's stated fold-noise tolerance. (The 22 current sampled adversarial comparisons; observable similarity does not establish conditional tracker-label equivalence.)
- finding_bc59e05d694f: All 22 current comparison rows are readable, including the six complementary metadata/degree filters. The two zero-degree filters show strong observable separation from targets rather than improved population matching. (Completeness and results of the current comparison artifacts, not recovery of unpublished historical filter identities.)
- finding_3f45fcefa7c4: The reported sampled graph slices contain 56,748,502 outgoing rows and 48,763,103 incoming rows. The recommended slicing policy restricts outgoing sources and incoming destinations to modelled or predicted domains, while category-neighbour features use classification metadata. (Reported graph artifacts and recommended feature-construction policy; row counts do not establish deduplication or exact inclusion of every target.)
- finding_1d6e3956bcc0: The requested model-error analysis remains incomplete because full out-of-fold predictions were unavailable to this exploration. Slice Recall@10, missed-tracker ranks, domain-normalized miss contributions, and representative prediction failures were not computed; no model-error-based experimental priority is supported. (Evidence available to this exploration, not whether the original probe artifact exists.)
- finding_ff8e14259cc6: Validation and target domains have similar coarse hyperlink-coverage proportions, but these counts do not establish performance by coverage or availability of externally labelled neighbours. (Incoming/outgoing hyperlink-degree coverage for 6,000 reserved validation domains and 50,000 targets.)
- finding_edca7cc48d68: Domains with hyperlinks in both directions have higher mean known-tracker cardinality than the other coverage groups in every validation fold. This is a descriptive label-cardinality association, not evidence of where the model misses trackers. (Retrospective cardinality summaries across the three locked validation folds.)
- finding_27a71df1b55c: The ten-guess limit accounts for only about 0.1704 percentage points of mean validation recall loss. Relative to the recorded best candidate's Recall@10, approximately 13.3458 percentage points remain below the cardinality oracle ceiling, so excess cardinality explains only about 1.26% of its total loss from perfect recall. (Aggregate arithmetic on the locked validation domains using supplied known labels; the oracle gap is not a guarantee of practically recoverable performance.)
- finding_30517f4bae6b: Validation positive edges are concentrated among the eight trackers observed on more than 100,000 common-training domains, but frequency exposure alone cannot identify which frequency bands contribute most model error. (Tracker-frequency exposure measured from the 1,200,000 common training domains, excluding all 6,000 reserved domains; edge counts are not domain-normalized loss.)
- finding_b8a824505097: The visible geography table describes target TLD composition, not geographic model errors, and the hostname preview cannot be treated as representative failures. (Displayed geography and validation-metadata previews only; TLD is not verified operating geography.)
- finding_ef0363c61acf: Both bounded two-hop blocks were constructed at substantial scale without materializing an unrestricted two-hop graph. (Retained graph slices for the locked pool and prediction domains in this exploration.)
- finding_ac1707c23b2f: Truncation affects a meaningful minority of domains and intermediaries, and retained neighbours are selected by parquet row order rather than random sampling. (Actual truncation under the 16-first-hop and 64-terminal bounds; these rates describe entities truncated, not fractions of edges discarded.)
- finding_517e0d8bc208: Shared-source profiles cover more domains than shared-destination profiles, while each block has similar coverage across training, validation, and targets. (Nonzero bounded-profile coverage, not learned-model performance or joint coverage of the two blocks.)
- finding_c3a7d0c30796: The bounded profiles are broad distributions with approximately unit mass when covered; target and training summaries are close, including their upper quantiles. (Profile and covered-intermediary distributions across all rows, including uncovered rows with zero mass.)
- finding_7e41d4c57c68: Shared-destination rankings recover substantial true-tracker relevance outside the diagnostic one-hop top ten on covered validation domains. (Standalone shared-destination feature rankings on their covered subsets; the comparator is an equal-weight sum of normalized incoming and outgoing one-hop profiles, not the learned parent.)
- finding_73c2cff6994a: Shared-source rankings also recover true trackers outside diagnostic one-hop guesses, with broader coverage but lower conditional recall than shared destination; the two blocks are evaluated on different covered populations. (Standalone shared-source feature rankings on their covered validation subsets, not an identical-domain comparison between blocks.)
- finding_f594649c80c6: The reported terminal-label exclusion checks support prediction-time availability and prevent a self-return path from directly recovering a locked domain's supplied labels. Frequent near-perfect rankings warrant scrutiny but do not by themselves demonstrate leakage. (Reported label-source audit and ranking flags for these bounded features; not an independent execution-code audit or proof against every possible leakage mechanism.)
- finding_880d1c994aff: Measured construction costs are compatible with further experiments under the stated planning assumptions, but isolated full-build times and memory allocations for each block were not measured. (Exploration timing and cumulative process-memory diagnostics, plus an explicitly conservative three-fold estimate.)
- finding_71d3da9ee8d5: Normalized classification metadata contains 959,198 hosts across 15 categories; 47,585 hosts have multiple category assignments. Multi-category membership is retained with equal fractional weights rather than selecting one category. (Returned classification parsing diagnostics; exact normalization code is not supplied for independent verification.)
- finding_4deac4a0794b: The semantic graph audit uses a substantial focal-domain hyperlink slice and reports category profiles normalized over classified neighbours, with each neighbour's unit mass split equally across its categories. (Reported one-hop semantic-feature construction for locked and target domains; edge counts alone do not independently verify endpoint filtering.)
- finding_5d63b69a3931: Direct classification coverage is sparse, whereas classified hyperlink neighbours provide substantially broader coverage. Training and target coverage proportions are close for each block. (Classification availability for 1,200,000 common training domains, three 2,000-domain validation subsets, and 50,000 targets; joint coverage is not reported.)
- finding_e2a80d0d99b1: Training and target category-profile summaries are descriptively similar, but most domains lack either directional category profile and neighbour counts have substantial upper tails. (Population-wide summaries include uncovered domains with zero category mass; similarity does not establish target-label equivalence.)
- finding_8e81923dc7d9: Standalone category-conditioned rankings recover known trackers on covered validation domains, but these conditional scores do not demonstrate complementary signal or improvement over the learned parent. (Separate covered validation populations for direct, outgoing, and incoming category diagnostics; not full-population or augmented-model Recall@10.)
- finding_85ab739ae421: The reported auxiliary-label audit shows zero locked-pool overlap and states that pool labels are excluded before category-label joins and aggregation, with validation labels used only for diagnostic scoring. (Reported label-source safety checks, not an independent execution-code audit or verification of every weighting operation.)
- finding_f9e99b0ca494: Complete output evaluation for semantic construction and diagnostics took approximately 123 seconds, and the full harness run took approximately 143 seconds. Isolated semantic-feature timing and peak memory remain unmeasured. (This bounded audit, including diagnostic scoring but no learned-model fitting; not a per-fold augmented-pipeline cost.)
- finding_4c390ec78121: Launching the augmented three-fold experiment is not supported by the remaining budget under the parent's measured-cost planning assumptions. (Runtime planning using measured parent costs; not proof that every alternative implementation would exceed the budget.)
- finding_e3206f213180: The exploration stopped after the first baseline fold, so it does not establish whether adding content-category features improves the lightweight baseline. Three-fold means, spreads, paired differences, and augmented-model scores remain unavailable. (Returned learned semantic comparison on the locked evaluation population; not evidence against semantic features.)
- finding_d5074f266685: The measured baseline is a CUDA numeric/TLD MLP trained on the exact common locked training set, with no relational tracker profiles or auxiliary tracker labels. (Reported exploratory model and split protocol; not an independent execution-code audit.)
- finding_09427e9e877f: Construction took approximately 99.97 seconds and first-fold feature fitting, training and scoring took 73.20 seconds. The actual complete harness run took 232.98 seconds; this is the cost of the partial audit, not a completed six-fit comparison. (Measured execution costs for this exploration, distinguished from projected full-comparison costs.)
- finding_e2131a51fff7: Under the stated repeated-construction estimate and 20% safety margin, the full comparison exceeded its 1,200-second allowance, supporting the timing-gate stop. This projection does not measure augmented-model runtime. (Feasibility planning based on construction and the first baseline fold, without assuming shared construction.)
- finding_4239ae483a29: Reported first-baseline-fold memory measures are 1.631744 GiB of prepared arrays and 0.071909 GiB peak CUDA allocation. Process peak RSS and augmented-model memory remain unmeasured. (Partial memory instrumentation; these quantities are not total peak system memory.)
- finding_8a6cc7c46683: Direct category coverage is sparse, while classified hyperlink neighbours provide substantially broader coverage. Training and target coverage rates are descriptively close. (Separate direct, outgoing and incoming category availability; joint coverage and performance by coverage are not reported.)
- finding_276d57a9da66: Complementarity to the selected relational parent remains unresolved: no parent top-ten overlap or domain-normalized recovery outside its guesses was computed, and even a completed lightweight comparison would not directly establish improvement over that parent. (Parent-comparison limitations of this exploration.)

All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.
