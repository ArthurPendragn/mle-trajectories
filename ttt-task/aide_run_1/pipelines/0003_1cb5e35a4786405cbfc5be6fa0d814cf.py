import gc
import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

print("Starting TrackTheTrackers model pipeline...")

# 1. Load trackers metadata
print("Loading tracker metadata...")
trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
tracker_id_to_tdid = dict(
    zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
)
tdid_to_tracker_id = dict(
    zip(trackers_df["tracking_domain_id"], trackers_df["tracker_id"])
)
tracker_domain_ids = set(trackers_df["tracking_domain_id"])
num_trackers = len(trackers_df)
print(f"Loaded {num_trackers} candidate trackers.")

# 2. Load target domains
print("Loading target domains...")
target_df = pd.read_csv("input/target.tsv", sep="\t")
target_domains = target_df["domain_id"].drop_duplicates().values
print(f"Loaded {len(target_domains)} target domains.")

# 3. Load tracking training graph
print("Loading tracking training graph...")
train_df = pd.read_parquet(
    "input/tracking_graph_train.parquet", columns=["domain_id", "tracker_id"]
)
train_df = train_df.drop_duplicates()
print(f"Loaded {len(train_df)} training domain-tracker observations.")

# Create 10,000 hold-out validation set
unique_train_domains = train_df["domain_id"].unique()
num_val = 10000
shuffled_domains = np.random.permutation(unique_train_domains)
val_domains = shuffled_domains[:num_val]
train_split_domains = shuffled_domains[num_val:]

val_domains_set = set(val_domains)
train_split_set = set(train_split_domains)

# Build validation ground truth mapping
val_records = train_df[train_df["domain_id"].isin(val_domains_set)]
val_truth = val_records.groupby("domain_id")["tracker_id"].apply(set).to_dict()

# Training split observations (excluding validation domains)
train_split_df = train_df[~train_df["domain_id"].isin(val_domains_set)]
print(
    f"Validation set: {len(val_domains)} domains. Train split: {len(train_split_domains)} domains."
)

# 4. Load hyperlink graph and calculate domain degrees
print("Loading link graph...")
link_df = pd.read_parquet(
    "input/link-graph.parquet", columns=["source_domain_id", "target_domain_id"]
)
print(f"Loaded {len(link_df)} edges from link graph.")

# Compute global in-degree and out-degree for Adamic-Adar weighting
print("Computing domain degrees for normalization...")
in_deg = link_df["target_domain_id"].value_counts()
out_deg = link_df["source_domain_id"].value_counts()

# Filter links touching validation or target domains to optimize memory and speed
all_query_domains = np.unique(np.concatenate([val_domains, target_domains]))
all_query_set = set(all_query_domains)

print("Filtering link graph to active query domain neighborhoods...")
mask = link_df["source_domain_id"].isin(all_query_set) | link_df[
    "target_domain_id"
].isin(all_query_set)
filtered_links = link_df[mask].copy()

# Remove self-loops
filtered_links = filtered_links[
    filtered_links["source_domain_id"] != filtered_links["target_domain_id"]
]
print(f"Retained {len(filtered_links)} neighborhood edges. Freeing raw graph...")
del link_df
gc.collect()


# 5. Core scoring function using direct links, 1-hop bidirectional neighbors, and popularity priors
def score_and_predict(query_domains, train_tracker_pairs, train_domain_subset):
    """Computes tracker prediction scores for query domains and returns top 10 tracker IDs."""
    query_set = set(query_domains)
    q_to_idx = {d: i for i, d in enumerate(query_domains)}
    n_queries = len(query_domains)

    # Global tracker prior distribution from training knowledge
    tracker_freq = train_tracker_pairs["tracker_id"].value_counts(normalize=True)
    prior_scores = np.zeros(num_trackers, dtype=np.float32)
    for tid, freq in tracker_freq.items():
        prior_scores[tid] = freq * 0.05

    # Initialize scores matrix with prior
    scores = np.tile(prior_scores, (n_queries, 1))

    # A. Direct Outbound Links to Tracker Domains
    w_direct = 5.0
    direct_edges = filtered_links[
        filtered_links["source_domain_id"].isin(query_set)
        & filtered_links["target_domain_id"].isin(tracker_domain_ids)
    ].drop_duplicates(subset=["source_domain_id", "target_domain_id"])

    if len(direct_edges) > 0:
        d_q_idx = [q_to_idx[d] for d in direct_edges["source_domain_id"]]
        d_t_idx = direct_edges["target_domain_id"].map(tdid_to_tracker_id).values
        np.add.at(
            scores,
            (d_q_idx, d_t_idx),
            np.full(len(direct_edges), w_direct, dtype=np.float32),
        )

    # B. Outbound 1-Hop Neighbor Propagation (query -> neighbor)
    w_out = 1.0
    out_edges = filtered_links[
        filtered_links["source_domain_id"].isin(query_set)
        & filtered_links["target_domain_id"].isin(train_domain_subset)
    ].drop_duplicates(subset=["source_domain_id", "target_domain_id"])

    if len(out_edges) > 0:
        deg_vals = out_edges["target_domain_id"].map(in_deg).fillna(0).values
        out_edges = out_edges.assign(
            weight=1.0 / np.log(2.0 + deg_vals).astype(np.float32)
        )
        out_merged = out_edges[
            ["source_domain_id", "target_domain_id", "weight"]
        ].merge(
            train_tracker_pairs,
            left_on="target_domain_id",
            right_on="domain_id",
            how="inner",
        )
        if len(out_merged) > 0:
            out_agg = (
                out_merged.groupby(["source_domain_id", "tracker_id"])["weight"]
                .sum()
                .reset_index()
            )
            o_q_idx = [q_to_idx[d] for d in out_agg["source_domain_id"]]
            o_t_idx = out_agg["tracker_id"].values
            np.add.at(
                scores,
                (o_q_idx, o_t_idx),
                (out_agg["weight"].values * w_out).astype(np.float32),
            )

    # C. Inbound 1-Hop Neighbor Propagation (neighbor -> query)
    w_in = 0.5
    in_edges = filtered_links[
        filtered_links["target_domain_id"].isin(query_set)
        & filtered_links["source_domain_id"].isin(train_domain_subset)
    ].drop_duplicates(subset=["source_domain_id", "target_domain_id"])

    if len(in_edges) > 0:
        deg_vals = in_edges["source_domain_id"].map(out_deg).fillna(0).values
        in_edges = in_edges.assign(
            weight=1.0 / np.log(2.0 + deg_vals).astype(np.float32)
        )
        in_merged = in_edges[["source_domain_id", "target_domain_id", "weight"]].merge(
            train_tracker_pairs,
            left_on="source_domain_id",
            right_on="domain_id",
            how="inner",
        )
        if len(in_merged) > 0:
            in_agg = (
                in_merged.groupby(["target_domain_id", "tracker_id"])["weight"]
                .sum()
                .reset_index()
            )
            i_q_idx = [q_to_idx[d] for d in in_agg["target_domain_id"]]
            i_t_idx = in_agg["tracker_id"].values
            np.add.at(
                scores,
                (i_q_idx, i_t_idx),
                (in_agg["weight"].values * w_in).astype(np.float32),
            )

    # Extract Top 10 Predictions per Domain
    top10_part = np.argpartition(-scores, kth=10, axis=1)[:, :10]
    top10_scores = np.take_along_axis(scores, top10_part, axis=1)
    sort_order = np.argsort(-top10_scores, axis=1)
    top10_ranked = np.take_along_axis(top10_part, sort_order, axis=1)

    return top10_ranked


# 6. Evaluate on Validation Set
print("Running validation inference on 10,000 held-out domains...")
val_preds = score_and_predict(val_domains, train_split_df, train_split_set)

# Calculate Recall@10
recalls = []
for i, d in enumerate(val_domains):
    true_set = val_truth[d]
    pred_set = set(val_preds[i])
    hits = len(pred_set.intersection(true_set))
    recalls.append(hits / len(true_set))

val_recall_at_10 = float(np.mean(recalls))
print(f"Validation Recall@10: {val_recall_at_10:.4f}")

# 7. Generate Predictions for Target Domains Using Full Training Data
print("Generating predictions for target domains using full tracking graph data...")
all_train_domains_set = set(train_df["domain_id"].unique())
test_preds = score_and_predict(target_domains, train_df, all_train_domains_set)

# Format predictions into required TSV submission
print("Formatting submission rows...")
sub_domain_ids = np.repeat(target_domains, 10)
sub_tracking_domain_ids = [tracker_id_to_tdid[tid] for row in test_preds for tid in row]

sub_df = pd.DataFrame(
    {
        "domain_id": sub_domain_ids.astype(np.int64),
        "tracking_domain_id": np.array(sub_tracking_domain_ids, dtype=np.int64),
    }
)

os.makedirs("working", exist_ok=True)
sub_csv_path = "working/submission.csv"
sub_tsv_path = "working/submission.tsv"

sub_df.to_csv(sub_csv_path, sep="\t", index=False)
sub_df.to_csv(sub_tsv_path, sep="\t", index=False)

# Validation of submission output
assert os.path.exists(sub_csv_path), "submission.csv was not created!"
assert (
    len(sub_df) == len(target_domains) * 10
), f"Expected {len(target_domains) * 10} rows, found {len(sub_df)}"
assert list(sub_df.columns) == [
    "domain_id",
    "tracking_domain_id",
], "Incorrect column names!"

print(f"Submission saved successfully to {sub_csv_path} with {len(sub_df)} rows.")
print("Preview of submission:")
print(sub_df.head(10).to_string(index=False))
