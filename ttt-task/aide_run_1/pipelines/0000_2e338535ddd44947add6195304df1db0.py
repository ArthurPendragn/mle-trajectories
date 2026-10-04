import os
import re
import sys
import numpy as np
import pandas as pd
import polars as pl

# ---------------------------------------------------------
# Configuration & Seed
# ---------------------------------------------------------
SEED = 42
np.random.seed(SEED)
NUM_TRACKERS = 355
VAL_SIZE = 20000

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

print("--- Step 1: Loading tracker metadata and targets ---")
# Load trackers metadata
trackers_df = pd.read_csv(os.path.join(INPUT_DIR, "trackers.tsv"), sep="\t")
print(f"Loaded {len(trackers_df)} trackers.")

tracker_id_to_tdid = np.zeros(NUM_TRACKERS, dtype=np.int64)
tdid_to_tracker_id = {}
for _, row in trackers_df.iterrows():
    t_id = int(row["tracker_id"])
    td_id = int(row["tracking_domain_id"])
    tracker_id_to_tdid[t_id] = td_id
    tdid_to_tracker_id[td_id] = t_id

tracker_domain_ids = set(trackers_df["tracking_domain_id"].astype(int).tolist())

# Load target domains
target_df = pd.read_csv(os.path.join(INPUT_DIR, "target.tsv"), sep="\t")
target_domain_ids = target_df["domain_id"].astype(np.int64).values
print(f"Loaded {len(target_domain_ids)} target domains.")

print("--- Step 2: Loading training tracking graph and splitting validation ---")
train_tracking_pl = (
    pl.read_parquet(os.path.join(INPUT_DIR, "tracking_graph_train.parquet"))
    .select(["domain_id", "tracker_id"])
    .unique()
)

all_train_domains = (
    train_tracking_pl.select("domain_id").unique()["domain_id"].to_numpy()
)
print(f"Total unique domains in training tracking graph: {len(all_train_domains):,}")

# Hold-out validation split
np.random.seed(SEED)
val_domain_indices = np.random.choice(
    len(all_train_domains), size=VAL_SIZE, replace=False
)
val_domain_ids = all_train_domains[val_domain_indices]
val_domain_set = set(val_domain_ids)

train_domain_mask = ~np.isin(all_train_domains, val_domain_ids)
effective_train_domains = all_train_domains[train_domain_mask]
train_domain_set = set(effective_train_domains)
print(
    f"Train domain split: {len(effective_train_domains):,}, Validation domain split: {len(val_domain_ids):,}"
)

# Split tracking graph into train and val (truth)
train_edges = train_tracking_pl.filter(pl.col("domain_id").is_in(train_domain_set))
val_edges = train_tracking_pl.filter(pl.col("domain_id").is_in(val_domain_set))

# Build ground truth dictionary for validation
val_truth_df = val_edges.to_pandas()
val_truth_map = (
    val_truth_df.groupby("domain_id")["tracker_id"]
    .apply(lambda x: set(x.tolist()))
    .to_dict()
)

# Compute global tracker prior
global_counts = np.zeros(NUM_TRACKERS, dtype=np.float32)
train_counts_pl = train_edges.group_by("tracker_id").len().to_pandas()
for _, r in train_counts_pl.iterrows():
    t_id = int(r["tracker_id"])
    if 0 <= t_id < NUM_TRACKERS:
        global_counts[t_id] = r["len"]

global_prior = (global_counts + 1.0) / (len(effective_train_domains) + NUM_TRACKERS)
print("Global tracker prior computed.")

print("--- Step 3: Processing domains metadata, TLDs, and categories ---")
domains_pl = pl.read_parquet(os.path.join(INPUT_DIR, "domains.parquet"))


# Parse TLD and keywords
def parse_domain_info(domain_str):
    if not isinstance(domain_str, str):
        return "unknown"
    parts = domain_str.lower().split(".")
    if len(parts) >= 3 and parts[-1] in {
        "uk",
        "au",
        "jp",
        "br",
        "nz",
        "za",
        "in",
        "mx",
        "ar",
        "ru",
        "tr",
        "cn",
    }:
        if parts[-2] in {
            "co",
            "com",
            "org",
            "edu",
            "gov",
            "net",
            "ac",
            "ne",
            "or",
            "go",
        }:
            return f"{parts[-2]}.{parts[-1]}"
    return parts[-1] if len(parts) > 1 else "unknown"


domains_df = domains_pl.to_pandas()
domains_df["tld"] = domains_df["domain"].apply(parse_domain_info)
domains_df["clean_domain"] = (
    domains_df["domain"].astype(str).str.lower().str.replace("^www\\.", "", regex=True)
)

domain_to_tld = dict(zip(domains_df["domain_id"], domains_df["tld"]))

# URL Classification mapping
if os.path.exists(os.path.join(INPUT_DIR, "url-classification.csv")):
    print("Loading URL classification...")
    url_df = pd.read_csv(os.path.join(INPUT_DIR, "url-classification.csv"))

    def extract_clean_host(u):
        if not isinstance(u, str):
            return ""
        if "://" in u:
            u = u.split("://", 1)[1]
        u = u.split("/")[0].split(":")[0].split("?")[0].lower()
        if u.startswith("www."):
            u = u[4:]
        return u

    url_df["clean_domain"] = url_df["url"].apply(extract_clean_host)
    url_df = url_df[url_df["clean_domain"] != ""].drop_duplicates(
        subset=["clean_domain"]
    )

    merged_cat = domains_df[["domain_id", "clean_domain"]].merge(
        url_df[["clean_domain", "category"]], on="clean_domain", how="inner"
    )
    domain_to_cat = dict(zip(merged_cat["domain_id"], merged_cat["category"]))
    print(f"Assigned categories to {len(domain_to_cat):,} domains.")
else:
    domain_to_cat = {}

# Compute TLD prior
print("Computing TLD empirical priors...")
train_tld_df = train_edges.to_pandas()
train_tld_df["tld"] = train_tld_df["domain_id"].map(domain_to_tld).fillna("unknown")
tld_group_counts = (
    train_tld_df.groupby(["tld", "tracker_id"]).size().unstack(fill_value=0)
)
tld_total_domains = (
    domains_df[domains_df["domain_id"].isin(train_domain_set)]
    .groupby("tld")["domain_id"]
    .count()
)

tld_priors = {}
ALPHA_TLD = 10.0
for tld_val, row in tld_group_counts.iterrows():
    arr = np.zeros(NUM_TRACKERS, dtype=np.float32)
    for col_idx in row.index:
        if 0 <= col_idx < NUM_TRACKERS:
            arr[col_idx] = row[col_idx]
    denom = tld_total_domains.get(tld_val, arr.sum()) + ALPHA_TLD
    tld_priors[tld_val] = (arr + ALPHA_TLD * global_prior) / denom

# Compute Category prior
print("Computing Category empirical priors...")
train_cat_df = train_edges.to_pandas()
train_cat_df["category"] = train_cat_df["domain_id"].map(domain_to_cat)
train_cat_df = train_cat_df.dropna(subset=["category"])
cat_group_counts = (
    train_cat_df.groupby(["category", "tracker_id"]).size().unstack(fill_value=0)
)
cat_total_domains = pd.Series(
    {d: domain_to_cat[d] for d in train_domain_set if d in domain_to_cat}
).value_counts()

cat_priors = {}
ALPHA_CAT = 10.0
for cat_val, row in cat_group_counts.iterrows():
    arr = np.zeros(NUM_TRACKERS, dtype=np.float32)
    for col_idx in row.index:
        if 0 <= col_idx < NUM_TRACKERS:
            arr[col_idx] = row[col_idx]
    denom = cat_total_domains.get(cat_val, arr.sum()) + ALPHA_CAT
    cat_priors[cat_val] = (arr + ALPHA_CAT * global_prior) / denom

print("--- Step 4: Processing link graph for target and validation domains ---")
eval_domains_set = set(val_domain_ids).union(set(target_domain_ids))

link_graph_pl = pl.read_parquet(os.path.join(INPUT_DIR, "link-graph.parquet"))

# Direct tracker links
print("Finding direct links to tracker domains...")
direct_edges_pl = (
    link_graph_pl.filter(
        pl.col("source_domain_id").is_in(eval_domains_set)
        & pl.col("target_domain_id").is_in(tracker_domain_ids)
    )
    .select(["source_domain_id", "target_domain_id"])
    .to_pandas()
)

direct_tracker_map = {}
for _, r in direct_edges_pl.iterrows():
    src = r["source_domain_id"]
    td = r["target_domain_id"]
    if td in tdid_to_tracker_id:
        t_id = tdid_to_tracker_id[td]
        if src not in direct_tracker_map:
            direct_tracker_map[src] = set()
        direct_tracker_map[src].add(t_id)
print(f"Domains with direct tracker links: {len(direct_tracker_map):,}")

# Out-neighbor links to train domains
print("Aggregating out-neighbor tracker usage...")
out_edges_pl = link_graph_pl.filter(
    pl.col("source_domain_id").is_in(eval_domains_set)
    & pl.col("target_domain_id").is_in(train_domain_set)
)

out_degrees_pl = (
    out_edges_pl.group_by("source_domain_id")
    .agg(pl.col("target_domain_id").n_unique().alias("deg"))
    .to_pandas()
)
out_deg_map = dict(zip(out_degrees_pl["source_domain_id"], out_degrees_pl["deg"]))

out_tracker_edges = (
    out_edges_pl.join(train_edges, left_on="target_domain_id", right_on="domain_id")
    .group_by(["source_domain_id", "tracker_id"])
    .len()
    .to_pandas()
)

out_tracker_counts = {}
for _, r in out_tracker_edges.iterrows():
    src = r["source_domain_id"]
    t_id = int(r["tracker_id"])
    c = r["len"]
    if src not in out_tracker_counts:
        out_tracker_counts[src] = {}
    out_tracker_counts[src][t_id] = c

# In-neighbor links from train domains
print("Aggregating in-neighbor tracker usage...")
in_edges_pl = link_graph_pl.filter(
    pl.col("target_domain_id").is_in(eval_domains_set)
    & pl.col("source_domain_id").is_in(train_domain_set)
)

in_degrees_pl = (
    in_edges_pl.group_by("target_domain_id")
    .agg(pl.col("source_domain_id").n_unique().alias("deg"))
    .to_pandas()
)
in_deg_map = dict(zip(in_degrees_pl["target_domain_id"], in_degrees_pl["deg"]))

in_tracker_edges = (
    in_edges_pl.join(train_edges, left_on="source_domain_id", right_on="domain_id")
    .group_by(["target_domain_id", "tracker_id"])
    .len()
    .to_pandas()
)

in_tracker_counts = {}
for _, r in in_tracker_edges.iterrows():
    tgt = r["target_domain_id"]
    t_id = int(r["tracker_id"])
    c = r["len"]
    if tgt not in in_tracker_counts:
        in_tracker_counts[tgt] = {}
    in_tracker_counts[tgt][t_id] = c

del link_graph_pl
print("Graph neighbor aggregation complete.")

print("--- Step 5: Constructing dense feature matrices ---")


def build_feature_matrices(domain_id_list):
    n = len(domain_id_list)
    V_direct = np.zeros((n, NUM_TRACKERS), dtype=np.float32)
    V_out = np.zeros((n, NUM_TRACKERS), dtype=np.float32)
    V_in = np.zeros((n, NUM_TRACKERS), dtype=np.float32)
    V_cat = np.zeros((n, NUM_TRACKERS), dtype=np.float32)
    V_tld = np.zeros((n, NUM_TRACKERS), dtype=np.float32)
    V_glob = np.tile(global_prior, (n, 1)).astype(np.float32)

    BETA_OUT = 3.0
    BETA_IN = 5.0

    for i, d in enumerate(domain_id_list):
        # Direct
        if d in direct_tracker_map:
            for t_id in direct_tracker_map[d]:
                V_direct[i, t_id] = 1.0

        # Out neighbors
        if d in out_tracker_counts:
            denom = out_deg_map.get(d, 1.0) + BETA_OUT
            for t_id, cnt in out_tracker_counts[d].items():
                if 0 <= t_id < NUM_TRACKERS:
                    V_out[i, t_id] = cnt / denom

        # In neighbors
        if d in in_tracker_counts:
            denom = in_deg_map.get(d, 1.0) + BETA_IN
            for t_id, cnt in in_tracker_counts[d].items():
                if 0 <= t_id < NUM_TRACKERS:
                    V_in[i, t_id] = cnt / denom

        # TLD
        tld = domain_to_tld.get(d, "unknown")
        if tld in tld_priors:
            V_tld[i] = tld_priors[tld]
        else:
            V_tld[i] = global_prior

        # Category
        cat = domain_to_cat.get(d, None)
        if cat is not None and cat in cat_priors:
            V_cat[i] = cat_priors[cat]
        else:
            V_cat[i] = V_tld[i]

    return V_direct, V_out, V_in, V_cat, V_tld, V_glob


V_dir_val, V_out_val, V_in_val, V_cat_val, V_tld_val, V_glob_val = (
    build_feature_matrices(val_domain_ids)
)

print("--- Step 6: Optimizing feature weights on validation set for Recall@10 ---")


def compute_val_recall(scores, val_ids, truth_map):
    top10 = np.argpartition(-scores, 10, axis=1)[:, :10]
    recalls = []
    for i, d in enumerate(val_ids):
        true_set = truth_map.get(d, set())
        if len(true_set) == 0:
            continue
        pred_set = top10[i]
        hits = len(true_set.intersection(pred_set))
        recalls.append(hits / len(true_set))
    return float(np.mean(recalls))


# Initial weights
weights = np.array([10.0, 5.0, 2.5, 1.0, 1.5, 0.5], dtype=np.float32)


def evaluate_weights(w):
    S = (
        w[0] * V_dir_val
        + w[1] * V_out_val
        + w[2] * V_in_val
        + w[3] * V_cat_val
        + w[4] * V_tld_val
        + w[5] * V_glob_val
    )
    return compute_val_recall(S, val_domain_ids, val_truth_map)


current_recall = evaluate_weights(weights)
print(f"Initial Validation Recall@10: {current_recall:.5f}")

# Coordinate line search for direct metric optimization
for iteration in range(2):
    for param_idx in range(len(weights)):
        best_val = weights[param_idx]
        best_score = current_recall
        step = 0.5 if param_idx > 0 else 1.0
        candidates = [
            max(0.01, best_val + delta * step) for delta in [-3, -2, -1, 1, 2, 3]
        ]
        for cand in candidates:
            test_w = weights.copy()
            test_w[param_idx] = cand
            score = evaluate_weights(test_w)
            if score > best_score:
                best_score = score
                best_val = cand
        weights[param_idx] = best_val
        current_recall = best_score

print(f"Optimized Weights: {weights.round(3)}")
print(f"Validation Recall@10: {current_recall:.5f}")

print("--- Step 7: Generating predictions for target domains ---")
V_dir_tgt, V_out_tgt, V_in_tgt, V_cat_tgt, V_tld_tgt, V_glob_tgt = (
    build_feature_matrices(target_domain_ids)
)

target_scores = (
    weights[0] * V_dir_tgt
    + weights[1] * V_out_tgt
    + weights[2] * V_in_tgt
    + weights[3] * V_cat_tgt
    + weights[4] * V_tld_tgt
    + weights[5] * V_glob_tgt
)

num_targets = len(target_domain_ids)
top10_part = np.argpartition(-target_scores, 10, axis=1)[:, :10]
row_indices = np.arange(num_targets)[:, None]
top10_sorted_within = np.argsort(-target_scores[row_indices, top10_part], axis=1)
top10_tracker_ids = top10_part[row_indices, top10_sorted_within]

# Map tracker_id (0-354) to tracking_domain_id
predicted_tdids = tracker_id_to_tdid[top10_tracker_ids]

# Flatten to submission dataframe
flat_domain_ids = np.repeat(target_domain_ids, 10)
flat_tracking_domain_ids = predicted_tdids.flatten()

sub_df = pd.DataFrame(
    {"domain_id": flat_domain_ids, "tracking_domain_id": flat_tracking_domain_ids}
)

print(f"Submission shape: {sub_df.shape}")
assert sub_df["domain_id"].nunique() == len(
    target_domain_ids
), "Mismatch in domain count!"
assert len(sub_df) == len(target_domain_ids) * 10, "Mismatch in prediction row count!"

# Save both submission.csv and submission.tsv with tab separator
sub_csv_path = os.path.join(WORKING_DIR, "submission.csv")
sub_tsv_path = os.path.join(WORKING_DIR, "submission.tsv")

sub_df.to_csv(sub_csv_path, sep="\t", index=False)
sub_df.to_csv(sub_tsv_path, sep="\t", index=False)

print(f"Successfully saved test predictions to {sub_csv_path} and {sub_tsv_path}")
print(f"Validation Recall@10: {current_recall:.5f}")
