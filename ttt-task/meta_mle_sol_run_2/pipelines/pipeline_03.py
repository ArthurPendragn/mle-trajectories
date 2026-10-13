import numpy as np
import pandas as pd
import skrub
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator

class NeighborhoodAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.fallback = nn.Parameter(torch.zeros(355))
        self.temperatures = nn.Parameter(torch.zeros(n_channels))
        self.gate = nn.Sequential(
            nn.Linear(n_channels * 357, 64),
            nn.ReLU(),
            nn.Linear(64, n_channels + 1),
        )

    def forward(self, vectors, metadata):
        fallback = torch.softmax(self.fallback, dim=0)
        smoothed = (vectors + 0.05 * fallback) / 1.05
        temperatures = torch.exp(self.temperatures).clamp(0.25, 4.0)
        channels = torch.softmax(
            torch.log(smoothed.clamp_min(1e-9)) * temperatures[None, :, None],
            dim=-1,
        )
        gate_input = torch.cat(
            [vectors.reshape(len(vectors), -1), metadata.reshape(len(vectors), -1)],
            dim=1,
        )
        logits = self.gate(gate_input)
        available = vectors.sum(dim=-1) > 0
        logits = torch.cat(
            [logits[:, :-1].masked_fill(~available, -1e4), logits[:, -1:]],
            dim=1,
        )
        weights = torch.softmax(logits, dim=1)
        return (
            (channels * weights[:, :-1, None]).sum(dim=1)
            + weights[:, -1:] * fallback
        )

class GraphAttentionRanker(BaseEstimator):
    def __init__(self, direction="both", epochs=24, batch_size=2048, learning_rate=0.002):
        self.direction = direction
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _arrays(self, X):
        directions = ["outgoing", "incoming"] if self.direction == "both" else [self.direction]
        vectors = np.stack(
            [X[[f"{d}_{i}" for i in range(355)]].to_numpy(dtype=np.float32)
             for d in directions],
            axis=1,
        )
        metadata = np.stack(
            [X[[f"{d}_degree", f"{d}_covered"]].to_numpy(dtype=np.float32)
             for d in directions],
            axis=1,
        )
        metadata = np.log1p(metadata)
        return vectors, metadata

    def fit(self, X, y):
        torch.set_num_threads(16)
        torch.manual_seed(42)
        np.random.seed(42)
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        vectors, metadata = self._arrays(X)
        labels = np.asarray(y, dtype=np.float32)
        labels /= labels.sum(axis=1, keepdims=True).clip(min=1)
        self.net_ = NeighborhoodAttention(vectors.shape[1]).to(self.device_)
        with torch.no_grad():
            prior = labels.mean(axis=0).clip(min=1e-7)
            self.net_.fallback.copy_(torch.tensor(np.log(prior), device=self.device_))
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(vectors),
                torch.from_numpy(metadata),
                torch.from_numpy(labels),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        for epoch in range(self.epochs):
            self.net_.train()
            total = 0.0
            for v, m, target in loader:
                v, m, target = v.to(self.device_), m.to(self.device_), target.to(self.device_)
                optimizer.zero_grad()
                probability = self.net_(v, m)
                loss = -(target * probability.clamp_min(1e-9).log()).sum(dim=1).mean()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(v)
            if epoch % 6 == 0:
                print("graph attention epoch", epoch, "loss", total / len(labels))
        return self

    def predict(self, X):
        vectors, metadata = self._arrays(X)
        self.net_.eval()
        result = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                v = torch.from_numpy(vectors[start:start + self.batch_size]).to(self.device_)
                m = torch.from_numpy(metadata[start:start + self.batch_size]).to(self.device_)
                result.append(self.net_(v, m).topk(10, dim=1).indices.cpu().numpy())
        return np.concatenate(result, axis=0)

def neighborhood_block(edges, labels, direction):
    degree = edges.groupby("domain_id").size().rename(f"{direction}_degree").reset_index()
    labelled_edges = edges.merge(labels, on="neighbor_id", how="inner")
    covered = labelled_edges[["domain_id", "neighbor_id"]].drop_duplicates()
    covered = covered.groupby("domain_id").size().rename(f"{direction}_covered").reset_index()
    sums = labelled_edges.groupby(["domain_id", "tracker_id"])["weight"].sum().reset_index()
    wide = sums.pivot_table(
        index="domain_id", columns="tracker_id", values="weight", aggfunc="sum", fill_value=0
    )
    wide = wide.reindex(columns=list(range(355)), fill_value=0)
    totals = covered.set_index("domain_id")[f"{direction}_covered"]
    wide = wide.div(totals, axis=0).fillna(0)
    wide = wide.rename(columns={i: f"{direction}_{i}" for i in range(355)}).reset_index()
    block = degree.merge(covered, on="domain_id", how="left")
    block = block.merge(wide, on="domain_id", how="left").fillna(0)
    return block.skb.set_name(f"{direction}_neighbor_evidence")

def build():
    setup = build_evaluation()
    X, y = setup["X"], setup["y"]

    tracking = skrub.as_data_op(ROOT + "tracking_graph_train.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"]
    )
    tracking = tracking.drop_duplicates(["domain_id", "tracker_id"])
    counts = tracking.groupby("domain_id").size().rename("tracker_count").reset_index()
    eligible = counts[counts["tracker_count"] >= 2][["domain_id"]]
    entire_locked_pool = eligible.sample(n=POOL_SIZE, random_state=42).reset_index(drop=True)
    targets = skrub.as_data_op(ROOT + "target.tsv").skb.apply_func(pd.read_csv, sep="\t")
    relevant = entire_locked_pool.skb.concat([targets[["domain_id"]]], axis=0)
    relevant = relevant.drop_duplicates("domain_id")

    links = skrub.as_data_op(ROOT + "link-graph.parquet").skb.apply_func(
        pd.read_parquet, columns=["source_domain_id", "target_domain_id"]
    )
    incident = links[
        links["source_domain_id"].isin(relevant["domain_id"])
        | links["target_domain_id"].isin(relevant["domain_id"])
    ]
    incident = incident[
        incident["source_domain_id"] != incident["target_domain_id"]
    ].drop_duplicates()
    outgoing = incident[incident["source_domain_id"].isin(relevant["domain_id"])]
    outgoing = outgoing.rename(
        columns={"source_domain_id": "domain_id", "target_domain_id": "neighbor_id"}
    )
    incoming = incident[incident["target_domain_id"].isin(relevant["domain_id"])]
    incoming = incoming.rename(
        columns={"target_domain_id": "domain_id", "source_domain_id": "neighbor_id"}
    )
    neighbors = outgoing[["neighbor_id"]].skb.concat([incoming[["neighbor_id"]]], axis=0)
    neighbors = neighbors.drop_duplicates()
    safe_labels = tracking[
        ~tracking["domain_id"].isin(entire_locked_pool["domain_id"])
        & tracking["domain_id"].isin(neighbors["neighbor_id"])
    ]
    safe_labels = safe_labels.merge(counts, on="domain_id", how="left")
    safe_labels = safe_labels.assign(weight=1.0 / safe_labels["tracker_count"])
    safe_labels = safe_labels[["domain_id", "tracker_id", "weight"]].rename(
        columns={"domain_id": "neighbor_id"}
    ).skb.set_name("disjoint_neighbor_labels")

    out_features = neighborhood_block(outgoing, safe_labels, "outgoing")
    in_features = neighborhood_block(incoming, safe_labels, "incoming")
    features = X[["domain_id"]].merge(out_features, on="domain_id", how="left")
    features = features.merge(in_features, on="domain_id", how="left").fillna(0)
    features = features.drop(columns=["domain_id"]).skb.set_name("neighborhood_channels")
    model = skrub.choose_from(
        {
            "outgoing": GraphAttentionRanker(direction="outgoing"),
            "incoming": GraphAttentionRanker(direction="incoming"),
            "both": GraphAttentionRanker(direction="both"),
        },
        name="neighborhood_direction",
    )
    pred = features.skb.apply(model, y=y).skb.set_name("model")
    return {"pred": pred, "scoring": setup["scoring"], "row_keys": setup["row_keys"]}