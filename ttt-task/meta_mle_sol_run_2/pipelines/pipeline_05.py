import numpy as np
import pandas as pd
import skrub
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer


class NeighborhoodAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.fallback = nn.Parameter(torch.zeros(355))
        self.temperatures = nn.Parameter(torch.zeros(n_channels))
        self.gate = nn.Sequential(
            nn.Linear(n_channels * 357, 64), nn.ReLU(),
            nn.Linear(64, n_channels + 1)
        )

    def forward(self, vectors, metadata):
        fallback = torch.softmax(self.fallback, dim=0)
        smoothed = (vectors + 0.05 * fallback) / 1.05
        temperatures = torch.exp(self.temperatures).clamp(0.25, 4.0)
        channels = torch.softmax(
            torch.log(smoothed.clamp_min(1e-9)) * temperatures[None, :, None],
            dim=-1
        )
        gate_input = torch.cat([
            vectors.reshape(len(vectors), -1),
            metadata.reshape(len(vectors), -1)
        ], dim=1)
        logits = self.gate(gate_input)
        available = vectors.sum(dim=-1) > 0
        logits = torch.cat([
            logits[:, :-1].masked_fill(~available, -10000.0),
            logits[:, -1:]
        ], dim=1)
        weights = torch.softmax(logits, dim=1)
        return (
            (channels * weights[:, :-1, None]).sum(dim=1)
            + weights[:, -1:] * fallback
        )


class JointMixture(nn.Module):
    def __init__(self, n_features, residual_width):
        super().__init__()
        self.graph = NeighborhoodAttention(2)
        self.hostname = nn.Linear(n_features, 355)
        self.global_logit = nn.Parameter(torch.tensor(-1.0))
        self.mixture_gate = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.mixture_gate[-1].weight)
        nn.init.constant_(self.mixture_gate[-1].bias, -1.0)
        self.hostname_projection = nn.Linear(n_features, 64)
        self.residual_head = nn.Sequential(
            nn.Linear(710 + 4 + 64, residual_width),
            nn.ReLU(),
            nn.Linear(residual_width, 355)
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, vectors, metadata, hostname):
        graph = self.graph(vectors, metadata)
        host = torch.softmax(self.hostname(hostname), dim=1)
        entropy_graph = -(graph * graph.clamp_min(1e-9).log()).sum(
            dim=1, keepdim=True
        )
        entropy_host = -(host * host.clamp_min(1e-9).log()).sum(
            dim=1, keepdim=True
        )
        gate_input = torch.cat([
            metadata.reshape(len(graph), -1), entropy_graph, entropy_host
        ], dim=1)
        weight = torch.sigmoid(self.mixture_gate(gate_input))
        parent_probability = (1.0 - weight) * graph + weight * host
        joint_input = torch.cat([
            vectors.reshape(len(graph), -1),
            metadata.reshape(len(graph), -1),
            self.hostname_projection(hostname)
        ], dim=1)
        residual = self.residual_head(joint_input)
        probability = torch.softmax(
            parent_probability.clamp_min(1e-9).log() + residual, dim=1
        )
        return probability, parent_probability, weight, residual


class GraphHostnameResidual(BaseEstimator):
    def __init__(self, residual_width=128, epochs=24, batch_size=512):
        self.residual_width = residual_width
        self.epochs = epochs
        self.batch_size = batch_size

    def _arrays(self, X):
        frame = X['graph']
        vectors = np.stack([
            frame[[f'{d}_{i}' for i in range(355)]].to_numpy(dtype=np.float32)
            for d in ['outgoing', 'incoming']
        ], axis=1)
        metadata = np.stack([
            frame[[f'{d}_degree', f'{d}_covered']].to_numpy(dtype=np.float32)
            for d in ['outgoing', 'incoming']
        ], axis=1)
        return vectors, np.log1p(metadata), X['hostname'].tocsr().astype(np.float32)

    def fit(self, X, y):
        torch.set_num_threads(16)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        vectors, metadata, hosts = self._arrays(X)
        labels = np.asarray(y, dtype=np.float32)
        labels = labels / labels.sum(axis=1, keepdims=True).clip(min=1)
        self.net_ = JointMixture(hosts.shape[1], self.residual_width).to(self.device_)
        with torch.no_grad():
            prior = labels.mean(axis=0).clip(min=1e-7)
            self.net_.graph.fallback.copy_(
                torch.tensor(np.log(prior), device=self.device_)
            )
        graph_optimizer = torch.optim.AdamW(
            self.net_.graph.parameters(), lr=0.002, weight_decay=0.0001
        )
        host_optimizer = torch.optim.Adam(
            self.net_.hostname.parameters(), lr=0.001, weight_decay=1e-6
        )
        gate_optimizer = torch.optim.Adam(
            self.net_.mixture_gate.parameters(), lr=0.002
        )
        residual_optimizer = torch.optim.AdamW(
            list(self.net_.hostname_projection.parameters())
            + list(self.net_.residual_head.parameters()),
            lr=0.001
        )
        optimizers = [
            graph_optimizer, host_optimizer, gate_optimizer, residual_optimizer
        ]
        loader = DataLoader(
            TensorDataset(torch.arange(len(labels))),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )
        for epoch in range(self.epochs):
            self.net_.train()
            total = 0.0
            for indices, in loader:
                rows = indices.numpy()
                v = torch.from_numpy(vectors[rows]).to(self.device_)
                m = torch.from_numpy(metadata[rows]).to(self.device_)
                h = torch.from_numpy(hosts[rows].toarray()).to(self.device_)
                target = torch.from_numpy(labels[rows]).to(self.device_)
                for optimizer in optimizers:
                    optimizer.zero_grad()
                probability, parent, weight, residual = self.net_(v, m, h)
                loss = -(target * probability.clamp_min(1e-9).log()).sum(
                    dim=1
                ).mean()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                total += float(loss.detach()) * len(rows)
            if epoch % 6 == 0:
                print('joint residual epoch', epoch, 'loss', total / len(labels))
        return self

    def predict(self, X):
        vectors, metadata, hosts = self._arrays(X)
        self.net_.eval()
        result, weights, magnitudes, shifts = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(vectors), self.batch_size):
                stop = start + self.batch_size
                v = torch.from_numpy(vectors[start:stop]).to(self.device_)
                m = torch.from_numpy(metadata[start:stop]).to(self.device_)
                h = torch.from_numpy(hosts[start:stop].toarray()).to(self.device_)
                probability, parent, weight, residual = self.net_(v, m, h)
                result.append(probability.topk(10, dim=1).indices.cpu().numpy())
                weights.append(weight.cpu().numpy())
                centered = residual - residual.mean(dim=1, keepdim=True)
                magnitudes.append(centered.square().mean(dim=1).sqrt().cpu().numpy())
                shifts.append((probability - parent).abs().sum(dim=1).cpu().numpy())
        print('hostname mixture-weight quantiles',
              np.quantile(np.concatenate(weights), [0, 0.1, 0.5, 0.9, 1]))
        print('centered residual RMS quantiles',
              np.quantile(np.concatenate(magnitudes), [0, 0.1, 0.5, 0.9, 1]))
        print('probability L1 correction quantiles',
              np.quantile(np.concatenate(shifts), [0, 0.1, 0.5, 0.9, 1]))
        return np.concatenate(result, axis=0)


def neighborhood_block(edges, labels, direction):
    degree = edges.groupby('domain_id').size().rename(
        f'{direction}_degree'
    ).reset_index()
    labelled_edges = edges.merge(labels, on='neighbor_id', how='inner')
    covered = labelled_edges[['domain_id', 'neighbor_id']].drop_duplicates()
    covered = covered.groupby('domain_id').size().rename(
        f'{direction}_covered'
    ).reset_index()
    sums = labelled_edges.groupby(['domain_id', 'tracker_id'])['weight'].sum().reset_index()
    wide = sums.pivot_table(
        index='domain_id', columns='tracker_id', values='weight',
        aggfunc='sum', fill_value=0
    )
    wide = wide.reindex(columns=list(range(355)), fill_value=0)
    totals = covered.set_index('domain_id')[f'{direction}_covered']
    wide = wide.div(totals, axis=0).fillna(0)
    wide = wide.rename(
        columns={i: f'{direction}_{i}' for i in range(355)}
    ).reset_index()
    block = degree.merge(covered, on='domain_id', how='left')
    block = block.merge(wide, on='domain_id', how='left').fillna(0)
    return block.skb.set_name(f'{direction}_neighbor_evidence')


def build():
    setup = build_evaluation()
    X, y = setup['X'], setup['y']
    tracking = skrub.as_data_op(
        ROOT + 'tracking_graph_train.parquet'
    ).skb.apply_func(pd.read_parquet, columns=['domain_id', 'tracker_id'])
    tracking = tracking.drop_duplicates(['domain_id', 'tracker_id'])
    counts = tracking.groupby('domain_id').size().rename('tracker_count').reset_index()
    eligible = counts[counts['tracker_count'] >= 2][['domain_id']]
    entire_locked_pool = eligible.sample(n=POOL_SIZE, random_state=42).reset_index(drop=True)
    targets = skrub.as_data_op(ROOT + 'target.tsv').skb.apply_func(pd.read_csv, sep='\t')
    relevant = entire_locked_pool.skb.concat(
        [targets[['domain_id']]], axis=0
    ).drop_duplicates('domain_id')
    links = skrub.as_data_op(ROOT + 'link-graph.parquet').skb.apply_func(
        pd.read_parquet, columns=['source_domain_id', 'target_domain_id']
    )
    incident = links[
        links['source_domain_id'].isin(relevant['domain_id'])
        | links['target_domain_id'].isin(relevant['domain_id'])
    ]
    incident = incident[
        incident['source_domain_id'] != incident['target_domain_id']
    ].drop_duplicates()
    outgoing = incident[
        incident['source_domain_id'].isin(relevant['domain_id'])
    ].rename(columns={'source_domain_id': 'domain_id', 'target_domain_id': 'neighbor_id'})
    incoming = incident[
        incident['target_domain_id'].isin(relevant['domain_id'])
    ].rename(columns={'target_domain_id': 'domain_id', 'source_domain_id': 'neighbor_id'})
    neighbors = outgoing[['neighbor_id']].skb.concat(
        [incoming[['neighbor_id']]], axis=0
    ).drop_duplicates()
    safe_labels = tracking[
        ~tracking['domain_id'].isin(entire_locked_pool['domain_id'])
        & tracking['domain_id'].isin(neighbors['neighbor_id'])
    ]
    safe_labels = safe_labels.merge(counts, on='domain_id', how='left')
    safe_labels = safe_labels.assign(weight=1.0 / safe_labels['tracker_count'])
    safe_labels = safe_labels[['domain_id', 'tracker_id', 'weight']].rename(
        columns={'domain_id': 'neighbor_id'}
    ).skb.set_name('disjoint_neighbor_labels')
    out_features = neighborhood_block(outgoing, safe_labels, 'outgoing')
    in_features = neighborhood_block(incoming, safe_labels, 'incoming')
    graph = X[['domain_id']].merge(
        out_features, on='domain_id', how='left'
    ).merge(in_features, on='domain_id', how='left').fillna(0)
    graph = graph.drop(columns=['domain_id']).skb.set_name('neighborhood_channels')
    hosts = X['domain'].fillna('').str.lower().skb.set_name('hostname_normalization')
    hashed = hosts.skb.apply(
        HashingVectorizer(
            analyzer='char', ngram_range=(2, 5), n_features=16384,
            alternate_sign=False, norm='l2', lowercase=False, dtype=np.float32
        ), no_wrap=True
    ).skb.set_name('hostname_character_hash')
    features = skrub.as_data_op(
        {'graph': graph, 'hostname': hashed}
    ).skb.set_name('graph_hostname_experts')
    width = skrub.choose_from([128, 256], name='joint_residual_width')
    pred = features.skb.apply(
        GraphHostnameResidual(residual_width=width), y=y
    ).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}