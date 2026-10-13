import numpy as np
import pandas as pd
import skrub
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

class CandidateNetwork(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.embedding = nn.Embedding(355, 16)
        self.scorer = nn.Sequential(
            nn.Linear(n_features + 16, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, features):
        embedding = self.embedding.weight.unsqueeze(0).expand(len(features), -1, -1)
        return self.scorer(torch.cat([features, embedding], dim=2)).squeeze(2)

class GraphGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.0)
    def forward(self, tree, neural, metadata):
        et = -(tree * tree.clamp_min(1e-9).log()).sum(1, keepdim=True)
        en = -(neural * neural.clamp_min(1e-9).log()).sum(1, keepdim=True)
        weight = torch.sigmoid(self.gate(torch.cat([metadata.reshape(len(tree), -1), et, en], dim=1)))
        return (1 - weight) * tree + weight * neural, weight

class TreeMixtureNetwork(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hostname = nn.Linear(n_features, 355)
        self.mixture_gate = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.mixture_gate[-1].weight)
        nn.init.constant_(self.mixture_gate[-1].bias, -1.0)
    def forward(self, graph, metadata, hostname):
        host = torch.softmax(self.hostname(hostname), dim=1)
        eg = -(graph * graph.clamp_min(1e-9).log()).sum(1, keepdim=True)
        eh = -(host * host.clamp_min(1e-9).log()).sum(1, keepdim=True)
        weight = torch.sigmoid(self.mixture_gate(torch.cat([metadata.reshape(len(graph), -1), eg, eh], dim=1)))
        return (1 - weight) * graph + weight * host

class BoostedNeuralGraphMixture(BaseEstimator):
    def __init__(self, capacity=(600, 6), epochs=24, batch_size=512, neural_epochs=20):
        self.capacity = capacity
        self.epochs = epochs
        self.batch_size = batch_size
        self.neural_epochs = neural_epochs

    def _arrays(self, X):
        frame = X['graph']
        vectors = np.stack([frame[[f'{d}_{i}' for i in range(355)]].to_numpy(dtype=np.float32)
                            for d in ['outgoing', 'incoming']], axis=1)
        metadata = np.stack([frame[[f'{d}_degree', f'{d}_covered']].to_numpy(dtype=np.float32)
                             for d in ['outgoing', 'incoming']], axis=1)
        direct = np.stack([frame[[f'{d}_{i}' for i in range(355)]].to_numpy(dtype=np.float32)
                           for d in ['direct_out', 'direct_in']], axis=1)
        return vectors, np.log1p(metadata), X['hostname'].tocsr().astype(np.float32), direct

    def _pairs(self, vectors, metadata, direct):
        n = len(vectors)
        data = {}
        for channel, name in enumerate(['outgoing', 'incoming']):
            values = vectors[:, channel]
            order = np.argsort(-values, axis=1, kind='stable')
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.arange(355)[None, :], axis=1)
            data[name + '_evidence'] = values.reshape(-1)
            data[name + '_rank'] = (ranks.astype(np.float32) / 355).reshape(-1)
            entropy = -(values * np.log(values.clip(min=1e-9))).sum(axis=1)
            data[name + '_entropy'] = np.repeat(entropy, 355)
            data[name + '_degree'] = np.repeat(metadata[:, channel, 0], 355)
            data[name + '_covered'] = np.repeat(metadata[:, channel, 1], 355)
        data['direct_out'] = direct[:, 0].reshape(-1)
        data['direct_in'] = direct[:, 1].reshape(-1)
        frame = pd.DataFrame(data, copy=False)
        frame['tracker_id'] = pd.Categorical(np.tile(np.arange(355, dtype=np.int32), n),
                                              categories=list(range(355)))
        return frame

    def _new_tree(self):
        return XGBRegressor(n_estimators=self.capacity[0], max_depth=self.capacity[1],
                            learning_rate=0.05, objective='reg:squarederror',
                            tree_method='hist', device='cuda', enable_categorical=True,
                            max_cat_to_onehot=1, max_cat_threshold=64, max_bin=128,
                            n_jobs=16, random_state=42, subsample=1.0, colsample_bytree=1.0)

    def _graph_predict(self, model, vectors, metadata, direct):
        result = []
        for start in range(0, len(vectors), 1024):
            stop = start + 1024
            raw = model.predict(self._pairs(vectors[start:stop], metadata[start:stop], direct[start:stop]))
            probability = np.maximum(raw.reshape(-1, 355), 0) + 1e-8
            probability /= probability.sum(axis=1, keepdims=True)
            result.append(probability)
        return np.concatenate(result).astype(np.float32)

    def _neural_features(self, vectors, metadata, direct):
        result = []
        for start in range(0, len(vectors), 1024):
            stop = start + 1024
            frame = self._pairs(vectors[start:stop], metadata[start:stop], direct[start:stop])
            result.append(frame.drop(columns=['tracker_id']).to_numpy(dtype=np.float32).reshape(-1, 355, 12))
        return np.concatenate(result)

    def _fit_neural(self, features, labels, indices):
        torch.manual_seed(42)
        net = CandidateNetwork(features.shape[2]).to(self.device_)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6)
        loader = DataLoader(TensorDataset(torch.from_numpy(indices)), batch_size=256,
                            shuffle=True, generator=torch.Generator().manual_seed(42))
        for epoch in range(self.neural_epochs):
            total = 0.0
            net.train()
            for rows_tensor, in loader:
                rows = rows_tensor.numpy()
                values = torch.from_numpy(features[rows]).to(self.device_)
                target = torch.from_numpy(labels[rows]).to(self.device_)
                optimizer.zero_grad()
                logits = net(values)
                loss = -(target * torch.log_softmax(logits, dim=1)).sum(1).mean()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(rows)
            if epoch % 5 == 0:
                print('candidate neural epoch', epoch, 'loss', total / len(indices))
        return net

    def _neural_predict(self, net, features, indices):
        result = []
        net.eval()
        with torch.no_grad():
            for start in range(0, len(indices), 256):
                rows = indices[start:start + 256]
                values = torch.from_numpy(features[rows]).to(self.device_)
                result.append(torch.softmax(net(values), dim=1).cpu().numpy())
        return np.concatenate(result)

    def _combine(self, tree, neural, metadata):
        result, weights = [], []
        self.graph_gate_.eval()
        with torch.no_grad():
            for start in range(0, len(tree), self.batch_size):
                stop = start + self.batch_size
                t = torch.from_numpy(tree[start:stop]).to(self.device_)
                n = torch.from_numpy(neural[start:stop]).to(self.device_)
                m = torch.from_numpy(metadata[start:stop]).to(self.device_)
                probability, weight = self.graph_gate_(t, n, m)
                result.append(probability.cpu().numpy())
                weights.append(weight.cpu().numpy())
        return np.concatenate(result), np.concatenate(weights).ravel()

    def _report(self, name, probability, truth, metadata, weights=None):
        top = np.argsort(-probability, axis=1, kind='stable')[:, :10]
        recall = np.take_along_axis(truth, top, axis=1).sum(1) / truth.sum(1)
        coverage = np.expm1(metadata[:, :, 1]).sum(1)
        print(name, 'inner Recall@10', float(recall.mean()))
        for stratum, mask in [('zero', coverage == 0), ('low', (coverage > 0) & (coverage <= 5)),
                               ('high', coverage > 5)]:
            print(name, stratum, int(mask.sum()),
                  float(recall[mask].mean()) if mask.any() else None,
                  'neural gate mean', float(weights[mask].mean()) if weights is not None and mask.any() else None)

    def fit(self, X, y):
        torch.set_num_threads(16)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        vectors, metadata, hosts, direct = self._arrays(X)
        truth = np.asarray(y, dtype=np.float32)
        labels = truth / truth.sum(axis=1, keepdims=True).clip(min=1)
        neural_features = self._neural_features(vectors, metadata, direct)
        tree_oof = np.empty_like(labels)
        neural_oof = np.empty_like(labels)
        for train, valid in KFold(n_splits=2, shuffle=True, random_state=42).split(labels):
            tree = self._new_tree()
            pairs = self._pairs(vectors[train], metadata[train], direct[train])
            tree.fit(pairs, labels[train].reshape(-1))
            del pairs
            tree_oof[valid] = self._graph_predict(tree, vectors[valid], metadata[valid], direct[valid])
            del tree
            neural = self._fit_neural(neural_features, labels, train)
            neural_oof[valid] = self._neural_predict(neural, neural_features, valid)
            del neural
        self._report('tree', tree_oof, truth, metadata)
        self._report('neural', neural_oof, truth, metadata)

        torch.manual_seed(42)
        self.graph_gate_ = GraphGate().to(self.device_)
        optimizer = torch.optim.Adam(self.graph_gate_.parameters(), lr=0.002)
        loader = DataLoader(TensorDataset(torch.arange(len(labels))), batch_size=self.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(42))
        for epoch in range(20):
            self.graph_gate_.train()
            for indices, in loader:
                rows = indices.numpy()
                t = torch.from_numpy(tree_oof[rows]).to(self.device_)
                n = torch.from_numpy(neural_oof[rows]).to(self.device_)
                m = torch.from_numpy(metadata[rows]).to(self.device_)
                target = torch.from_numpy(labels[rows]).to(self.device_)
                optimizer.zero_grad()
                probability, weight = self.graph_gate_(t, n, m)
                loss = -(target * probability.clamp_min(1e-9).log()).sum(1).mean()
                loss.backward()
                optimizer.step()
        graph, weights = self._combine(tree_oof, neural_oof, metadata)
        self._report('combined OOF experts, fitted gate', graph, truth, metadata, weights)
        print('neural mixture weight quantiles', np.quantile(weights, [0, .1, .5, .9, 1]))

        self.tree_ = self._new_tree()
        pairs = self._pairs(vectors, metadata, direct)
        self.tree_.fit(pairs, labels.reshape(-1))
        del pairs
        self.neural_ = self._fit_neural(neural_features, labels, np.arange(len(labels)))
        del neural_features

        torch.manual_seed(42)
        self.net_ = TreeMixtureNetwork(hosts.shape[1]).to(self.device_)
        host_optimizer = torch.optim.Adam(self.net_.hostname.parameters(), lr=0.001, weight_decay=1e-6)
        gate_optimizer = torch.optim.Adam(self.net_.mixture_gate.parameters(), lr=0.002)
        loader = DataLoader(TensorDataset(torch.arange(len(labels))), batch_size=self.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(42))
        for epoch in range(self.epochs):
            total = 0.0
            self.net_.train()
            for indices, in loader:
                rows = indices.numpy()
                g = torch.from_numpy(graph[rows]).to(self.device_)
                m = torch.from_numpy(metadata[rows]).to(self.device_)
                h = torch.from_numpy(hosts[rows].toarray()).to(self.device_)
                target = torch.from_numpy(labels[rows]).to(self.device_)
                host_optimizer.zero_grad()
                gate_optimizer.zero_grad()
                probability = self.net_(g, m, h)
                loss = -(target * probability.clamp_min(1e-9).log()).sum(1).mean()
                loss.backward()
                host_optimizer.step()
                gate_optimizer.step()
                total += float(loss.detach()) * len(rows)
            if epoch % 6 == 0:
                print('hostname mixture epoch', epoch, 'loss', total / len(labels))
        return self

    def predict(self, X):
        vectors, metadata, hosts, direct = self._arrays(X)
        tree = self._graph_predict(self.tree_, vectors, metadata, direct)
        values = self._neural_features(vectors, metadata, direct)
        neural = self._neural_predict(self.neural_, values, np.arange(len(vectors)))
        del values
        graph, weights = self._combine(tree, neural, metadata)
        print('prediction neural gate mean', float(weights.mean()))
        self.net_.eval()
        result = []
        with torch.no_grad():
            for start in range(0, len(vectors), self.batch_size):
                stop = start + self.batch_size
                g = torch.from_numpy(graph[start:stop]).to(self.device_)
                m = torch.from_numpy(metadata[start:stop]).to(self.device_)
                h = torch.from_numpy(hosts[start:stop].toarray()).to(self.device_)
                result.append(self.net_(g, m, h).topk(10, dim=1).indices.cpu().numpy())
        return np.concatenate(result)

def neighborhood_block(edges, labels, direction):
    degree = edges.groupby('domain_id').size().rename(f'{direction}_degree').reset_index()
    labelled_edges = edges.merge(labels, on='neighbor_id', how='inner')
    covered = labelled_edges[['domain_id', 'neighbor_id']].drop_duplicates()
    covered = covered.groupby('domain_id').size().rename(f'{direction}_covered').reset_index()
    sums = labelled_edges.groupby(['domain_id', 'tracker_id'])['weight'].sum().reset_index()
    wide = sums.pivot_table(index='domain_id', columns='tracker_id', values='weight', aggfunc='sum', fill_value=0)
    wide = wide.reindex(columns=list(range(355)), fill_value=0)
    wide = wide.div(covered.set_index('domain_id')[f'{direction}_covered'], axis=0).fillna(0)
    wide = wide.rename(columns={i: f'{direction}_{i}' for i in range(355)}).reset_index()
    block = degree.merge(covered, on='domain_id', how='left')
    block = block.merge(wide, on='domain_id', how='left').fillna(0)
    return block.skb.set_name(f'{direction}_neighbor_evidence')

def direct_block(edges, trackers, direction):
    pairs = edges.merge(trackers, left_on='neighbor_id', right_on='tracking_domain_id', how='inner')
    pairs = pairs[['domain_id', 'tracker_id']].drop_duplicates()
    wide = pairs.assign(present=1).pivot_table(index='domain_id', columns='tracker_id', values='present', aggfunc='max', fill_value=0)
    wide = wide.reindex(columns=list(range(355)), fill_value=0)
    return wide.rename(columns={i: f'{direction}_{i}' for i in range(355)}).reset_index().skb.set_name(direction + '_indicators')

def build():
    setup = build_evaluation()
    X, y = setup['X'], setup['y']
    tracking = skrub.as_data_op(ROOT + 'tracking_graph_train.parquet').skb.apply_func(pd.read_parquet, columns=['domain_id', 'tracker_id'])
    tracking = tracking.drop_duplicates(['domain_id', 'tracker_id'])
    counts = tracking.groupby('domain_id').size().rename('tracker_count').reset_index()
    eligible = counts[counts['tracker_count'] >= 2][['domain_id']]
    entire_locked_pool = eligible.sample(n=POOL_SIZE, random_state=42).reset_index(drop=True)
    targets = skrub.as_data_op(ROOT + 'target.tsv').skb.apply_func(pd.read_csv, sep='\t')
    relevant = entire_locked_pool.skb.concat([targets[['domain_id']]], axis=0).drop_duplicates('domain_id')
    links = skrub.as_data_op(ROOT + 'link-graph.parquet').skb.apply_func(pd.read_parquet, columns=['source_domain_id', 'target_domain_id'])
    incident = links[links['source_domain_id'].isin(relevant['domain_id']) | links['target_domain_id'].isin(relevant['domain_id'])]
    incident = incident[incident['source_domain_id'] != incident['target_domain_id']].drop_duplicates()
    outgoing = incident[incident['source_domain_id'].isin(relevant['domain_id'])].rename(columns={'source_domain_id': 'domain_id', 'target_domain_id': 'neighbor_id'})
    incoming = incident[incident['target_domain_id'].isin(relevant['domain_id'])].rename(columns={'target_domain_id': 'domain_id', 'source_domain_id': 'neighbor_id'})
    neighbors = outgoing[['neighbor_id']].skb.concat([incoming[['neighbor_id']]], axis=0).drop_duplicates()
    safe_labels = tracking[~tracking['domain_id'].isin(entire_locked_pool['domain_id']) & tracking['domain_id'].isin(neighbors['neighbor_id'])]
    safe_labels = safe_labels.merge(counts, on='domain_id', how='left')
    safe_labels = safe_labels.assign(weight=1.0 / safe_labels['tracker_count'])
    safe_labels = safe_labels[['domain_id', 'tracker_id', 'weight']].rename(columns={'domain_id': 'neighbor_id'}).skb.set_name('disjoint_neighbor_labels')
    out_features = neighborhood_block(outgoing, safe_labels, 'outgoing')
    in_features = neighborhood_block(incoming, safe_labels, 'incoming')
    trackers = skrub.as_data_op(ROOT + 'trackers.tsv').skb.apply_func(pd.read_csv, sep='\t', usecols=['tracking_domain_id', 'tracker_id'])
    direct_out = direct_block(outgoing, trackers, 'direct_out')
    direct_in = direct_block(incoming, trackers, 'direct_in')
    graph = X[['domain_id']].merge(out_features, on='domain_id', how='left').merge(in_features, on='domain_id', how='left').fillna(0)
    graph = graph.merge(direct_out, on='domain_id', how='left').merge(direct_in, on='domain_id', how='left').fillna(0)
    graph = graph.drop(columns=['domain_id']).skb.set_name('neighborhood_channels')
    target_direct = targets[['domain_id']].merge(direct_out, on='domain_id', how='left').merge(direct_in, on='domain_id', how='left').fillna(0).drop(columns=['domain_id']).skb.set_name('target_direct_link_audit')
    hosts = X['domain'].fillna('').str.lower().skb.set_name('hostname_normalization')
    hashed = hosts.skb.apply(HashingVectorizer(analyzer='char', ngram_range=(2, 5), n_features=16384, alternate_sign=False, norm='l2', lowercase=False, dtype=np.float32), no_wrap=True).skb.set_name('hostname_character_hash')
    features = skrub.as_data_op({'graph': graph, 'hostname': hashed, 'target_direct': target_direct}).skb.set_name('graph_hostname_experts')
    pred = features.skb.apply(BoostedNeuralGraphMixture(), y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}