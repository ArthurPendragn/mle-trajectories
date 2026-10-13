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

class HostCNN(nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 32, padding_idx=0)
        self.convolutions = nn.ModuleList([nn.Conv1d(32, 64, width, padding=width // 2) for width in (3, 5, 7)])
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(192, 355)

    def forward(self, tokens):
        mask = tokens.ne(0).unsqueeze(1)
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = []
        for convolution in self.convolutions:
            values = torch.relu(convolution(embedded))
            values = values.masked_fill(~mask, -1e4).amax(2)
            values = torch.where(mask.any(2), values, torch.zeros_like(values))
            pooled.append(values)
        return self.head(self.dropout(torch.cat(pooled, 1)))

class ExpertGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(7, 16), nn.ReLU(), nn.Linear(16, 3))
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.tensor([0.0, -1.0, -1.0]))

    def forward(self, probabilities, metadata):
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(2)
        inputs = torch.cat([metadata.reshape(len(metadata), -1), entropy], 1)
        weights = torch.softmax(self.network(inputs), 1)
        return (weights.unsqueeze(2) * probabilities).sum(1)

class SequenceGraphEnsemble(BaseEstimator):
    def __init__(self, capacity=(600, 6), cnn_epochs=12, linear_epochs=24, gate_epochs=24, batch_size=512):
        self.capacity = capacity
        self.cnn_epochs = cnn_epochs
        self.linear_epochs = linear_epochs
        self.gate_epochs = gate_epochs
        self.batch_size = batch_size

    def _context_arrays(self, frame):
        priors = np.stack([frame[[f'{key}_prior_{i}' for i in range(355)]].to_numpy(dtype=np.float32) for key in ['tld', 'category']], 1)
        support = frame[['tld_support', 'category_support', 'category_missing']].to_numpy(dtype=np.float32)
        return priors, support

    def _arrays(self, X):
        frame = X['graph']
        vectors = np.stack([frame[[f'{d}_{i}' for i in range(355)]].to_numpy(dtype=np.float32) for d in ['outgoing', 'incoming']], 1)
        metadata = np.stack([frame[[f'{d}_degree', f'{d}_covered']].to_numpy(dtype=np.float32) for d in ['outgoing', 'incoming']], 1)
        direct = np.stack([frame[[f'{d}_{i}' for i in range(355)]].to_numpy(dtype=np.float32) for d in ['direct_out', 'direct_in']], 1)
        context, support = self._context_arrays(X['context'])
        return vectors, np.log1p(metadata), X['hostname'].tocsr().astype(np.float32), direct, context, support

    def _tokens(self, strings, audit_name=None):
        strings = np.asarray(strings).reshape(-1)
        tokens = np.zeros((len(strings), 96), dtype=np.int64)
        unknown = 0
        total = 0
        truncated = 0
        for i, value in enumerate(strings):
            text = str(value)
            truncated += len(text) > 96
            # Preserve both initial motifs and suffix structure on long names.
            text = text if len(text) <= 96 else text[:48] + text[-48:]
            for j, character in enumerate(text):
                token = self.characters_.get(character, 1)
                tokens[i, j] = token
                unknown += token == 1
                total += 1
        if audit_name:
            print(audit_name, 'hostname truncation fraction', truncated / max(len(strings), 1),
                  'character coverage', 1 - unknown / max(total, 1))
        return tokens

    def _pairs(self, vectors, metadata, direct, context, support):
        n = len(vectors)
        data = {}
        for channel, name in enumerate(['outgoing', 'incoming']):
            values = vectors[:, channel]
            order = np.argsort(-values, axis=1, kind='stable')
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.arange(355)[None, :], axis=1)
            data[name + '_evidence'] = values.reshape(-1)
            data[name + '_rank'] = (ranks.astype(np.float32) / 355).reshape(-1)
            entropy = -(values * np.log(values.clip(min=1e-9))).sum(1)
            data[name + '_entropy'] = np.repeat(entropy, 355)
            data[name + '_degree'] = np.repeat(metadata[:, channel, 0], 355)
            data[name + '_covered'] = np.repeat(metadata[:, channel, 1], 355)
        data['direct_out'] = direct[:, 0].reshape(-1)
        data['direct_in'] = direct[:, 1].reshape(-1)
        for channel, name in enumerate(['tld', 'category']):
            data[name + '_prior'] = context[:, channel].reshape(-1)
            data[name + '_support'] = np.repeat(np.log1p(support[:, channel]), 355)
        data['category_missing'] = np.repeat(support[:, 2], 355)
        frame = pd.DataFrame(data, copy=False)
        frame['tracker_id'] = pd.Categorical(np.tile(np.arange(355, dtype=np.int32), n), categories=list(range(355)))
        return frame

    def _new_tree(self):
        return XGBRegressor(n_estimators=self.capacity[0], max_depth=self.capacity[1], learning_rate=0.05,
                            objective='reg:squarederror', tree_method='hist', device='cuda',
                            enable_categorical=True, max_cat_to_onehot=1, max_cat_threshold=64,
                            max_bin=128, n_jobs=16, random_state=42, subsample=1.0, colsample_bytree=1.0)

    def _graph_predict(self, model, vectors, metadata, direct, context, support):
        results = []
        for start in range(0, len(vectors), 1024):
            stop = start + 1024
            raw = model.predict(self._pairs(vectors[start:stop], metadata[start:stop], direct[start:stop], context[start:stop], support[start:stop]))
            probability = np.maximum(raw.reshape(-1, 355), 0) + 1e-8
            probability /= probability.sum(1, keepdims=True)
            results.append(probability)
        return np.concatenate(results).astype(np.float32)

    def _train_host(self, hosts, tokens, labels, rows, kind):
        torch.manual_seed(42)
        if kind == 'cnn':
            network = HostCNN(len(self.characters_) + 2).to(self.device_)
            epochs = self.cnn_epochs
        else:
            network = nn.Linear(hosts.shape[1], 355).to(self.device_)
            epochs = self.linear_epochs
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-6)
        loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(rows, dtype=np.int64))),
                            batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        for epoch in range(epochs):
            network.train()
            total = 0.0
            for (indices,) in loader:
                selection = indices.numpy()
                if kind == 'cnn':
                    inputs = torch.from_numpy(tokens[selection]).to(self.device_)
                else:
                    inputs = torch.from_numpy(hosts[selection].toarray()).to(self.device_)
                target = torch.from_numpy(labels[selection]).to(self.device_)
                optimizer.zero_grad()
                logits = network(inputs)
                loss = -(target * torch.log_softmax(logits, 1)).sum(1).mean()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(selection)
            if epoch % 6 == 0:
                print(kind, 'epoch', epoch, 'loss', total / len(rows))
        return network

    def _host_predict(self, network, hosts, tokens, rows, kind):
        network.eval()
        results = []
        with torch.no_grad():
            for start in range(0, len(rows), self.batch_size):
                selection = rows[start:start + self.batch_size]
                if kind == 'cnn':
                    inputs = torch.from_numpy(tokens[selection]).to(self.device_)
                else:
                    inputs = torch.from_numpy(hosts[selection].toarray()).to(self.device_)
                results.append(torch.softmax(network(inputs), 1).cpu().numpy())
        return np.concatenate(results)

    def _fit_gate(self, probabilities, metadata, labels, rows, channels):
        torch.manual_seed(42)
        gate = ExpertGate().to(self.device_)
        optimizer = torch.optim.Adam(gate.parameters(), lr=0.002)
        loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(rows, dtype=np.int64))),
                            batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        for epoch in range(self.gate_epochs):
            for (indices,) in loader:
                selected = indices.numpy()
                p = torch.from_numpy(probabilities[selected]).to(self.device_)
                if channels == 2:
                    # The CNN slot duplicates the linear expert for the parent comparison.
                    p = torch.stack([p[:, 0], p[:, 1], p[:, 1]], 1)
                m = torch.from_numpy(metadata[selected]).to(self.device_)
                target = torch.from_numpy(labels[selected]).to(self.device_)
                optimizer.zero_grad()
                output = gate(p, m)
                loss = -(target * output.clamp_min(1e-9).log()).sum(1).mean()
                loss.backward()
                optimizer.step()
        return gate

    def _gate_predict(self, gate, probabilities, metadata, channels=3):
        gate.eval()
        results = []
        with torch.no_grad():
            for start in range(0, len(probabilities), self.batch_size):
                p = torch.from_numpy(probabilities[start:start + self.batch_size]).to(self.device_)
                if channels == 2:
                    p = torch.stack([p[:, 0], p[:, 1], p[:, 1]], 1)
                m = torch.from_numpy(metadata[start:start + self.batch_size]).to(self.device_)
                results.append(gate(p, m).cpu().numpy())
        return np.concatenate(results)

    def _recall(self, probabilities, truth):
        top = np.argsort(-probabilities, axis=1, kind='stable')[:, :10]
        return np.take_along_axis(truth, top, 1).sum(1) / truth.sum(1)

    def fit(self, X, y):
        torch.set_num_threads(16)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.characters_ = {character: i + 2 for i, character in enumerate("abcdefghijklmnopqrstuvwxyz0123456789.-_")}
        vectors, metadata, hosts, direct, context, support = self._arrays(X)
        tokens = self._tokens(X['sequence'], 'training')
        self._tokens(X['target_sequence'], 'target')
        truth = np.asarray(y, dtype=np.float32)
        labels = truth / truth.sum(1, keepdims=True).clip(min=1)
        probabilities = np.empty((len(labels), 3, 355), dtype=np.float32)
        inner = list(KFold(n_splits=2, shuffle=True, random_state=42).split(labels))
        for train, valid in inner:
            tree = self._new_tree()
            pairs = self._pairs(vectors[train], metadata[train], direct[train], context[train], support[train])
            tree.fit(pairs, labels[train].reshape(-1))
            del pairs
            probabilities[valid, 0] = self._graph_predict(tree, vectors[valid], metadata[valid], direct[valid], context[valid], support[valid])
            del tree
            linear = self._train_host(hosts, tokens, labels, train, 'linear')
            probabilities[valid, 1] = self._host_predict(linear, hosts, tokens, valid, 'linear')
            del linear
            cnn = self._train_host(hosts, tokens, labels, train, 'cnn')
            probabilities[valid, 2] = self._host_predict(cnn, hosts, tokens, valid, 'cnn')
            del cnn
        print('inner cross-fitted standalone CNN Recall@10', float(self._recall(probabilities[:, 2], truth).mean()))
        # Cross-fit gates too, so this diagnostic is not a gate-training score.
        parent = np.empty_like(labels)
        ensemble = np.empty_like(labels)
        for train, valid in KFold(n_splits=2, shuffle=True, random_state=713).split(labels):
            parent_gate = self._fit_gate(probabilities, metadata, labels, train, 2)
            parent[valid] = self._gate_predict(parent_gate, probabilities[valid], metadata[valid], 2)
            del parent_gate
            gate = self._fit_gate(probabilities, metadata, labels, train, 3)
            ensemble[valid] = self._gate_predict(gate, probabilities[valid], metadata[valid])
            del gate
        parent_recall = self._recall(parent, truth)
        ensemble_recall = self._recall(ensemble, truth)
        cnn_recall = self._recall(probabilities[:, 2], truth)
        print('inner cross-fitted parent/ensemble Recall@10', float(parent_recall.mean()), float(ensemble_recall.mean()))
        coverage = np.expm1(metadata[:, :, 1]).sum(1)
        for name, mask in [('zero', coverage == 0), ('low', (coverage > 0) & (coverage <= 5)), ('high', coverage > 5)]:
            if mask.any():
                print('coverage', name, int(mask.sum()), 'CNN recall', float(cnn_recall[mask].mean()),
                      'parent/ensemble', float(parent_recall[mask].mean()), float(ensemble_recall[mask].mean()),
                      'net missed-recall recovery', float((ensemble_recall[mask] - parent_recall[mask]).sum() / (1 - parent_recall[mask]).sum().clip(min=1e-9)))
        self.gate_ = self._fit_gate(probabilities, metadata, labels, np.arange(len(labels)), 3)
        self.tree_ = self._new_tree()
        pairs = self._pairs(vectors, metadata, direct, context, support)
        self.tree_.fit(pairs, labels.reshape(-1))
        del pairs
        rows = np.arange(len(labels))
        self.linear_ = self._train_host(hosts, tokens, labels, rows, 'linear')
        self.cnn_ = self._train_host(hosts, tokens, labels, rows, 'cnn')
        return self

    def predict(self, X):
        vectors, metadata, hosts, direct, context, support = self._arrays(X)
        tokens = self._tokens(X['sequence'], 'prediction')
        rows = np.arange(len(tokens))
        graph = self._graph_predict(self.tree_, vectors, metadata, direct, context, support)
        linear = self._host_predict(self.linear_, hosts, tokens, rows, 'linear')
        cnn = self._host_predict(self.cnn_, hosts, tokens, rows, 'cnn')
        probability = self._gate_predict(self.gate_, np.stack([graph, linear, cnn], 1), metadata)
        return np.argsort(-probability, axis=1, kind='stable')[:, :10]

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
    block = degree.merge(covered, on='domain_id', how='left').merge(wide, on='domain_id', how='left').fillna(0)
    return block.skb.set_name(f'{direction}_neighbor_evidence')

def direct_block(edges, trackers, direction):
    pairs = edges.merge(trackers, left_on='neighbor_id', right_on='tracking_domain_id', how='inner')
    pairs = pairs[['domain_id', 'tracker_id']].drop_duplicates()
    wide = pairs.assign(present=1).pivot_table(index='domain_id', columns='tracker_id', values='present', aggfunc='max', fill_value=0)
    wide = wide.reindex(columns=list(range(355)), fill_value=0)
    return wide.rename(columns={i: f'{direction}_{i}' for i in range(355)}).reset_index().skb.set_name(direction + '_indicators')

def context_prior_block(labels, source_metadata, global_prior, key):
    meta = source_metadata[['domain_id', key]]
    meta = meta[meta[key].notna() & (meta[key] != '')]
    support = meta.groupby(key).size().rename(key + '_support').reset_index()
    joined = labels.merge(meta, on='domain_id', how='inner')
    sums = joined.groupby([key, 'tracker_id'])['weight'].sum().reset_index()
    wide = sums.pivot_table(index=key, columns='tracker_id', values='weight', aggfunc='sum', fill_value=0)
    wide = wide.reindex(columns=list(range(355)), fill_value=0)
    wide = wide.add(global_prior * 100.0, axis='columns')
    wide = wide.div(support.set_index(key)[key + '_support'] + 100.0, axis=0)
    wide = wide.rename(columns={i: f'{key}_prior_{i}' for i in range(355)}).reset_index()
    return support.merge(wide, on=key, how='left').skb.set_name(key + '_disjoint_priors')

def attach_context(rows, metadata, tld_priors, category_priors, global_prior):
    result = rows[['domain_id']].merge(metadata[['domain_id', 'tld', 'category']], on='domain_id', how='left')
    result = result.assign(category_missing=result['category'].isna().astype('float32'))
    result = result.merge(tld_priors, on='tld', how='left').merge(category_priors, on='category', how='left')
    for key in ['tld', 'category']:
        for i in range(355):
            col = f'{key}_prior_{i}'
            result = result.assign(**{col: result[col].fillna(global_prior[i])})
    return result.drop(columns=['domain_id', 'tld', 'category']).fillna(0)

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
    source_pool = counts[~counts['domain_id'].isin(entire_locked_pool['domain_id'])]
    source_pool = source_pool.sample(n=1000000, random_state=117).reset_index(drop=True)
    source_labels = tracking.merge(source_pool, on='domain_id', how='inner')
    source_labels = source_labels.assign(weight=1.0 / source_labels['tracker_count'])
    source_labels = source_labels[['domain_id', 'tracker_id', 'weight']].skb.set_name('disjoint_context_labels')
    lookup_ids = source_pool[['domain_id']].skb.concat([relevant[['domain_id']]], axis=0).drop_duplicates('domain_id')
    domains = skrub.as_data_op(ROOT + 'domains.parquet').skb.apply_func(pd.read_parquet, columns=['domain_id', 'domain'])
    domains = domains[domains['domain_id'].isin(lookup_ids['domain_id'])]
    domains = domains.assign(domain=domains['domain'].fillna('').str.lower().str.rstrip('.'))
    domains = domains.assign(tld=domains['domain'].str.rsplit('.', n=1).str[-1])
    classified = skrub.as_data_op(ROOT + 'url-classification.csv').skb.apply_func(pd.read_csv, usecols=['url', 'category'])
    host = classified['url'].fillna('').str.lower().str.replace('^[a-z][a-z0-9+.-]*://', '', regex=True).str.split('/').str[0].str.split('?').str[0].str.split('#').str[0].str.split(':').str[0].str.rstrip('.')
    classified = classified.assign(domain=host, category=classified['category'].fillna('').astype(str).str.lower())
    classified = classified[classified['domain'].isin(domains['domain'])]
    classified = classified[classified['category'] != ''][['domain', 'category']]
    classified = classified.drop_duplicates().sort_values(['domain', 'category'])
    classified = classified.assign(category=classified['category'] + '|')
    classified = classified.groupby('domain')['category'].sum().reset_index()
    classified = classified.assign(category=classified['category'].str.rstrip('|'))
    metadata = domains.merge(classified, on='domain', how='left').skb.set_name('domain_context_metadata')
    source_metadata = source_pool[['domain_id']].merge(metadata, on='domain_id', how='left')
    global_prior = source_labels.groupby('tracker_id')['weight'].sum()
    global_prior = global_prior.reindex(list(range(355)), fill_value=0) / source_pool.shape[0]
    global_prior = global_prior.skb.set_name('disjoint_global_recall_prior')
    tld_priors = context_prior_block(source_labels, source_metadata, global_prior, 'tld')
    category_priors = context_prior_block(source_labels, source_metadata, global_prior, 'category')
    context = attach_context(X, metadata, tld_priors, category_priors, global_prior).skb.set_name('domain_context_priors')
    target_context = attach_context(targets, metadata, tld_priors, category_priors, global_prior).skb.set_name('target_context_audit')
    hosts = X['domain'].fillna('').str.lower().skb.set_name('hostname_normalization')
    hashed = hosts.skb.apply(HashingVectorizer(analyzer='char', ngram_range=(2, 5), n_features=16384, alternate_sign=False, norm='l2', lowercase=False, dtype=np.float32), no_wrap=True).skb.set_name('hostname_character_hash')
    sequence = hosts.str.rstrip('.').skb.set_name('hostname_sequence')
    target_sequence = targets[['domain_id']].merge(domains[['domain_id', 'domain']], on='domain_id', how='left')['domain'].fillna('').str.lower().str.rstrip('.').skb.set_name('target_hostname_sequence_audit')
    features = skrub.as_data_op({'graph': graph, 'hostname': hashed, 'target_direct': target_direct,
                                'context': context, 'target_context': target_context,
                                'sequence': sequence, 'target_sequence': target_sequence}).skb.set_name('graph_hostname_experts')
    pred = features.skb.apply(SequenceGraphEnsemble(), y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}