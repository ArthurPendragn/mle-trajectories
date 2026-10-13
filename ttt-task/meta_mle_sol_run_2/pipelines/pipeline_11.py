import numpy as np
import pandas as pd
import skrub
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold
from xgboost import XGBRanker
from concurrent.futures import ThreadPoolExecutor


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
        weight = torch.sigmoid(self.mixture_gate(
            torch.cat([metadata.reshape(len(graph), -1), eg, eh], dim=1)))
        return (1 - weight) * graph + weight * host


class RankedGraphMixture(BaseEstimator):
    def __init__(self, capacity=(600, 6), epochs=24, batch_size=512):
        self.capacity = capacity
        self.epochs = epochs
        self.batch_size = batch_size

    def _context_arrays(self, frame):
        priors = np.stack([
            frame[[f'{key}_prior_{i}' for i in range(355)]].to_numpy(dtype=np.float32)
            for key in ['tld', 'category']], axis=1)
        support = frame[['tld_support', 'category_support', 'category_missing']].to_numpy(dtype=np.float32)
        return priors, support

    def _arrays(self, X):
        frame = X['graph']
        vectors = np.stack([
            frame[[f'{d}_{i}' for i in range(355)]].to_numpy(dtype=np.float32)
            for d in ['outgoing', 'incoming']], axis=1)
        metadata = np.stack([
            frame[[f'{d}_degree', f'{d}_covered']].to_numpy(dtype=np.float32)
            for d in ['outgoing', 'incoming']], axis=1)
        direct = np.stack([
            frame[[f'{d}_{i}' for i in range(355)]].to_numpy(dtype=np.float32)
            for d in ['direct_out', 'direct_in']], axis=1)
        context, support = self._context_arrays(X['context'])
        return vectors, np.log1p(metadata), X['hostname'].tocsr().astype(np.float32), direct, context, support

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
            entropy = -(values * np.log(values.clip(min=1e-9))).sum(axis=1)
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
        frame['tracker_id'] = pd.Categorical(
            np.tile(np.arange(355, dtype=np.int32), n), categories=list(range(355)))
        return frame

    def _new_tree(self, device='cuda:0', n_jobs=16):
        return XGBRanker(
            n_estimators=self.capacity[0], max_depth=self.capacity[1],
            learning_rate=0.05, objective='rank:pairwise',
            lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=10,
            tree_method='hist', device=device, enable_categorical=True,
            max_cat_to_onehot=1, max_cat_threshold=64, max_bin=128,
            n_jobs=n_jobs, random_state=42, subsample=1.0, colsample_bytree=1.0)

    def _fit_tree(self, tree, pairs, truth):
        tree.fit(
            pairs, truth.reshape(-1),
            group=np.full(len(truth), 355, dtype=np.uint32),
            sample_weight=(1.0 / truth.sum(axis=1).clip(min=1)).astype(np.float32))
        return tree

    def _graph_predict(self, model, vectors, metadata, direct, context, support):
        result = []
        for start in range(0, len(vectors), 8192):
            stop = start + 8192
            margins = model.predict(self._pairs(
                vectors[start:stop], metadata[start:stop], direct[start:stop],
                context[start:stop], support[start:stop])).reshape(-1, 355)
            margins = margins - margins.max(axis=1, keepdims=True)
            probability = np.exp(margins)
            probability /= probability.sum(axis=1, keepdims=True)
            result.append(probability)
        return np.concatenate(result).astype(np.float32)

    def _audit_direct(self, direct, name):
        print(name, 'direct coverage', (direct.sum(2) > 0).mean(0),
              'pair prevalence', direct.mean(axis=(0, 2)),
              'candidate count quantiles',
              np.quantile(direct.sum(2), [0, .5, .9, .99, 1], axis=0))

    def _audit_context(self, context, support, name):
        print(name, 'category coverage', float((support[:, 2] == 0).mean()),
              'support quantiles', np.quantile(support[:, :2], [0, .5, .9, .99, 1], axis=0),
              'prior max quantiles', np.quantile(context.max(2), [0, .5, .9, .99, 1], axis=0))

    def _audit_context_signal(self, context, support, truth, metadata, name):
        coverage = np.expm1(metadata[:, :, 1]).sum(1)
        for channel, key in enumerate(['tld', 'category']):
            covered = support[:, channel] > 0
            top = np.argsort(-context[:, channel], axis=1, kind='stable')[:, :10]
            recall = np.take_along_axis(truth, top, axis=1).sum(1) / truth.sum(1)
            print(name, key, 'standalone covered Recall@10',
                  float(recall[covered].mean()) if covered.any() else None,
                  'covered rows', int(covered.sum()))
            for stratum, mask in [('zero', coverage == 0),
                                  ('low', (coverage > 0) & (coverage <= 5)),
                                  ('high', coverage > 5)]:
                mask = mask & covered
                print(name, key, stratum, int(mask.sum()),
                      float(recall[mask].mean()) if mask.any() else None)

    def _audit_signal(self, direct, truth, prior, name):
        union = direct.max(axis=1)
        covered = union.sum(1) > 0
        score = union + prior[None, :] / max(float(prior.max()), 1e-8) * .5
        top = np.argsort(-score, axis=1, kind='stable')[:, :10]
        recall = np.take_along_axis(truth, top, axis=1).sum(1) / truth.sum(1)
        print(name, 'standalone direct-link Recall@10 covered',
              float(recall[covered].mean()) if covered.any() else None,
              'covered rows', int(covered.sum()))
        for channel in range(2):
            mask = direct[:, channel] > 0
            print(name, 'direct channel', channel, 'positive precision',
                  float(truth[mask].mean()) if mask.any() else None)

    def _inner_fit(self, train, valid, device, n_jobs, vectors, metadata, direct,
                   context, support, truth):
        tree = self._new_tree(device=device, n_jobs=n_jobs)
        pairs = self._pairs(vectors[train], metadata[train], direct[train],
                            context[train], support[train])
        self._fit_tree(tree, pairs, truth[train])
        del pairs
        probability = self._graph_predict(
            tree, vectors[valid], metadata[valid], direct[valid],
            context[valid], support[valid])
        return valid, probability

    def _final_fit(self, device, n_jobs, vectors, metadata, direct, context, support, truth):
        tree = self._new_tree(device=device, n_jobs=n_jobs)
        pairs = self._pairs(vectors, metadata, direct, context, support)
        self._fit_tree(tree, pairs, truth)
        return tree

    def _crossfits(self, splits, device, n_jobs, vectors, metadata, direct,
                   context, support, truth):
        # The two half-sized fits occupy GPU 1 while the full fit occupies GPU 0.
        # Run them sequentially here to avoid competing GPU kernels and allocations.
        results = []
        for train, valid in splits:
            results.append(self._inner_fit(
                train, valid, device, n_jobs, vectors, metadata, direct,
                context, support, truth))
        return results

    def fit(self, X, y):
        import time
        started = time.perf_counter()
        print('Pairwise top-k ranking is an ordering surrogate, not exact Recall@10.')
        torch.set_num_threads(16)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.device_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        vectors, metadata, hosts, direct, context, support = self._arrays(X)
        truth = (np.asarray(y) > 0).astype(np.float32)
        labels = truth / truth.sum(axis=1, keepdims=True).clip(min=1)
        self.prior_ = labels.mean(0)
        self._audit_direct(direct, 'training')
        self._audit_context(context, support, 'training')
        target_frame = X['target_direct']
        target_direct = np.stack([
            target_frame[[f'{d}_{i}' for i in range(355)]].to_numpy(dtype=np.float32)
            for d in ['direct_out', 'direct_in']], axis=1)
        self._audit_direct(target_direct, 'target')
        target_context, target_support = self._context_arrays(X['target_context'])
        self._audit_context(target_context, target_support, 'target')
        graph = np.empty_like(labels)
        splits = list(KFold(n_splits=2, shuffle=True, random_state=42).split(labels))
        for train, valid in splits:
            self._audit_signal(direct[valid], truth[valid], labels[train].mean(0), 'inner held-out')
            self._audit_context_signal(context[valid], support[valid], truth[valid],
                                       metadata[valid], 'inner held-out')
        if torch.cuda.device_count() >= 2:
            # These fits are independent: final training does not depend on OOF
            # probabilities. Overlap the former sequential full-pool fit with
            # cross-fitting, balancing 160k rows of work on each GPU.
            with ThreadPoolExecutor(max_workers=2) as executor:
                final_future = executor.submit(
                    self._final_fit, 'cuda:0', 8, vectors, metadata,
                    direct, context, support, truth)
                inner_future = executor.submit(
                    self._crossfits, splits, 'cuda:1', 8, vectors,
                    metadata, direct, context, support, truth)
                for valid, probability in inner_future.result():
                    graph[valid] = probability
                self.tree_ = final_future.result()
        else:
            for valid, probability in self._crossfits(
                    splits, self.device_, 16, vectors, metadata, direct,
                    context, support, truth):
                graph[valid] = probability
            self.tree_ = self._final_fit(
                self.device_, 16, vectors, metadata, direct, context, support, truth)
        top = np.argsort(-graph, axis=1)[:, :10]
        recall = np.take_along_axis(truth, top, axis=1).sum(1) / truth.sum(1)
        print('inner cross-fitted graph Recall@10', float(recall.mean()))
        coverage = np.expm1(metadata[:, :, 1]).sum(1)
        for name, mask in [('zero', coverage == 0),
                           ('low', (coverage > 0) & (coverage <= 5)),
                           ('high', coverage > 5)]:
            print('inner graph recall stratum', name, int(mask.sum()),
                  float(recall[mask].mean()) if mask.any() else None)
        self.net_ = TreeMixtureNetwork(hosts.shape[1]).to(self.device_)
        host_optimizer = torch.optim.Adam(self.net_.hostname.parameters(), lr=.001, weight_decay=1e-6)
        gate_optimizer = torch.optim.Adam(self.net_.mixture_gate.parameters(), lr=.002)
        loader = DataLoader(TensorDataset(torch.arange(len(labels))),
                            batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
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
                print('tree mixture epoch', epoch, 'loss', total / len(labels))
        print('complete model fit seconds', time.perf_counter() - started)
        return self

    def predict(self, X):
        vectors, metadata, hosts, direct, context, support = self._arrays(X)
        self._audit_direct(direct, 'held-out/prediction')
        self._audit_context(context, support, 'held-out/prediction')
        graph = self._graph_predict(self.tree_, vectors, metadata, direct, context, support)
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
    wide = sums.pivot_table(index='domain_id', columns='tracker_id', values='weight',
                           aggfunc='sum', fill_value=0)
    wide = wide.reindex(columns=list(range(355)), fill_value=0)
    wide = wide.div(covered.set_index('domain_id')[f'{direction}_covered'], axis=0).fillna(0)
    wide = wide.rename(columns={i: f'{direction}_{i}' for i in range(355)}).reset_index()
    block = degree.merge(covered, on='domain_id', how='left')
    block = block.merge(wide, on='domain_id', how='left').fillna(0)
    return block.skb.set_name(f'{direction}_neighbor_evidence')


def direct_block(edges, trackers, direction):
    pairs = edges.merge(trackers, left_on='neighbor_id', right_on='tracking_domain_id', how='inner')
    pairs = pairs[['domain_id', 'tracker_id']].drop_duplicates()
    wide = pairs.assign(present=1).pivot_table(
        index='domain_id', columns='tracker_id', values='present', aggfunc='max', fill_value=0)
    wide = wide.reindex(columns=list(range(355)), fill_value=0)
    return wide.rename(columns={i: f'{direction}_{i}' for i in range(355)}).reset_index().skb.set_name(direction + '_indicators')


def context_prior_block(labels, source_metadata, global_prior, key):
    meta = source_metadata[['domain_id', key]]
    meta = meta[(meta[key].notna() & (meta[key] != '')).fillna(False)]
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
    result = result.merge(tld_priors, on='tld', how='left')
    result = result.merge(category_priors, on='category', how='left')
    for key in ['tld', 'category']:
        for i in range(355):
            col = f'{key}_prior_{i}'
            result = result.assign(**{col: result[col].fillna(global_prior[i])})
    return result.drop(columns=['domain_id', 'tld', 'category']).fillna(0)


def build():
    setup = build_evaluation()
    X, y = setup['X'], setup['y']
    tracking = skrub.as_data_op(ROOT + 'tracking_graph_train.parquet').skb.apply_func(
        pd.read_parquet, columns=['domain_id', 'tracker_id'])
    tracking = tracking.drop_duplicates(['domain_id', 'tracker_id'])
    counts = tracking.groupby('domain_id').size().rename('tracker_count').reset_index()
    eligible = counts[counts['tracker_count'] >= 2][['domain_id']]
    entire_locked_pool = eligible.sample(n=POOL_SIZE, random_state=42).reset_index(drop=True)
    targets = skrub.as_data_op(ROOT + 'target.tsv').skb.apply_func(pd.read_csv, sep='\t')
    relevant = entire_locked_pool.skb.concat([targets[['domain_id']]], axis=0).drop_duplicates('domain_id')
    links = skrub.as_data_op(ROOT + 'link-graph.parquet').skb.apply_func(
        pd.read_parquet, columns=['source_domain_id', 'target_domain_id'])
    incident = links[links['source_domain_id'].isin(relevant['domain_id']) |
                     links['target_domain_id'].isin(relevant['domain_id'])]
    incident = incident[incident['source_domain_id'] != incident['target_domain_id']].drop_duplicates()
    outgoing = incident[incident['source_domain_id'].isin(relevant['domain_id'])].rename(
        columns={'source_domain_id': 'domain_id', 'target_domain_id': 'neighbor_id'})
    incoming = incident[incident['target_domain_id'].isin(relevant['domain_id'])].rename(
        columns={'target_domain_id': 'domain_id', 'source_domain_id': 'neighbor_id'})
    neighbors = outgoing[['neighbor_id']].skb.concat([incoming[['neighbor_id']]], axis=0).drop_duplicates()
    safe_labels = tracking[
        ~tracking['domain_id'].isin(entire_locked_pool['domain_id']) &
        tracking['domain_id'].isin(neighbors['neighbor_id'])]
    safe_labels = safe_labels.merge(counts, on='domain_id', how='left')
    safe_labels = safe_labels.assign(weight=1.0 / safe_labels['tracker_count'])
    safe_labels = safe_labels[['domain_id', 'tracker_id', 'weight']].rename(
        columns={'domain_id': 'neighbor_id'}).skb.set_name('disjoint_neighbor_labels')
    out_features = neighborhood_block(outgoing, safe_labels, 'outgoing')
    in_features = neighborhood_block(incoming, safe_labels, 'incoming')
    trackers = skrub.as_data_op(ROOT + 'trackers.tsv').skb.apply_func(
        pd.read_csv, sep='\t', usecols=['tracking_domain_id', 'tracker_id'])
    direct_out = direct_block(outgoing, trackers, 'direct_out')
    direct_in = direct_block(incoming, trackers, 'direct_in')
    graph = X[['domain_id']].merge(out_features, on='domain_id', how='left').merge(
        in_features, on='domain_id', how='left').fillna(0)
    graph = graph.merge(direct_out, on='domain_id', how='left').merge(
        direct_in, on='domain_id', how='left').fillna(0)
    graph = graph.drop(columns=['domain_id']).skb.set_name('neighborhood_channels')
    target_direct = targets[['domain_id']].merge(direct_out, on='domain_id', how='left').merge(
        direct_in, on='domain_id', how='left').fillna(0).drop(
        columns=['domain_id']).skb.set_name('target_direct_link_audit')
    source_pool = counts[~counts['domain_id'].isin(entire_locked_pool['domain_id'])]
    source_pool = source_pool.sample(n=1000000, random_state=117).reset_index(drop=True)
    source_labels = tracking.merge(source_pool, on='domain_id', how='inner')
    source_labels = source_labels.assign(weight=1.0 / source_labels['tracker_count'])
    source_labels = source_labels[['domain_id', 'tracker_id', 'weight']].skb.set_name('disjoint_context_labels')
    lookup_ids = source_pool[['domain_id']].skb.concat([relevant[['domain_id']]], axis=0).drop_duplicates('domain_id')
    domains = skrub.as_data_op(ROOT + 'domains.parquet').skb.apply_func(
        pd.read_parquet, columns=['domain_id', 'domain'])
    domains = domains[domains['domain_id'].isin(lookup_ids['domain_id'])]
    domains = domains.assign(domain=domains['domain'].fillna('').str.lower().str.rstrip('.'))
    domains = domains.assign(tld=domains['domain'].str.rsplit('.', n=1).str[-1])
    classified = skrub.as_data_op(ROOT + 'url-classification.csv').skb.apply_func(
        pd.read_csv, usecols=['url', 'category'])
    host = classified['url'].fillna('').str.lower().str.replace(
        '^[a-z][a-z0-9+.-]*://', '', regex=True).str.split('/').str[0].str.split(
        '?').str[0].str.split('#').str[0].str.split(':').str[0].str.rstrip('.')
    classified = classified.assign(
        domain=host, category=classified['category'].fillna('').astype(str).str.lower())
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
    target_context = attach_context(
        targets, metadata, tld_priors, category_priors, global_prior).skb.set_name('target_context_audit')
    hosts = X['domain'].fillna('').str.lower().skb.set_name('hostname_normalization')
    hashed = hosts.skb.apply(HashingVectorizer(
        analyzer='char', ngram_range=(2, 5), n_features=16384,
        alternate_sign=False, norm='l2', lowercase=False, dtype=np.float32),
        no_wrap=True).skb.set_name('hostname_character_hash')
    features = skrub.as_data_op({
        'graph': graph, 'hostname': hashed, 'target_direct': target_direct,
        'context': context, 'target_context': target_context}).skb.set_name('graph_hostname_experts')
    pred = features.skb.apply(RankedGraphMixture(capacity=(600, 6)), y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}