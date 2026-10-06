import numpy as np
import pandas as pd
import skrub
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer

BASE = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/'
N_TRAIN = 1200000
N_VALIDATION = 2000
N_SPLITS = 3
N_RESERVED = N_VALIDATION * N_SPLITS
N_POOL = N_TRAIN + N_RESERVED
N_TRACKERS = 355

class ReservedDomainSplits:
    """All splits share a training set; validation domains never enter training."""

    def get_n_splits(self, X=None, y=None, groups=None):
        return N_SPLITS

    def split(self, X, y=None, groups=None):
        if len(X) != N_POOL:
            raise ValueError('The locked population must contain exactly 1,206,000 rows.')
        training = np.arange(N_RESERVED, N_POOL, dtype=np.int64)
        for fold in range(N_SPLITS):
            start = fold * N_VALIDATION
            validation = np.arange(start, start + N_VALIDATION, dtype=np.int64)
            yield (training.copy(), validation)

def recall_at_10(estimator, X, y):
    """predict returns 355 ranking scores or up to ten compact tracker IDs."""
    truth = np.asarray(y)
    prediction = np.asarray(estimator.predict(X))
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)
    if prediction.ndim != 2 or prediction.shape[0] != truth.shape[0]:
        raise ValueError('Predictions must have one row per validation domain.')
    if prediction.shape[1] == N_TRACKERS:
        scores = np.nan_to_num(prediction.astype(np.float64), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        guesses = np.argsort(-scores, axis=1, kind='stable')[:, :10]
    elif prediction.shape[1] <= 10:
        guesses = prediction
    else:
        raise ValueError('Return either 355 scores or at most ten compact tracker IDs.')
    recalls = np.zeros(len(truth), dtype=np.float64)
    for i in range(len(truth)):
        selected = set()
        for value in guesses[i]:
            if pd.isna(value):
                continue
            try:
                tracker = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if tracker != value or not 0 <= tracker < N_TRACKERS:
                continue
            selected.add(tracker)
            if len(selected) == 10:
                break
        denominator = np.count_nonzero(truth[i])
        if denominator and selected:
            recalls[i] = np.count_nonzero(truth[i, list(selected)]) / denominator
    return float(recalls.mean())

def locked_setup_helper():
    labels = skrub.as_data_op(BASE + 'tracking_graph_train.parquet').skb.apply_func(pd.read_parquet, columns=['domain_id', 'tracker_id'])
    labels = labels.drop_duplicates(['domain_id', 'tracker_id'])
    counts = labels.groupby('domain_id')['tracker_id'].nunique().rename('known_tracker_count').reset_index()
    eligible = counts[counts['known_tracker_count'] >= 2]
    pool = eligible[['domain_id']].sort_values('domain_id').sample(n=N_POOL, replace=False, random_state=42).reset_index(drop=True)
    row_keys = pool[['domain_id']]
    X = pool[['domain_id']].skb.mark_as_X(cv=ReservedDomainSplits(), split_kwargs={})
    pool_labels = labels.merge(pool, on='domain_id', how='inner', sort=False)
    raw_y = pool_labels.assign(present=1).pivot_table(index='domain_id', columns='tracker_id', values='present', aggfunc='max', fill_value=0).reindex(index=pool['domain_id'], columns=list(range(N_TRACKERS)), fill_value=0).fillna(0).astype('uint8').reset_index(drop=True)
    y = raw_y.skb.mark_as_y()
    targets = skrub.as_data_op(BASE + 'target.tsv').skb.apply_func(pd.read_csv, sep='\t', usecols=['domain_id'])
    audit = {'eligible_population_shape': eligible.shape, 'labelled_population_shape': counts.shape, 'pool_shape': pool.shape, 'target_label_overlap_shape': targets.merge(counts[['domain_id']], on='domain_id', how='inner').shape, 'population_comparison_evidence': skrub.as_data_op({'source': 'exploration_e21b365199ae', 'mean_adversarial_auc': {'all': 0.581334, 'at_most_1': 0.657775, 'at_least_1': 0.581334, 'at_most_2': 0.625485, 'at_least_2': 0.5011, 'at_most_3': 0.604449, 'at_least_3': 0.562517, 'at_most_5': 0.583018, 'at_least_5': 0.700142, 'at_most_10': 0.578447, 'at_least_10': 0.88788, 'band_1_2': 0.625485, 'band_2_3': 0.538921, 'band_2_5': 0.515977, 'band_3_10': 0.556774, 'band_5_10': 0.692647, 'classified': 0.990407, 'unclassified': 0.584808, 'has_outgoing': 0.693845, 'no_outgoing': 0.88213, 'has_incoming': 0.636778, 'no_incoming': 0.945523}, 'selected_fold_auc': [0.506585, 0.500998, 0.495717], 'selected_auc_std': 0.005435, 'interpretation': 'At least two trackers has the lowest observed AUC; no less restrictive rule is within fold noise. Observable similarity does not establish label completeness or conditional-label equivalence.'}), 'pool_sizing_evidence': skrub.as_data_op({'source': 'exploration_e21b365199ae', 'model': 'CUDA character-ngram MLP, 256 hidden units, inverse-cardinality BCE', 'training_sizes': [120000, 400000, 1200000], 'mean_recall_at_10': [0.770724, 0.784351, 0.791325], 'fold_std': [0.00478, 0.007046, 0.004877], 'complete_three_split_seconds': [117.904947, 176.019659, 371.911776], 'reason': 'Use the largest measured size while preserving allowance for more expensive families. Twelve measured experiments cost about 4463 seconds. The conservative extrapolated limit is about 4.95 million training domains; diminishing measured gains motivate the smaller pool, not a proven statistical plateau.', 'auxiliary_label_policy': 'Exclude every locked-pool domain from external tracker-label sources before any supervised join, aggregation, or graph walk. External domains remain available. Fit preprocessing on training rows only. All 6000 reserved domains are excluded from every fit.', 'sampling': 'Sort eligible domain IDs, sample without replacement with seed 42, and reserve the first three consecutive 2000-row subsets. Recomputation of this recorded graph fixes membership and order; no output files are used as inputs.'})}
    return {'X': X, 'y': y, 'scoring': recall_at_10, 'row_keys': row_keys, 'audit': audit}

def locked_setup_entry():
    return locked_setup_helper()

def build_evaluation():
    return locked_setup_entry()

class ResidualTrackerNetwork(nn.Module):
    def __init__(self, n_features, hostname_signal=False):
        super().__init__()
        self.linear = nn.Linear(n_features, N_TRACKERS)
        self.nonlinear = nn.Sequential(
            nn.Linear(n_features + (64 if hostname_signal else 0), 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, N_TRACKERS))
        self.hostname_signal = hostname_signal
        if hostname_signal:
            self.embedding = nn.EmbeddingBag(32768, 64, mode='sum', include_last_offset=True)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def forward(self, X, indices=None, offsets=None, weights=None):
        if self.hostname_signal:
            lexical = self.embedding(indices, offsets, per_sample_weights=weights)
            nonlinear_input = torch.cat([X, lexical], dim=1)
        else:
            nonlinear_input = X
        return self.linear(X) + self.nonlinear(nonlinear_input)

class RelationalResidualMLP(BaseEstimator):
    def __init__(self, hostname_signal=False, epochs=8, batch_size=8192, seed=42):
        self.hostname_signal = hostname_signal
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        torch.set_num_threads(16)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.numeric_columns_ = [c for c in X.columns if c not in ['domain_id', 'tld', 'hostname']]
        self.scaler_ = StandardScaler()
        numeric = np.nan_to_num(X[self.numeric_columns_].to_numpy(dtype=np.float32))
        numeric = self.scaler_.fit_transform(numeric).astype(np.float32)
        np.clip(numeric, -20, 20, out=numeric)
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        categorical = self.encoder_.fit_transform(X[['tld']].fillna('unknown').astype(str))
        features = np.concatenate([numeric, categorical], axis=1)
        target = np.asarray(y, dtype=np.float32)
        weight = 1.0 / np.maximum(target.sum(axis=1), 1)
        if self.hostname_signal:
            self.hasher_ = HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=32768, lowercase=False, alternate_sign=False, norm=None, dtype=np.float32)
            hashed = self.hasher_.transform(X['hostname'].fillna('').astype(str))
            hashed.sort_indices()
            print('hostname_hash_training', 'rows', hashed.shape[0], 'mean_occupied_buckets', hashed.nnz / max(hashed.shape[0], 1), 'nonempty_fraction', float(np.mean(np.diff(hashed.indptr) > 0)), 'active_buckets', len(np.unique(hashed.indices)), flush=True)
        else:
            hashed = None
        self.net_ = ResidualTrackerNetwork(features.shape[1], hostname_signal=self.hostname_signal).to(self.device_)
        prior = (target * weight[:, None]).sum(axis=0) / weight.sum()
        prior = np.clip(prior, 1e-05, 1 - 1e-05)
        with torch.no_grad():
            self.net_.linear.weight.zero_()
            self.net_.linear.bias.copy_(torch.as_tensor(np.log(prior / (1 - prior)), device=self.device_))
            self.net_.nonlinear[-1].weight.zero_()
            self.net_.nonlinear[-1].bias.zero_()
        dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(target), torch.from_numpy(weight), torch.arange(len(features), dtype=torch.int64))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(self.seed))
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=0.001)
        for epoch in range(self.epochs):
            total = 0.0
            seen = 0
            self.net_.train()
            for batch, labels, weights, row_indices in loader:
                batch = batch.to(self.device_)
                labels = labels.to(self.device_)
                weights = weights.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                if self.hostname_signal:
                    block = hashed[row_indices.numpy()]
                    indices = torch.as_tensor(block.indices.astype(np.int64), device=self.device_)
                    offsets = torch.as_tensor(block.indptr.astype(np.int64), device=self.device_)
                    totals = np.asarray(block.sum(axis=1)).ravel()
                    sample_weights = block.data / np.repeat(np.maximum(totals, 1), np.diff(block.indptr))
                    sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device_)
                    logits = self.net_(batch, indices, offsets, sample_weights)
                else:
                    logits = self.net_(batch)
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none').sum(dim=1) * weights).mean()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
                seen += len(batch)
            print('relational_residual_mlp', 'hostname_signal', self.hostname_signal, 'epoch', epoch + 1, 'loss', total / seen, flush=True)
        return self

    def predict(self, X):
        self.net_.eval()
        predictions = []
        hash_nonempty = 0
        hash_nnz = 0
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                frame = X.iloc[start:start + self.batch_size]
                numeric = np.nan_to_num(frame[self.numeric_columns_].to_numpy(dtype=np.float32))
                numeric = self.scaler_.transform(numeric).astype(np.float32)
                np.clip(numeric, -20, 20, out=numeric)
                categorical = self.encoder_.transform(frame[['tld']].fillna('unknown').astype(str))
                batch = torch.from_numpy(np.concatenate([numeric, categorical], axis=1)).to(self.device_)
                if self.hostname_signal:
                    block = self.hasher_.transform(frame['hostname'].fillna('').astype(str))
                    block.sort_indices()
                    hash_nonempty += int(np.count_nonzero(np.diff(block.indptr)))
                    hash_nnz += block.nnz
                    indices = torch.as_tensor(block.indices.astype(np.int64), device=self.device_)
                    offsets = torch.as_tensor(block.indptr.astype(np.int64), device=self.device_)
                    totals = np.asarray(block.sum(axis=1)).ravel()
                    sample_weights = block.data / np.repeat(np.maximum(totals, 1), np.diff(block.indptr))
                    sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device_)
                    logits = self.net_(batch, indices, offsets, sample_weights)
                else:
                    logits = self.net_(batch)
                predictions.append(logits.sigmoid().cpu().numpy())
        if self.hostname_signal:
            print('hostname_hash_prediction', 'rows', len(X), 'mean_occupied_buckets', hash_nnz / max(len(X), 1), 'nonempty_fraction', hash_nonempty / max(len(X), 1), flush=True)
        return np.concatenate(predictions, axis=0)

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).predict(X)

def sparse_matrix(values, rows, columns, shape):
    specification = skrub.as_data_op((values, (rows, columns)))
    return specification.skb.apply_func(sp.csr_matrix, shape=shape, dtype=np.float32)

def mix_uint64(value):
    # Build-time helper emitting library-operation nodes, not a deferred UDF.
    value = value.skb.apply_func(np.add, np.uint64(0x9E3779B97F4A7C15), dtype='uint64')
    shifted = value.skb.apply_func(np.right_shift, np.uint64(30))
    value = value.skb.apply_func(np.bitwise_xor, shifted)
    value = value.skb.apply_func(np.multiply, np.uint64(0xBF58476D1CE4E5B9), dtype='uint64')
    shifted = value.skb.apply_func(np.right_shift, np.uint64(27))
    value = value.skb.apply_func(np.bitwise_xor, shifted)
    value = value.skb.apply_func(np.multiply, np.uint64(0x94D049BB133111EB), dtype='uint64')
    shifted = value.skb.apply_func(np.right_shift, np.uint64(31))
    return value.skb.apply_func(np.bitwise_xor, shifted)

def bottom_hash(frame, group, neighbour, cap, seed):
    group_ids = frame[group].astype('uint64').to_numpy()
    neighbour_ids = frame[neighbour].astype('uint64').to_numpy()
    group_seed = group_ids.skb.apply_func(np.bitwise_xor, np.uint64(seed))
    group_hash = mix_uint64(group_seed)
    combined = neighbour_ids.skb.apply_func(np.bitwise_xor, group_hash)
    edge_hash = mix_uint64(combined)
    ranked = frame.assign(selection_hash=edge_hash)
    # Global hash ordering gives the same bottom-k within every group.
    ranked = ranked.sort_values(['selection_hash', neighbour], kind='mergesort')
    selected = ranked.groupby(group, sort=False).head(cap)
    # Restore a canonical order independent of parquet order.
    return selected.drop(columns=['selection_hash']).sort_values([group, neighbour]).reset_index(drop=True)

def direction_features(edges, relevant, auxiliary, n_domains, source, neighbour, prefix, prior):
    selected = edges[edges[source].isin(relevant['domain_id'])]
    selected = selected[selected[source] != selected[neighbour]]
    selected = selected.drop_duplicates([source, neighbour])
    row_lookup = relevant[['domain_id', 'profile_row']].rename(columns={'domain_id': source})
    selected = selected.merge(row_lookup, on=source, how='inner', sort=False)
    row_numbers = selected['profile_row'].to_numpy()
    neighbour_numbers = selected[neighbour].to_numpy()
    values = selected[source].astype('float32') * 0 + 1
    adjacency = sparse_matrix(values.to_numpy(), row_numbers, neighbour_numbers, (relevant.shape[0], n_domains))
    degree = adjacency.sum(axis=1).skb.apply_func(np.asarray).ravel()
    sums = adjacency.dot(auxiliary)
    count = sums.sum(axis=1).skb.apply_func(np.asarray).ravel()
    names = [prefix + '_tracker_' + str(i) for i in range(N_TRACKERS)]
    profiles = sums.toarray().skb.apply_func(pd.DataFrame, columns=list(range(N_TRACKERS)))
    denominator = count.skb.apply_func(pd.Series).replace(0, np.nan)
    profiles = profiles.div(denominator, axis=0).fillna(prior)
    profiles = profiles.rename(columns=dict(enumerate(names)))
    table = relevant[['domain_id']].assign(**{prefix + '_degree': degree, prefix + '_labelled_count': count, prefix + '_coverage': count / (degree + 1e-06)})
    table = table.skb.concat([profiles], axis=1)
    for name in [prefix + '_degree', prefix + '_labelled_count']:
        table = table.assign(**{name: table[name].skb.apply_func(np.log1p)})
    return table

def bounded_shared_profiles(links, relevant, external_domains, auxiliary, n_domains, focal_column, hub_column, prefix):
    first = links[links[focal_column].isin(relevant['domain_id'])]
    first = first[first[focal_column] != first[hub_column]]
    first = first.drop_duplicates([focal_column, hub_column])
    first_before = first.groupby(focal_column).size().rename('first_before').reset_index().rename(columns={focal_column: 'domain_id'})
    first = bottom_hash(first, focal_column, hub_column, 16, 42)
    first_retained = first.groupby(focal_column).size().rename('first_retained').reset_index().rename(columns={focal_column: 'domain_id'})
    hubs = first[[hub_column]].drop_duplicates().reset_index(drop=True)
    hubs = hubs.reset_index().rename(columns={'index': 'hub_row'})
    terminal = links[links[hub_column].isin(hubs[hub_column])]
    terminal = terminal[terminal[focal_column].isin(external_domains['domain_id'])]
    terminal = terminal[terminal[focal_column] != terminal[hub_column]]
    terminal = terminal.drop_duplicates([hub_column, focal_column])
    terminal_before = terminal.groupby(hub_column).size().rename('terminal_before').reset_index()
    terminal = bottom_hash(terminal, hub_column, focal_column, 64, 43)
    terminal_retained = terminal.groupby(hub_column).size().rename('terminal_retained').reset_index()
    terminal = terminal.merge(hubs, on=hub_column, how='inner', sort=False)
    terminal = terminal.merge(terminal_retained, on=hub_column, how='inner', sort=False)
    terminal = terminal.assign(terminal_weight=1.0 / terminal['terminal_retained'])
    hub_terminal = sparse_matrix(terminal['terminal_weight'].astype('float32').to_numpy(), terminal['hub_row'].to_numpy(), terminal[focal_column].to_numpy(), (hubs.shape[0], n_domains))
    hub_profiles = hub_terminal.dot(auxiliary)
    covered_hubs = hubs.merge(terminal_retained, on=hub_column, how='inner', sort=False)
    first_covered = first.merge(covered_hubs[[hub_column, 'hub_row']], on=hub_column, how='inner', sort=False)
    focal_lookup = relevant[['domain_id', 'profile_row']].rename(columns={'domain_id': focal_column})
    first_covered = first_covered.merge(focal_lookup, on=focal_column, how='inner', sort=False)
    covered_counts = first_covered.groupby(focal_column).size().rename('covered_hubs').reset_index()
    first_covered = first_covered.merge(covered_counts, on=focal_column, how='inner', sort=False)
    first_covered = first_covered.assign(hub_weight=1.0 / first_covered['covered_hubs'])
    focal_hub = sparse_matrix(first_covered['hub_weight'].astype('float32').to_numpy(), first_covered['profile_row'].to_numpy(), first_covered['hub_row'].to_numpy(), (relevant.shape[0], hubs.shape[0]))
    profile = focal_hub.dot(hub_profiles).toarray()
    names = [prefix + '_tracker_' + str(i) for i in range(N_TRACKERS)]
    profile_frame = profile.skb.apply_func(pd.DataFrame, columns=names)
    counts = covered_counts.rename(columns={focal_column: 'domain_id'})
    statistics = relevant[['domain_id', 'profile_row']].merge(counts, on='domain_id', how='left', sort=False).fillna({'covered_hubs': 0})
    statistics = statistics.merge(first_before, on='domain_id', how='left', sort=False)
    statistics = statistics.merge(first_retained, on='domain_id', how='left', sort=False)
    statistics = statistics.fillna({'first_before': 0, 'first_retained': 0})
    statistics = statistics.assign(covered=(statistics['covered_hubs'] > 0).astype('float32'), first_truncated=(statistics['first_before'] > 16).astype('float32'), profile_mass=profile.sum(axis=1), profile_max=profile.max(axis=1), profile_nonzero=(profile > 0).sum(axis=1))
    statistics = statistics.assign(population='training')
    statistics = statistics.assign(population=statistics['population'].mask(statistics['profile_row'] >= N_POOL, 'prediction'))
    for fold in range(N_SPLITS):
        mask = (statistics['profile_row'] >= fold * N_VALIDATION) & (statistics['profile_row'] < (fold + 1) * N_VALIDATION)
        statistics = statistics.assign(population=statistics['population'].mask(mask, 'validation_' + str(fold)))
    features = relevant[['domain_id']].assign(**{prefix + '_covered': statistics['covered'].to_numpy(), prefix + '_log_covered_hubs': statistics['covered_hubs'].skb.apply_func(np.log1p).to_numpy()})
    features = features.skb.concat([profile_frame], axis=1)
    intermediary_audit = hubs.merge(terminal_before, on=hub_column, how='left', sort=False)
    intermediary_audit = intermediary_audit.merge(terminal_retained, on=hub_column, how='left', sort=False)
    intermediary_audit = intermediary_audit.fillna({'terminal_before': 0, 'terminal_retained': 0})
    intermediary_audit = intermediary_audit.assign(terminal_truncated=(intermediary_audit['terminal_before'] > 64).astype('float32'))
    terminal_ids = terminal[[focal_column]].drop_duplicates().rename(columns={focal_column: 'domain_id'})
    construction = skrub.as_data_op({'block': prefix, 'retained_first_edges': first.shape[0], 'intermediaries': hubs.shape[0], 'retained_terminal_edges': terminal.shape[0], 'terminals_used': terminal_ids.shape[0]})
    return features, statistics, intermediary_audit, terminal_ids, construction

def standalone_audit(statistics, features, prefix, truth):
    columns = [prefix + '_tracker_' + str(i) for i in range(N_TRACKERS)]
    scores = features.iloc[:N_RESERVED][columns].to_numpy()
    guesses = (-scores).skb.apply_func(np.argsort, axis=1, kind='stable')[:, :10]
    hits = truth.skb.apply_func(np.take_along_axis, guesses, axis=1).sum(axis=1)
    cardinality = truth.sum(axis=1)
    report = statistics.iloc[:N_RESERVED].assign(known_trackers=cardinality, recall=hits / cardinality, near_perfect=(hits == cardinality).astype('float32'))
    covered = report[report['covered'] > 0]
    return {
        'coverage': report.groupby('population').agg(domains=('domain_id', 'size'), covered_domains=('covered', 'sum'), coverage=('covered', 'mean')).reset_index(),
        'covered_rankings': covered.groupby('population').agg(covered_domains=('domain_id', 'size'), mean_known_trackers=('known_trackers', 'mean'), mean_recall=('recall', 'mean'), exact_recovery_fraction=('near_perfect', 'mean')).reset_index(),
        'preview': report.head(12)
    }

def build():
    setup = build_evaluation()
    targets = skrub.as_data_op(BASE + 'target.tsv').skb.apply_func(pd.read_csv, sep='\t', usecols=['domain_id'])
    relevant = setup['row_keys'].skb.concat([targets], axis=0)
    relevant = relevant.drop_duplicates('domain_id').reset_index(drop=True)
    relevant = relevant.reset_index().rename(columns={'index': 'profile_row'})
    domains = skrub.as_data_op(BASE + 'domains.parquet').skb.apply_func(pd.read_parquet, columns=['domain_id', 'domain'])
    n_domains = domains['domain_id'].max() + 1
    labels = skrub.as_data_op(BASE + 'tracking_graph_train.parquet').skb.apply_func(pd.read_parquet, columns=['domain_id', 'tracker_id'])
    external = labels[~labels['domain_id'].isin(setup['row_keys']['domain_id'])]
    external = external.drop_duplicates(['domain_id', 'tracker_id'])
    cardinality = external.groupby('domain_id')['tracker_id'].nunique().rename('aux_cardinality').reset_index()
    external = external.merge(cardinality, on='domain_id', how='inner', sort=False)
    external = external.assign(aux_weight=1.0 / external['aux_cardinality'])
    auxiliary = sparse_matrix(external['aux_weight'].astype('float32').to_numpy(), external['domain_id'].to_numpy(), external['tracker_id'].to_numpy(), (n_domains, N_TRACKERS))
    prior = external.groupby('tracker_id')['aux_weight'].sum()
    prior = prior.reindex(list(range(N_TRACKERS)), fill_value=0) / cardinality.shape[0]
    all_links = skrub.as_data_op(BASE + 'link-graph.parquet').skb.apply_func(pd.read_parquet, columns=['source_domain_id', 'target_domain_id'])
    links = all_links[all_links['source_domain_id'].isin(relevant['domain_id']) | all_links['target_domain_id'].isin(relevant['domain_id'])]
    outgoing = direction_features(links, relevant, auxiliary, n_domains, 'source_domain_id', 'target_domain_id', 'out', prior)
    incoming = direction_features(links, relevant, auxiliary, n_domains, 'target_domain_id', 'source_domain_id', 'in', prior)
    destination, destination_stats, destination_hubs, destination_terminals, destination_build = bounded_shared_profiles(all_links, relevant, cardinality[['domain_id']], auxiliary, n_domains, 'source_domain_id', 'target_domain_id', 'shared_destination')
    source, source_stats, source_hubs, source_terminals, source_build = bounded_shared_profiles(all_links, relevant, cardinality[['domain_id']], auxiliary, n_domains, 'target_domain_id', 'source_domain_id', 'shared_source')
    lookup = domains[domains['domain_id'].isin(relevant['domain_id'])]
    lookup = lookup.drop_duplicates('domain_id')
    host = lookup['domain'].fillna('').str.lower().str.replace('^www\\.', '', regex=True)
    lookup = lookup.assign(tld=host.str.split('.').str[-1], hostname_length=host.str.len().astype('float32'), hostname_dots=host.str.count('\\.').astype('float32'), hostname_digits=host.str.count('[0-9]').astype('float32'), hostname_hyphens=host.str.count('-').astype('float32'), hostname='^' + host + '$').drop(columns=['domain'])
    classified = skrub.as_data_op(BASE + 'url-classification.csv').skb.apply_func(pd.read_csv, usecols=['url'])
    classification_host = classified['url'].fillna('').str.lower()
    classification_host = classification_host.str.replace('^https?://', '', regex=True)
    classification_host = classification_host.str.split('/').str[0].str.replace('^www\\.', '', regex=True)
    classified = classified.assign(host=classification_host, classified=1)
    classified = classified[['host', 'classified']].drop_duplicates('host')
    host_lookup = domains[domains['domain_id'].isin(relevant['domain_id'])]
    host_lookup = host_lookup.assign(host=host_lookup['domain'].fillna('').str.lower().str.replace('^www\\.', '', regex=True))
    coverage = host_lookup.merge(classified, on='host', how='left', sort=False)
    coverage = coverage[['domain_id', 'classified']].drop_duplicates('domain_id')
    coverage = coverage.fillna({'classified': 0})
    X = setup['X'].merge(lookup, on='domain_id', how='left', sort=False)
    X = X.merge(coverage, on='domain_id', how='left', sort=False)
    X = X.merge(outgoing, on='domain_id', how='left', sort=False)
    X = X.merge(incoming, on='domain_id', how='left', sort=False)
    X = X.merge(destination, on='domain_id', how='left', sort=False)
    X = X.merge(source, on='domain_id', how='left', sort=False)
    base_columns = ['domain_id', 'tld', 'hostname_length', 'hostname_dots', 'hostname_digits', 'hostname_hyphens', 'hostname', 'classified']
    for prefix in ['out', 'in']:
        base_columns += [prefix + '_degree', prefix + '_labelled_count', prefix + '_coverage']
        base_columns += [prefix + '_tracker_' + str(i) for i in range(N_TRACKERS)]
    destination_columns = ['shared_destination_covered', 'shared_destination_log_covered_hubs']
    destination_columns += ['shared_destination_tracker_' + str(i) for i in range(N_TRACKERS)]
    source_columns = ['shared_source_covered', 'shared_source_log_covered_hubs']
    source_columns += ['shared_source_tracker_' + str(i) for i in range(N_TRACKERS)]
    X = X[base_columns + destination_columns + source_columns]
    prediction = X.skb.apply(RelationalResidualMLP(hostname_signal=True), y=setup['y'])

    validation_keys = setup['row_keys'].iloc[:N_RESERVED]
    validation_labels = labels.merge(validation_keys, on='domain_id', how='inner', sort=False)
    validation_truth = validation_labels.assign(present=1).pivot_table(index='domain_id', columns='tracker_id', values='present', aggfunc='max', fill_value=0).reindex(index=validation_keys['domain_id'], columns=list(range(N_TRACKERS)), fill_value=0).fillna(0).astype('uint8').to_numpy()
    audit = {}
    summary_columns = ['covered', 'covered_hubs', 'profile_mass', 'profile_max', 'profile_nonzero', 'first_before', 'first_retained', 'first_truncated']
    for prefix, features, stats, hubs, terminals, construction in [
        ('shared_destination', destination, destination_stats, destination_hubs, destination_terminals, destination_build),
        ('shared_source', source, source_stats, source_hubs, source_terminals, source_build)
    ]:
        audit[prefix + '_construction'] = construction
        audit[prefix + '_population_means'] = stats.groupby('population')[summary_columns].mean().reset_index()
        audit[prefix + '_population_counts'] = stats.groupby('population').size().rename('domains').reset_index()
        audit[prefix + '_population_quantiles'] = stats.groupby('population')[['covered_hubs', 'profile_mass', 'profile_max', 'profile_nonzero']].quantile([0.1, 0.5, 0.9, 0.99]).reset_index()
        audit[prefix + '_intermediary_bounds'] = hubs[['terminal_before', 'terminal_retained', 'terminal_truncated']].agg(['count', 'mean', 'sum', 'max']).reset_index()
        audit[prefix + '_terminal_pool_overlap'] = terminals.merge(setup['row_keys'], on='domain_id', how='inner').shape
        for name, value in standalone_audit(stats, features, prefix, validation_truth).items():
            audit[prefix + '_' + name] = value
    audit['auxiliary_pool_overlap'] = cardinality[['domain_id']].merge(setup['row_keys'], on='domain_id', how='inner').shape

    # Diagnostic rows come from an authorized source, not synthetic materialized data.
    diagnostic = all_links.head(10000).drop_duplicates(['source_domain_id', 'target_domain_id'])
    diagnostic = diagnostic[diagnostic['source_domain_id'] != diagnostic['target_domain_id']]
    permuted = diagnostic.sample(frac=1, random_state=617).reset_index(drop=True)
    for name, group, neighbour, cap, seed in [
        ('first', 'source_domain_id', 'target_domain_id', 16, 42),
        ('terminal', 'target_domain_id', 'source_domain_id', 64, 43)
    ]:
        original_selection = bottom_hash(diagnostic, group, neighbour, cap, seed)
        permuted_selection = bottom_hash(permuted, group, neighbour, cap, seed)
        audit[name + '_permutation_invariant'] = original_selection.equals(permuted_selection)
        audit[name + '_diagnostic_selection_shape'] = original_selection.shape
        audit[name + '_diagnostic_max_group_size'] = original_selection.groupby(group).size().max()

    audit['protocol'] = skrub.as_data_op({
        'selection': 'Fixed seeded uint64 SplitMix64 mixing of group and neighbour IDs; retain bottom hashes, breaking ties by neighbour ID. Caps 16 first-hop intermediaries and 64 external terminals. Canonical group/neighbour output order.',
        'implementation': 'Fine-grained graph nodes calling NumPy integer ufuncs; no deferred dataframe UDF and no dtype type objects in dataframe expressions.',
        'normalization': 'Unchanged inverse-cardinality terminal vectors, equal terminal averaging per hub, equal covered-hub averaging per focal domain.',
        'label_safety': 'Exclude every locked-pool domain before auxiliary construction and terminal filtering. No focal-return label exists in the auxiliary matrix.',
        'training': 'Unchanged hostname branch, residual architecture, inverse-cardinality BCE, eight epochs, batch size 8192, seed 42, AdamW lr 0.001.',
        'parent_fold_scores': [0.8825166244251964, 0.8844031836219336, 0.8846808600025705],
        'parent_mean': 0.8838668893499002,
        'parent_std': 0.0009614875864183428,
        'review': 'Compare harness fold scores pairwise with parent. Large gains or near-perfect standalone covered rankings require leakage review; audit outputs never enter features.',
        'planning_seconds': [4000, 6000]
    })
    return {'pred': prediction, 'scoring': setup['scoring'], 'row_keys': setup['row_keys'], 'audit': audit}