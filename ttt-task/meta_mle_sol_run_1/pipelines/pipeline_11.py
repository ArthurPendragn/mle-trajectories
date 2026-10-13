import numpy as np
import pandas as pd
import skrub
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

CATEGORIES = [
    'Adult', 'Arts', 'Business', 'Computers', 'Games', 'Health',
    'Home', 'Kids', 'News', 'Recreation', 'Reference', 'Science',
    'Shopping', 'Society', 'Sports'
]

class ResidualTrackerNetwork(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, N_TRACKERS)
        self.nonlinear = nn.Sequential(
            nn.Linear(n_features + 64, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, N_TRACKERS)
        )
        self.embedding = nn.EmbeddingBag(
            32768, 64, mode='sum', include_last_offset=True
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def forward(self, X, indices, offsets, weights):
        lexical = self.embedding(indices, offsets, per_sample_weights=weights)
        return self.linear(X) + self.nonlinear(torch.cat([X, lexical], dim=1))

class SemanticResidualMLP(BaseEstimator):
    """Same learned residual and hostname architecture, with semantic inputs."""

    def __init__(self, epochs=8, batch_size=8192, seed=42):
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
        self.numeric_columns_ = [
            c for c in X.columns if c not in ['domain_id', 'tld', 'hostname']
        ]
        self.scaler_ = StandardScaler()
        numeric = np.nan_to_num(X[self.numeric_columns_].to_numpy(dtype=np.float32))
        numeric = self.scaler_.fit_transform(numeric).astype(np.float32)
        np.clip(numeric, -20, 20, out=numeric)
        self.encoder_ = OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, dtype=np.float32
        )
        categorical = self.encoder_.fit_transform(
            X[['tld']].fillna('unknown').astype(str)
        )
        features = np.concatenate([numeric, categorical], axis=1)
        target = np.asarray(y, dtype=np.float32)
        weight = 1.0 / np.maximum(target.sum(axis=1), 1)
        self.hasher_ = HashingVectorizer(
            analyzer='char', ngram_range=(3, 5), n_features=32768,
            lowercase=False, alternate_sign=False, norm=None, dtype=np.float32
        )
        hashed = self.hasher_.transform(X['hostname'].fillna('').astype(str))
        hashed.sort_indices()
        self.net_ = ResidualTrackerNetwork(features.shape[1]).to(self.device_)
        prior = (target * weight[:, None]).sum(axis=0) / weight.sum()
        prior = np.clip(prior, 1e-05, 1 - 1e-05)
        with torch.no_grad():
            self.net_.linear.weight.zero_()
            self.net_.linear.bias.copy_(
                torch.as_tensor(np.log(prior / (1 - prior)), device=self.device_)
            )
            self.net_.nonlinear[-1].weight.zero_()
            self.net_.nonlinear[-1].bias.zero_()
        dataset = TensorDataset(
            torch.from_numpy(features), torch.from_numpy(target),
            torch.from_numpy(weight), torch.arange(len(features), dtype=torch.int64)
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(self.seed)
        )
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=0.001)
        for epoch in range(self.epochs):
            total = 0.0
            seen = 0
            self.net_.train()
            for batch, labels, weights, row_indices in loader:
                batch = batch.to(self.device_)
                labels = labels.to(self.device_)
                weights = weights.to(self.device_)
                block = hashed[row_indices.numpy()]
                indices = torch.as_tensor(block.indices.astype(np.int64), device=self.device_)
                offsets = torch.as_tensor(block.indptr.astype(np.int64), device=self.device_)
                totals = np.asarray(block.sum(axis=1)).ravel()
                sample_weights = block.data / np.repeat(
                    np.maximum(totals, 1), np.diff(block.indptr)
                )
                sample_weights = torch.as_tensor(
                    sample_weights, dtype=torch.float32, device=self.device_
                )
                optimizer.zero_grad(set_to_none=True)
                logits = self.net_(batch, indices, offsets, sample_weights)
                loss = (
                    nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, reduction='none'
                    ).sum(dim=1) * weights
                ).mean()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
                seen += len(batch)
            print('semantic_residual_mlp', 'epoch', epoch + 1,
                  'loss', total / seen, flush=True)
        return self

    def predict(self, X):
        self.net_.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                frame = X.iloc[start:start + self.batch_size]
                numeric = np.nan_to_num(
                    frame[self.numeric_columns_].to_numpy(dtype=np.float32)
                )
                numeric = self.scaler_.transform(numeric).astype(np.float32)
                np.clip(numeric, -20, 20, out=numeric)
                categorical = self.encoder_.transform(
                    frame[['tld']].fillna('unknown').astype(str)
                )
                batch = torch.from_numpy(
                    np.concatenate([numeric, categorical], axis=1)
                ).to(self.device_)
                block = self.hasher_.transform(
                    frame['hostname'].fillna('').astype(str)
                )
                block.sort_indices()
                indices = torch.as_tensor(block.indices.astype(np.int64), device=self.device_)
                offsets = torch.as_tensor(block.indptr.astype(np.int64), device=self.device_)
                totals = np.asarray(block.sum(axis=1)).ravel()
                sample_weights = block.data / np.repeat(
                    np.maximum(totals, 1), np.diff(block.indptr)
                )
                sample_weights = torch.as_tensor(
                    sample_weights, dtype=torch.float32, device=self.device_
                )
                logits = self.net_(batch, indices, offsets, sample_weights)
                predictions.append(logits.sigmoid().cpu().numpy())
        return np.concatenate(predictions, axis=0)

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).predict(X)

def normalize_host(host):
    return host.fillna('').str.lower().str.replace('^www\\.', '', regex=True)

def category_profiles(memberships, keys, prefix, denominator=None):
    wide = memberships.pivot_table(
        index='domain_id', columns='category', values='membership_weight',
        aggfunc='sum', fill_value=0
    )
    wide = wide.reindex(
        index=keys['domain_id'], columns=CATEGORIES, fill_value=0
    ).fillna(0).astype('float32').reset_index(drop=True)
    if denominator is not None:
        wide = wide.div(denominator.replace(0, np.nan), axis=0).fillna(0)
    entropy = -(wide * wide.clip(lower=1e-12).skb.apply_func(np.log)).sum(axis=1)
    names = {c: prefix + '_category_' + c.lower() for c in CATEGORIES}
    features = keys[['domain_id']].reset_index(drop=True)
    features = features.skb.concat([wide.rename(columns=names)], axis=1)
    features = features.assign(**{
        prefix + '_entropy': entropy.to_numpy(),
        prefix + '_covered': (wide.sum(axis=1) > 0).astype('float32').to_numpy()
    })
    return features

def directional_semantics(links, relevant, membership, focal, neighbour, prefix):
    edges = links[links[focal].isin(relevant['domain_id'])]
    edges = edges[edges[focal] != edges[neighbour]]
    edges = edges.drop_duplicates([focal, neighbour])
    degree = edges.groupby(focal).size().rename('degree').reset_index()
    degree = degree.rename(columns={focal: 'domain_id'})
    classified_ids = membership[['domain_id']].drop_duplicates()
    classified_ids = classified_ids.rename(columns={'domain_id': neighbour})
    covered_edges = edges.merge(classified_ids, on=neighbour, how='inner', sort=False)
    counts = covered_edges.groupby(focal).size().rename('classified_count').reset_index()
    counts = counts.rename(columns={focal: 'domain_id'})
    statistics = relevant[['domain_id']].merge(degree, on='domain_id', how='left', sort=False)
    statistics = statistics.merge(counts, on='domain_id', how='left', sort=False)
    statistics = statistics.fillna({'degree': 0, 'classified_count': 0}).reset_index(drop=True)
    neighbour_membership = membership.rename(columns={'domain_id': neighbour})
    joined = covered_edges.merge(neighbour_membership, on=neighbour, how='inner', sort=False)
    joined = joined[[focal, 'category', 'membership_weight']]
    joined = joined.rename(columns={focal: 'domain_id'})
    profiles = category_profiles(
        joined, relevant, prefix, denominator=statistics['classified_count']
    )
    profiles = profiles.assign(**{
        prefix + '_log_degree': statistics['degree'].skb.apply_func(np.log1p).to_numpy(),
        prefix + '_log_classified_count': statistics['classified_count'].skb.apply_func(np.log1p).to_numpy(),
        prefix + '_classified_fraction': (
            statistics['classified_count'] / statistics['degree'].replace(0, np.nan)
        ).fillna(0).to_numpy()
    })
    return profiles, edges.shape, covered_edges.shape

def population_audit(features, relevant, prefix):
    report = relevant[['domain_id', 'profile_row']].merge(
        features, on='domain_id', how='left', sort=False
    )
    report = report.assign(population='training')
    report = report.assign(
        population=report['population'].mask(report['profile_row'] >= N_POOL, 'prediction')
    )
    for fold in range(N_SPLITS):
        mask = (
            (report['profile_row'] >= fold * N_VALIDATION) &
            (report['profile_row'] < (fold + 1) * N_VALIDATION)
        )
        report = report.assign(
            population=report['population'].mask(mask, 'validation_' + str(fold))
        )
    names = [prefix + '_covered', prefix + '_entropy']
    if prefix != 'direct':
        names += [
            prefix + '_log_degree', prefix + '_log_classified_count',
            prefix + '_classified_fraction'
        ]
    return {
        'coverage': report.groupby('population').agg(
            domains=('domain_id', 'size'),
            covered_domains=(prefix + '_covered', 'sum'),
            coverage=(prefix + '_covered', 'mean')
        ).reset_index(),
        'means': report.groupby('population')[names].mean().reset_index(),
        'quantiles': report.groupby('population')[names].quantile(
            [0.1, 0.5, 0.9, 0.99]
        ).reset_index()
    }

def build():
    setup = build_evaluation()
    targets = skrub.as_data_op(BASE + 'target.tsv').skb.apply_func(
        pd.read_csv, sep='\t', usecols=['domain_id']
    )
    relevant = setup['row_keys'].skb.concat([targets], axis=0)
    relevant = relevant.drop_duplicates('domain_id').reset_index(drop=True)
    relevant = relevant.reset_index().rename(columns={'index': 'profile_row'})

    domains = skrub.as_data_op(BASE + 'domains.parquet').skb.apply_func(
        pd.read_parquet, columns=['domain_id', 'domain']
    )
    classified = skrub.as_data_op(BASE + 'url-classification.csv').skb.apply_func(
        pd.read_csv, usecols=['url', 'category']
    )
    classification_host = classified['url'].fillna('').str.lower()
    classification_host = classification_host.str.replace('^https?://', '', regex=True)
    classification_host = normalize_host(classification_host.str.split('/').str[0])
    classified = classified.assign(host=classification_host)
    classified = classified[['host', 'category']].dropna(subset=['category'])
    classified = classified[classified['host'] != '']
    classified = classified.drop_duplicates(['host', 'category'])
    counts = classified.groupby('host')['category'].nunique().rename('category_count').reset_index()
    classified = classified.merge(counts, on='host', how='inner', sort=False)
    classified = classified.assign(membership_weight=1.0 / classified['category_count'])

    domain_hosts = domains.assign(host=normalize_host(domains['domain']))
    classified_domain_hosts = domain_hosts[
        domain_hosts['host'].isin(classified['host'])
    ][['domain_id', 'host']].drop_duplicates('domain_id')
    membership = classified_domain_hosts.merge(
        classified[['host', 'category', 'membership_weight']],
        on='host', how='inner', sort=False
    )
    membership = membership[['domain_id', 'category', 'membership_weight']]

    links = skrub.as_data_op(BASE + 'link-graph.parquet').skb.apply_func(
        pd.read_parquet, columns=['source_domain_id', 'target_domain_id']
    )
    links = links[
        links['source_domain_id'].isin(relevant['domain_id']) |
        links['target_domain_id'].isin(relevant['domain_id'])
    ]
    outgoing, outgoing_shape, outgoing_classified_shape = directional_semantics(
        links, relevant, membership, 'source_domain_id', 'target_domain_id', 'out'
    )
    incoming, incoming_shape, incoming_classified_shape = directional_semantics(
        links, relevant, membership, 'target_domain_id', 'source_domain_id', 'in'
    )
    direct_membership = membership[membership['domain_id'].isin(relevant['domain_id'])]
    direct = category_profiles(direct_membership, relevant, 'direct')

    lookup = domain_hosts[domain_hosts['domain_id'].isin(relevant['domain_id'])]
    lookup = lookup.drop_duplicates('domain_id')
    host = lookup['host']
    lookup = lookup.assign(
        tld=host.str.split('.').str[-1],
        hostname_length=host.str.len().astype('float32'),
        hostname_dots=host.str.count('\\.').astype('float32'),
        hostname_digits=host.str.count('[0-9]').astype('float32'),
        hostname_hyphens=host.str.count('-').astype('float32'),
        hostname='^' + host + '$'
    ).drop(columns=['domain', 'host'])
    availability = classified[['host']].drop_duplicates().assign(classified=1)
    focal_hosts = domain_hosts[domain_hosts['domain_id'].isin(relevant['domain_id'])]
    availability = focal_hosts.merge(availability, on='host', how='left', sort=False)
    availability = availability[['domain_id', 'classified']].drop_duplicates('domain_id')
    availability = availability.fillna({'classified': 0})

    X = setup['X'].merge(lookup, on='domain_id', how='left', sort=False)
    X = X.merge(availability, on='domain_id', how='left', sort=False)
    X = X.merge(direct, on='domain_id', how='left', sort=False)
    X = X.merge(outgoing, on='domain_id', how='left', sort=False)
    X = X.merge(incoming, on='domain_id', how='left', sort=False)
    prediction = X.skb.apply(SemanticResidualMLP(), y=setup['y'])

    audit = {
        'classification_shape': classified.shape,
        'category_conflicts': counts[counts['category_count'] > 1].shape,
        'category_count_distribution': counts.groupby('category_count').size().rename('hosts').reset_index(),
        'touching_link_shape': links.shape,
        'outgoing_edge_shape': outgoing_shape,
        'incoming_edge_shape': incoming_shape,
        'outgoing_classified_edge_shape': outgoing_classified_shape,
        'incoming_classified_edge_shape': incoming_classified_shape,
        'protocol': skrub.as_data_op({
            'comparison': 'Content-centric replacement, not augmentation of the relational parent.',
            'recorded_parent_mean': 0.8838668893499002,
            'recorded_parent_std': 0.0009614875864183428,
            'recorded_parent_folds': [
                0.8825166244251964, 0.8844031836219336, 0.8846808600025705
            ],
            'labels': 'Only marked y supplies supervised labels; no auxiliary tracker-label features.',
            'membership': 'Distinct category memberships receive equal fractions of each classified host mass.',
            'directional_profiles': 'Average membership vectors over distinct classified non-self neighbours; uncovered profiles are zero.',
            'model': 'CUDA residual MLP plus 32768-bucket, 64-dimensional hostname embedding.',
            'training': 'Eight epochs, inverse-cardinality BCE, AdamW lr 0.001, batch size 8192, seed 42.',
            'runtime': 'Harness complete wall time is authoritative. No manual CV, partial-fold scoring, or assumed construction sharing.',
            'timing_gate_limitation': 'The lazy pipeline API does not expose a callback after a complete harness-scored fold; no estimator-only timing is represented as complete-fold timing.',
            'interpretation': 'Does not identify incremental semantic value over the full parent or performance gains on uncovered domains.'
        })
    }
    for prefix, block in [('direct', direct), ('out', outgoing), ('in', incoming)]:
        summaries = population_audit(block, relevant, prefix)
        for name, value in summaries.items():
            audit[prefix + '_' + name] = value
    return {
        'pred': prediction,
        'scoring': setup['scoring'],
        'row_keys': setup['row_keys'],
        'audit': audit
    }