import numpy as np
import pandas as pd
import skrub
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


import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC_COLUMNS = [
    'hostname_length', 'hostname_dots', 'hostname_digits', 'hostname_hyphens',
    'classified', 'press_missing', 'freedom_of_the_press',
    'out_degree_log', 'in_degree_log',
    'out_classified_fraction', 'in_classified_fraction',
]
CATEGORICAL_COLUMNS = ['tld', 'category']


class SparseHostnameNetwork(nn.Module):
    def __init__(self, text_width, metadata_width):
        super().__init__()
        self.text_weight = nn.Parameter(torch.empty(text_width, 256))
        self.text_bias = nn.Parameter(torch.zeros(256))
        nn.init.normal_(self.text_weight, mean=0.0, std=0.025)
        self.hidden = nn.Linear(256 + metadata_width, 256)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(256, N_TRACKERS)

    def forward(self, text, metadata):
        projected = torch.relu(torch.sparse.mm(text, self.text_weight) + self.text_bias)
        combined = torch.cat([projected, metadata], dim=1)
        hidden = self.dropout(torch.relu(self.hidden(combined)))
        return self.output(hidden)


class HostnameRankingMLP(BaseEstimator):
    def __init__(self, epochs=8, batch_size=4096, learning_rate=0.001, seed=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def _numeric(self, X):
        return X[NUMERIC_COLUMNS].to_numpy(dtype=np.float32, na_value=np.nan)

    def _categorical(self, X):
        return X[CATEGORICAL_COLUMNS].fillna('__missing__').astype(str)

    def _sparse_tensor(self, matrix):
        coo = matrix.tocoo()
        indices = np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
        return torch.sparse_coo_tensor(
            torch.from_numpy(indices),
            torch.from_numpy(coo.data.astype(np.float32, copy=False)),
            size=coo.shape,
            device=self.device_,
        ).coalesce()

    def _features(self, X, fitting):
        hosts = X['host'].fillna('').astype(str)
        categorical = self._categorical(X)
        numeric = self._numeric(X)
        if fitting:
            self.vectorizer_ = TfidfVectorizer(
                analyzer='char', ngram_range=(3, 5), max_features=16384,
                min_df=2, sublinear_tf=True, dtype=np.float32,
            )
            text = self.vectorizer_.fit_transform(hosts).tocsr()
            self.encoder_ = OneHotEncoder(
                sparse_output=False, handle_unknown='ignore', dtype=np.float32,
            )
            cat = self.encoder_.fit_transform(categorical)
            self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
            num = self.imputer_.fit_transform(numeric)
            self.scaler_ = StandardScaler()
            num = self.scaler_.fit_transform(num)
        else:
            text = self.vectorizer_.transform(hosts).tocsr()
            cat = self.encoder_.transform(categorical)
            num = self.scaler_.transform(self.imputer_.transform(numeric))
        metadata = np.concatenate([cat, num], axis=1).astype(np.float32, copy=False)
        return text, metadata

    def fit(self, X, y):
        started = time.perf_counter()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.set_num_threads(16)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        text, metadata = self._features(X, fitting=True)
        truth = np.asarray(y, dtype=np.float32)
        cardinality = truth.sum(axis=1)
        weights = 1.0 / np.maximum(cardinality, 1.0)
        weights /= weights.mean()
        self.net_ = SparseHostnameNetwork(text.shape[1], metadata.shape[1]).to(self.device_)
        weighted_prior = np.average(truth, axis=0, weights=weights)
        weighted_prior = np.clip(weighted_prior, 0.0001, 0.9999)
        with torch.no_grad():
            self.net_.output.bias.copy_(
                torch.as_tensor(np.log(weighted_prior / (1.0 - weighted_prior)),
                                dtype=torch.float32, device=self.device_)
            )
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.learning_rate, weight_decay=0.01,
        )
        dataset = TensorDataset(torch.arange(len(X), dtype=torch.int64))
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=generator, num_workers=0,
        )
        metadata_tensor = torch.from_numpy(metadata)
        truth_tensor = torch.from_numpy(truth)
        weight_tensor = torch.from_numpy(weights.astype(np.float32))
        print('MLP features', text.shape, metadata.shape, 'device', self.device_,
              'feature_seconds', time.perf_counter() - started, flush=True)
        self.net_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (indices,) in loader:
                index = indices.numpy()
                sparse_batch = self._sparse_tensor(text[index])
                meta_batch = metadata_tensor[indices].to(self.device_)
                target_batch = truth_tensor[indices].to(self.device_)
                sample_weights = weight_tensor[indices].to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.net_(sparse_batch, meta_batch)
                per_row = nn.functional.binary_cross_entropy_with_logits(
                    logits, target_batch, reduction='none',
                ).mean(dim=1)
                loss = (per_row * sample_weights).mean()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(index)
            print('MLP epoch', epoch + 1, 'weighted_BCE', total_loss / len(X),
                  'elapsed_seconds', time.perf_counter() - started, flush=True)
        self.net_.eval()
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        text, metadata = self._features(X, fitting=False)
        predictions = []
        self.net_.eval()
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                stop = min(start + self.batch_size, len(X))
                sparse_batch = self._sparse_tensor(text[start:stop])
                meta_batch = torch.from_numpy(metadata[start:stop]).to(self.device_)
                predictions.append(self.net_(sparse_batch, meta_batch).cpu().numpy())
        if not predictions:
            return np.empty((0, N_TRACKERS), dtype=np.float32)
        return np.concatenate(predictions, axis=0)


def metadata_tables(relevant):
    domains = skrub.as_data_op(BASE + 'domains.parquet').skb.apply_func(
        pd.read_parquet, columns=['domain_id', 'domain'],
    )
    classification = skrub.as_data_op(BASE + 'url-classification.csv').skb.apply_func(
        pd.read_csv, usecols=['url', 'category'],
    )
    classification = classification.assign(
        host=classification['url'].fillna('').str.lower()
        .str.replace(r'^[a-z][a-z0-9+.-]*://', '', regex=True)
        .str.split('/').str[0].str.split('?').str[0]
        .str.replace(r'^www\.', '', regex=True).str.rstrip('.'),
    )
    classification = classification[['host', 'category']].drop_duplicates('host')
    relevant_domains = domains[domains['domain_id'].isin(relevant['domain_id'])]
    relevant_domains = relevant_domains.assign(
        host=relevant_domains['domain'].fillna('').str.lower()
        .str.replace(r'^www\.', '', regex=True).str.rstrip('.'),
    )
    relevant_domains = relevant_domains.merge(
        classification, on='host', how='left', sort=False,
    )
    relevant_domains = relevant_domains.assign(
        tld=relevant_domains['host'].str.split('.').str[-1],
        hostname_length=relevant_domains['host'].str.len(),
        hostname_dots=relevant_domains['host'].str.count(r'\.'),
        hostname_digits=relevant_domains['host'].str.count(r'[0-9]'),
        hostname_hyphens=relevant_domains['host'].str.count('-'),
        classified=relevant_domains['category'].notna().astype('float32'),
    )
    press = skrub.as_data_op(BASE + 'freedom-of-the-press.csv').skb.apply_func(
        pd.read_csv, sep=None, engine='python',
        usecols=['tld', 'freedom_of_the_press'],
    )
    press = press.assign(tld=press['tld'].str.lower().str.lstrip('.')).drop_duplicates('tld')
    relevant_domains = relevant_domains.merge(press, on='tld', how='left', sort=False)
    relevant_domains = relevant_domains.assign(
        press_missing=relevant_domains['freedom_of_the_press'].isna().astype('float32'),
        category=relevant_domains['category'].fillna('__missing__'),
    )
    neighbour_domains = domains.assign(
        host=domains['domain'].fillna('').str.lower()
        .str.replace(r'^www\.', '', regex=True).str.rstrip('.'),
    )
    neighbour_domains = neighbour_domains[
        neighbour_domains['host'].isin(classification['host'])
    ]
    classified_ids = neighbour_domains[['domain_id']].drop_duplicates('domain_id')
    classified_ids = classified_ids.assign(neighbour_classified=1.0)
    return relevant_domains, classified_ids


def graph_observables(relevant, classified_ids):
    links = skrub.as_data_op(BASE + 'link-graph.parquet').skb.apply_func(
        pd.read_parquet, columns=['source_domain_id', 'target_domain_id'],
    )
    outgoing = links[links['source_domain_id'].isin(relevant['domain_id'])]
    incoming = links[links['target_domain_id'].isin(relevant['domain_id'])]
    outgoing = outgoing.merge(
        classified_ids.rename(columns={'domain_id': 'target_domain_id'}),
        on='target_domain_id', how='left', sort=False,
    )
    incoming = incoming.merge(
        classified_ids.rename(columns={'domain_id': 'source_domain_id'}),
        on='source_domain_id', how='left', sort=False,
    )
    outgoing = outgoing.assign(
        neighbour_classified=outgoing['neighbour_classified'].fillna(0.0),
    )
    incoming = incoming.assign(
        neighbour_classified=incoming['neighbour_classified'].fillna(0.0),
    )
    out_degree = outgoing.groupby('source_domain_id').size().rename('out_degree').reset_index()
    out_classified = outgoing.groupby('source_domain_id')['neighbour_classified'].sum().rename('out_classified').reset_index()
    out_stats = out_degree.merge(out_classified, on='source_domain_id', how='left', sort=False)
    out_stats = out_stats.assign(
        out_classified_fraction=out_stats['out_classified'] / out_stats['out_degree'],
        out_degree_log=out_stats['out_degree'].skb.apply_func(np.log1p),
    ).rename(columns={'source_domain_id': 'domain_id'})
    in_degree = incoming.groupby('target_domain_id').size().rename('in_degree').reset_index()
    in_classified = incoming.groupby('target_domain_id')['neighbour_classified'].sum().rename('in_classified').reset_index()
    in_stats = in_degree.merge(in_classified, on='target_domain_id', how='left', sort=False)
    in_stats = in_stats.assign(
        in_classified_fraction=in_stats['in_classified'] / in_stats['in_degree'],
        in_degree_log=in_stats['in_degree'].skb.apply_func(np.log1p),
    ).rename(columns={'target_domain_id': 'domain_id'})
    return (
        out_stats[['domain_id', 'out_degree_log', 'out_classified_fraction']],
        in_stats[['domain_id', 'in_degree_log', 'in_classified_fraction']],
    )


def build():
    setup = build_evaluation()
    targets = skrub.as_data_op(BASE + 'target.tsv').skb.apply_func(
        pd.read_csv, sep='\t', usecols=['domain_id'],
    )
    relevant = setup['row_keys'].skb.concat([targets], axis=0).drop_duplicates('domain_id')
    domain_metadata, classified_ids = metadata_tables(relevant)
    out_stats, in_stats = graph_observables(relevant, classified_ids)
    features = setup['X'].merge(
        domain_metadata.drop(columns=['domain']), on='domain_id', how='left', sort=False,
    )
    features = features.merge(out_stats, on='domain_id', how='left', sort=False)
    features = features.merge(in_stats, on='domain_id', how='left', sort=False)
    features = features.assign(
        out_degree_log=features['out_degree_log'].fillna(0.0),
        in_degree_log=features['in_degree_log'].fillna(0.0),
        out_classified_fraction=features['out_classified_fraction'].fillna(0.0),
        in_classified_fraction=features['in_classified_fraction'].fillna(0.0),
    )
    features = features[['host'] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    pred = features.skb.apply(
        HostnameRankingMLP(epochs=8, batch_size=4096, learning_rate=0.001, seed=42),
        y=setup['y'],
    )
    return {'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}