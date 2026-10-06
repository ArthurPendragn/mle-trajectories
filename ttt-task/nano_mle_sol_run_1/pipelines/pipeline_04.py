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
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, N_TRACKERS),
        )
        self.hostname_signal = hostname_signal
        if hostname_signal:
            self.embedding = nn.EmbeddingBag(
                32768, 64, mode='sum', include_last_offset=True
            )
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def forward(self, X, indices=None, offsets=None, weights=None):
        if self.hostname_signal:
            lexical = self.embedding(indices, offsets, per_sample_weights=weights)
            nonlinear_input = torch.cat([X, lexical], dim=1)
        else:
            nonlinear_input = X
        return self.linear(X) + self.nonlinear(nonlinear_input)


class RelationalResidualMLP(BaseEstimator):
    """Fold-fitted relational model with an optional learned hostname branch."""

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

        if self.hostname_signal:
            self.hasher_ = HashingVectorizer(
                analyzer='char',
                ngram_range=(3, 5),
                n_features=32768,
                lowercase=False,
                alternate_sign=False,
                norm=None,
                dtype=np.float32,
            )
            hashed = self.hasher_.transform(X['hostname'].fillna('').astype(str))
            hashed.sort_indices()
            print(
                'hostname_hash_training',
                'rows', hashed.shape[0],
                'mean_occupied_buckets', hashed.nnz / max(hashed.shape[0], 1),
                'nonempty_fraction', float(np.mean(np.diff(hashed.indptr) > 0)),
                'active_buckets', len(np.unique(hashed.indices)),
                flush=True,
            )
        else:
            hashed = None

        self.net_ = ResidualTrackerNetwork(
            features.shape[1], hostname_signal=self.hostname_signal
        ).to(self.device_)
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
            torch.from_numpy(features),
            torch.from_numpy(target),
            torch.from_numpy(weight),
            torch.arange(len(features), dtype=torch.int64),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed),
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
                optimizer.zero_grad(set_to_none=True)
                if self.hostname_signal:
                    block = hashed[row_indices.numpy()]
                    indices = torch.as_tensor(
                        block.indices.astype(np.int64), device=self.device_
                    )
                    offsets = torch.as_tensor(
                        block.indptr.astype(np.int64), device=self.device_
                    )
                    totals = np.asarray(block.sum(axis=1)).ravel()
                    sample_weights = block.data / np.repeat(
                        np.maximum(totals, 1), np.diff(block.indptr)
                    )
                    sample_weights = torch.as_tensor(
                        sample_weights, dtype=torch.float32, device=self.device_
                    )
                    logits = self.net_(batch, indices, offsets, sample_weights)
                else:
                    logits = self.net_(batch)
                loss = (
                    nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, reduction='none'
                    ).sum(dim=1) * weights
                ).mean()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
                seen += len(batch)
            print(
                'relational_residual_mlp',
                'hostname_signal', self.hostname_signal,
                'epoch', epoch + 1,
                'loss', total / seen,
                flush=True,
            )
        return self

    def predict(self, X):
        self.net_.eval()
        predictions = []
        hash_nonempty = 0
        hash_nnz = 0
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
                if self.hostname_signal:
                    block = self.hasher_.transform(
                        frame['hostname'].fillna('').astype(str)
                    )
                    block.sort_indices()
                    hash_nonempty += int(np.count_nonzero(np.diff(block.indptr)))
                    hash_nnz += block.nnz
                    indices = torch.as_tensor(
                        block.indices.astype(np.int64), device=self.device_
                    )
                    offsets = torch.as_tensor(
                        block.indptr.astype(np.int64), device=self.device_
                    )
                    totals = np.asarray(block.sum(axis=1)).ravel()
                    sample_weights = block.data / np.repeat(
                        np.maximum(totals, 1), np.diff(block.indptr)
                    )
                    sample_weights = torch.as_tensor(
                        sample_weights, dtype=torch.float32, device=self.device_
                    )
                    logits = self.net_(batch, indices, offsets, sample_weights)
                else:
                    logits = self.net_(batch)
                predictions.append(logits.sigmoid().cpu().numpy())
        if self.hostname_signal:
            print(
                'hostname_hash_prediction',
                'rows', len(X),
                'mean_occupied_buckets', hash_nnz / max(len(X), 1),
                'nonempty_fraction', hash_nonempty / max(len(X), 1),
                flush=True,
            )
        return np.concatenate(predictions, axis=0)

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).predict(X)


def sparse_matrix(values, rows, columns, shape):
    specification = skrub.as_data_op((values, (rows, columns)))
    return specification.skb.apply_func(sp.csr_matrix, shape=shape, dtype=np.float32)


def direction_features(edges, relevant, auxiliary, n_domains, source, neighbour, prefix, prior):
    selected = edges[edges[source].isin(relevant['domain_id'])]
    selected = selected[selected[source] != selected[neighbour]]
    selected = selected.drop_duplicates([source, neighbour])
    row_lookup = relevant[['domain_id', 'profile_row']].rename(columns={'domain_id': source})
    selected = selected.merge(row_lookup, on=source, how='inner', sort=False)
    row_numbers = selected['profile_row'].to_numpy()
    neighbour_numbers = selected[neighbour].to_numpy()
    values = selected[source].astype('float32') * 0 + 1
    adjacency = sparse_matrix(
        values.to_numpy(), row_numbers, neighbour_numbers,
        (relevant.shape[0], n_domains)
    )
    degree = adjacency.sum(axis=1).skb.apply_func(np.asarray).ravel()
    sums = adjacency.dot(auxiliary)
    count = sums.sum(axis=1).skb.apply_func(np.asarray).ravel()
    names = [prefix + '_tracker_' + str(i) for i in range(N_TRACKERS)]
    profiles = sums.toarray().skb.apply_func(pd.DataFrame, columns=list(range(N_TRACKERS)))
    denominator = count.skb.apply_func(pd.Series).replace(0, np.nan)
    profiles = profiles.div(denominator, axis=0).fillna(prior)
    profiles = profiles.rename(columns=dict(enumerate(names)))
    table = relevant[['domain_id']].assign(**{
        prefix + '_degree': degree,
        prefix + '_labelled_count': count,
        prefix + '_coverage': count / (degree + 1e-06),
    })
    table = table.skb.concat([profiles], axis=1)
    for name in [prefix + '_degree', prefix + '_labelled_count']:
        table = table.assign(**{name: table[name].skb.apply_func(np.log1p)})
    return table


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
    n_domains = domains['domain_id'].max() + 1
    labels = skrub.as_data_op(BASE + 'tracking_graph_train.parquet').skb.apply_func(
        pd.read_parquet, columns=['domain_id', 'tracker_id']
    )
    external = labels[~labels['domain_id'].isin(setup['row_keys']['domain_id'])]
    external = external.drop_duplicates(['domain_id', 'tracker_id'])
    cardinality = external.groupby('domain_id')['tracker_id'].nunique().rename('aux_cardinality').reset_index()
    external = external.merge(cardinality, on='domain_id', how='inner', sort=False)
    external = external.assign(aux_weight=1.0 / external['aux_cardinality'])
    auxiliary = sparse_matrix(
        external['aux_weight'].astype('float32').to_numpy(),
        external['domain_id'].to_numpy(),
        external['tracker_id'].to_numpy(),
        (n_domains, N_TRACKERS),
    )
    prior = external.groupby('tracker_id')['aux_weight'].sum()
    prior = prior.reindex(list(range(N_TRACKERS)), fill_value=0) / cardinality.shape[0]

    links = skrub.as_data_op(BASE + 'link-graph.parquet').skb.apply_func(
        pd.read_parquet, columns=['source_domain_id', 'target_domain_id']
    )
    links = links[
        links['source_domain_id'].isin(relevant['domain_id'])
        | links['target_domain_id'].isin(relevant['domain_id'])
    ]
    outgoing = direction_features(
        links, relevant, auxiliary, n_domains,
        'source_domain_id', 'target_domain_id', 'out', prior
    )
    incoming = direction_features(
        links, relevant, auxiliary, n_domains,
        'target_domain_id', 'source_domain_id', 'in', prior
    )

    lookup = domains[domains['domain_id'].isin(relevant['domain_id'])]
    lookup = lookup.drop_duplicates('domain_id')
    host = lookup['domain'].fillna('').str.lower().str.replace('^www\\.', '', regex=True)
    lookup = lookup.assign(
        tld=host.str.split('.').str[-1],
        hostname_length=host.str.len().astype('float32'),
        hostname_dots=host.str.count('\\.').astype('float32'),
        hostname_digits=host.str.count('[0-9]').astype('float32'),
        hostname_hyphens=host.str.count('-').astype('float32'),
        hostname='^' + host + '$',
    ).drop(columns=['domain'])

    classified = skrub.as_data_op(BASE + 'url-classification.csv').skb.apply_func(
        pd.read_csv, usecols=['url']
    )
    classification_host = classified['url'].fillna('').str.lower()
    classification_host = classification_host.str.replace('^https?://', '', regex=True)
    classification_host = classification_host.str.split('/').str[0].str.replace('^www\\.', '', regex=True)
    classified = classified.assign(host=classification_host, classified=1)
    classified = classified[['host', 'classified']].drop_duplicates('host')
    host_lookup = domains[domains['domain_id'].isin(relevant['domain_id'])]
    host_lookup = host_lookup.assign(
        host=host_lookup['domain'].fillna('').str.lower().str.replace('^www\\.', '', regex=True)
    )
    coverage = host_lookup.merge(classified, on='host', how='left', sort=False)
    coverage = coverage[['domain_id', 'classified']].drop_duplicates('domain_id')
    coverage = coverage.fillna({'classified': 0})

    X = setup['X'].merge(lookup, on='domain_id', how='left', sort=False)
    X = X.merge(coverage, on='domain_id', how='left', sort=False)
    X = X.merge(outgoing, on='domain_id', how='left', sort=False)
    X = X.merge(incoming, on='domain_id', how='left', sort=False)

    hostname_signal = skrub.choose_from(
        {
            'relational_only': False,
            'relational_plus_character_ngrams': True,
        },
        name='hostname_signal',
    )
    prediction = X.skb.apply(
        RelationalResidualMLP(hostname_signal=hostname_signal),
        y=setup['y'],
    )

    # Label-free audits, kept separate from the model input. Hash occupancy is
    # additionally printed by the fitted model for each training/test frame.
    metadata = relevant[['domain_id']].merge(
        lookup[['domain_id', 'hostname_length']], on='domain_id', how='left', sort=False
    )
    length = metadata['hostname_length'].fillna(0)
    metadata = metadata.assign(
        hostname_available=(length > 0).astype('float32'),
        hostname_length=length,
        character_ngram_count=length + (length - 1).clip(lower=0) + (length - 2).clip(lower=0),
        hashed_feature_nonempty=(length > 0).astype('float32'),
    )
    metadata = metadata.merge(
        outgoing[['domain_id', 'out_degree', 'out_labelled_count']],
        on='domain_id', how='left', sort=False
    )
    metadata = metadata.merge(
        incoming[['domain_id', 'in_degree', 'in_labelled_count']],
        on='domain_id', how='left', sort=False
    )
    metadata = metadata.assign(
        hyperlink_coverage_code=(
            (metadata['out_degree'] > 0).astype('int64')
            + 2 * (metadata['in_degree'] > 0).astype('int64')
        ),
        external_label_coverage_code=(
            (metadata['out_labelled_count'] > 0).astype('int64')
            + 2 * (metadata['in_labelled_count'] > 0).astype('int64')
        ),
    )
    audit_columns = [
        'hostname_available', 'hostname_length', 'character_ngram_count',
        'hashed_feature_nonempty',
    ]
    pool_audit = setup['row_keys'].merge(
        metadata, on='domain_id', how='left', sort=False
    ).assign(population='locked_pool')
    target_audit = targets.merge(
        metadata, on='domain_id', how='left', sort=False
    ).assign(population='prediction')
    audit_rows = pool_audit.skb.concat([target_audit], axis=0)
    audits = {
        'hostname_population_means': audit_rows.groupby('population')[audit_columns].mean().reset_index(),
        'hyperlink_coverage_counts': audit_rows.groupby(
            ['population', 'hyperlink_coverage_code']
        ).size().rename('domains').reset_index(),
        'external_label_coverage_counts': audit_rows.groupby(
            ['population', 'external_label_coverage_code']
        ).size().rename('domains').reset_index(),
        'validation_slice_keys': pool_audit.iloc[:N_RESERVED][[
            'domain_id', 'hyperlink_coverage_code', 'external_label_coverage_code'
        ]],
        'audit_semantics': skrub.as_data_op({
            'hyperlink_coverage_code': '0 neither, 1 outgoing only, 2 incoming only, 3 both',
            'external_label_coverage_code': 'Same encoding, but requires externally labelled neighbours.',
            'hashed_feature_nonempty': 'Whether at least one boundary-marked 3-5 gram exists; actual occupied buckets are printed by the estimator.',
            'standalone_signal_evidence': 'Earlier independently evaluated hostname candidate: Recall@10 0.797973; different construction, not an exact ablation.',
            'slice_scores': 'Require harness OOF predictions and validation truth; the locked scorer remains unchanged. Audit graphs do not fit or score models.',
        }),
    }
    return {
        'pred': prediction,
        'scoring': setup['scoring'],
        'row_keys': setup['row_keys'],
        'audit': audits,
    }