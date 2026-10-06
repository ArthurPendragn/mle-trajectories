"""
Evaluation Setup for TrackTheTrackers task.

POPULATION COMPARISON (Adversarial ROC AUC against target domains across selection rules):
- tc_ge_2 (tracker_count >= 2): AUC = 0.4965 +/- 0.0059 (retention 46.2%, 8,635,505 domains)
- tc_2_to_20 (2 <= tracker_count <= 20): AUC = 0.4994 +/- 0.0030 (retention 46.2%, 8,630,889 domains) [LOCKED: indistinguishable from target, AUC ~ 0.50]
- tc_2_to_15 (2 <= tracker_count <= 15): AUC = 0.5005 +/- 0.0048 (retention 46.2%, 8,623,306 domains)
- tc_2_to_10 (2 <= tracker_count <= 10): AUC = 0.5057 +/- 0.0050 (retention 45.9%, 8,575,627 domains)
- link_and_tc_2_to_20 (has link & 2 <= tc <= 20): AUC = 0.5305 +/- 0.0059 (retention 43.0%, 8,030,239 domains)
- link_and_tc_ge_2 (has link & tc >= 2): AUC = 0.5349 +/- 0.0055 (retention 43.0%, 8,034,846 domains)
- tc_ge_3 (tracker_count >= 3): AUC = 0.5526 +/- 0.0096 (retention 23.1%, 4,313,286 domains)
- tc_3_to_20 (3 <= tracker_count <= 20): AUC = 0.5533 +/- 0.0051 (retention 23.1%, 4,308,670 domains)
- tc_le_15 (tracker_count <= 15): AUC = 0.5720 +/- 0.0031 (retention 99.9%, 18,670,700 domains)
- all_train (tracker_count >= 1): AUC = 0.5729 +/- 0.0045 (retention 100.0%, 18,682,899 domains)
- tc_1_to_20 (1 <= tracker_count <= 20): AUC = 0.5740 +/- 0.0014 (retention 100.0%, 18,678,283 domains)
- tc_le_20 (tracker_count <= 20): AUC = 0.5740 +/- 0.0014 (retention 100.0%, 18,678,283 domains)
- tc_le_10 (tracker_count <= 10): AUC = 0.5741 +/- 0.0035 (retention 99.7%, 18,623,021 domains)
- tc_le_5 (tracker_count <= 5): AUC = 0.5870 +/- 0.0051 (retention 96.6%, 18,055,440 domains)
- has_link (in_degree > 0 or out_degree > 0): AUC = 0.6018 +/- 0.0061 (retention 94.1%, 17,577,712 domains)
- in_link (in_degree > 0): AUC = 0.6276 +/- 0.0085 (retention 87.5%, 16,347,018 domains)
- tc_ge_5 (tracker_count >= 5): AUC = 0.6962 +/- 0.0030 (retention 6.2%, 1,158,047 domains)
- tc_ge_10 (tracker_count >= 10): AUC = 0.8743 +/- 0.0028 (retention 0.5%, 90,936 domains)

LEARNING CURVE STUDY:
- Model: PyTorch 2-layer MLP (TF-IDF char n-grams + degree -> 128 -> 355 logits, BCEWithLogitsLoss on GPU)
- Scores and times:
  * N = 1,000: Recall@10 = 0.7742 +/- 0.0070, time/fold = 1.47s, 5-fold pipeline = ~7.4s
  * N = 3,000: Recall@10 = 0.7996 +/- 0.0159, time/fold = 0.15s, 5-fold pipeline = ~0.8s
  * N = 10,000: Recall@10 = 0.8857 +/- 0.0035, time/fold = 0.86s, 5-fold pipeline = ~4.3s
  * N = 25,000: Recall@10 = 0.9117 +/- 0.0036, time/fold = 1.69s, 5-fold pipeline = ~8.4s
  * N = 50,000: Recall@10 = 0.8635 +/- 0.0004, time/fold = 2.78s, 5-fold pipeline = ~13.9s
  * N = 18,682,899 (all train): Recall@10 = 0.8132 +/- 0.0005, time/fold = 899.2s, 5-fold pipeline = ~4496s
- Choice of pool size: 50,000 domains sampled from the locked 2 <= tracker_count <= 20 population.
  It matches the prediction target size (50,000 domains), runs 5-fold cross-validation in under 14 seconds
  (allowing over 1,500 full model explorations and tuning iterations within the remaining budget), avoids
  the extreme computational burden of 18.7M domains (~75 min per fold run), and achieves strong recall.
"""
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.sparse import hstack
import skrub
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

TRACKING_GRAPH_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/tracking_graph_train.parquet'
DOMAINS_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/domains.parquet'
LINK_GRAPH_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/link-graph.parquet'

def recall_at_10(estimator, X, y):
    """
    Competition metric: Recall@10.
    For each domain, the fraction of its true trackers that appear anywhere in the
    top 10 predicted trackers for that domain, averaged across all domains.
    """
    y_arr = np.asarray(y)
    n_samples = len(y_arr)
    n_classes = y_arr.shape[1] if y_arr.ndim > 1 else 1
    if hasattr(estimator, 'predict_proba'):
        preds = estimator.predict_proba(X)
    elif hasattr(estimator, 'decision_function'):
        preds = estimator.decision_function(X)
    else:
        preds = estimator.predict(X)
    if isinstance(preds, list):
        cols = []
        for p in preds:
            p_arr = np.asarray(p)
            if p_arr.ndim == 2 and p_arr.shape[1] >= 2:
                cols.append(p_arr[:, 1])
            elif p_arr.ndim == 2 and p_arr.shape[1] == 1:
                cols.append(p_arr[:, 0])
            else:
                cols.append(p_arr.ravel())
        preds = np.column_stack(cols)
    else:
        preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] == n_classes:
        top_k = min(10, n_classes)
        top10 = np.argpartition(-preds, top_k, axis=1)[:, :top_k]
        hits = y_arr[np.arange(n_samples)[:, None], top10].sum(axis=1)
    elif preds.ndim == 2 and preds.shape[1] <= 10:
        hits = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            pred_set = set(preds[i])
            true_indices = np.flatnonzero(y_arr[i])
            hits[i] = len(pred_set.intersection(true_indices))
    else:
        top_k = min(10, preds.shape[1])
        top10 = np.argpartition(-preds, top_k, axis=1)[:, :top_k]
        hits = y_arr[np.arange(n_samples)[:, None], top10].sum(axis=1)
    true_counts = y_arr.sum(axis=1)
    recalls = hits / np.maximum(true_counts, 1)
    return float(np.mean(recalls))

def locked_setup_helper():
    tracking_graph = skrub.as_data_op(TRACKING_GRAPH_PATH).skb.apply_func(pd.read_parquet, columns=['domain_id', 'tracker_id'])
    tracker_counts = tracking_graph.groupby('domain_id', as_index=False).agg({'tracker_id': 'count'}).rename(columns={'tracker_id': 'tracker_count'})
    valid_domains = tracker_counts[(tracker_counts['tracker_count'] >= 2) & (tracker_counts['tracker_count'] <= 20)]
    sampled_domains = valid_domains.sample(n=50000, random_state=42).sort_values('domain_id').reset_index(drop=True)
    domains = skrub.as_data_op(DOMAINS_PATH).skb.apply_func(pd.read_parquet)
    X_unmarked = sampled_domains[['domain_id']].merge(domains, on='domain_id', how='left')
    sampled_tracking = sampled_domains[['domain_id']].merge(tracking_graph, on='domain_id', how='inner')
    sampled_tracking = sampled_tracking.assign(val=1)
    pivot = sampled_tracking.pivot_table(index='domain_id', columns='tracker_id', values='val', fill_value=0)
    all_tracker_ids = list(range(355))
    pivot = pivot.reindex(columns=all_tracker_ids, fill_value=0)
    pivot = pivot.rename(columns={i: f't_{i}' for i in range(355)})
    y_unmarked = pivot.reset_index(drop=True).astype('int8')
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    X = X_unmarked.skb.mark_as_X(cv=cv, split_kwargs={})
    y = y_unmarked.skb.mark_as_y()
    return {'X': X, 'y': y, 'scoring': recall_at_10}

def locked_setup_entry():
    return locked_setup_helper()

def build_evaluation():
    return locked_setup_entry()

def extract_lexical_features(domain_series):
    s = domain_series.astype(str)
    length = s.str.len().to_numpy(dtype=np.float32)[:, None]
    dots = s.str.count('\\.').to_numpy(dtype=np.float32)[:, None]
    hyphens = s.str.count('-').to_numpy(dtype=np.float32)[:, None]
    digits = s.str.count('\\d').to_numpy(dtype=np.float32)[:, None]
    digit_ratio = digits / np.maximum(length, 1.0)
    return np.hstack([length, dots, hyphens, digit_ratio])

def extract_graph_features(X_df):
    if isinstance(X_df, pd.DataFrame):
        in_deg = pd.to_numeric(X_df.get('in_degree', 0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        out_deg = pd.to_numeric(X_df.get('out_degree', 0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    else:
        in_deg = np.zeros(len(X_df), dtype=np.float32)
        out_deg = np.zeros(len(X_df), dtype=np.float32)
    in_deg = np.maximum(in_deg, 0.0)
    out_deg = np.maximum(out_deg, 0.0)
    log_in = np.log1p(in_deg)[:, None]
    log_out = np.log1p(out_deg)[:, None]
    has_in = (in_deg > 0.0).astype(np.float32)[:, None]
    has_out = (out_deg > 0.0).astype(np.float32)[:, None]
    in_ratio = (in_deg / (in_deg + out_deg + 1.0))[:, None]
    return np.hstack([log_in, log_out, has_in, has_out, in_ratio])

class MultiOutputRidgeTracker(BaseEstimator):

    def __init__(self, alpha=100.0, max_features=12000, use_graph=True):
        self.alpha = alpha
        self.max_features = max_features
        self.use_graph = use_graph

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame) and 'domain' in X.columns:
            domain_series = X['domain'].fillna('').astype(str)
        else:
            domain_series = pd.Series(X).fillna('').astype(str)
        self.tfidf_ = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=3,
            max_features=self.max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        X_tfidf = self.tfidf_.fit_transform(domain_series)
        X_lex = extract_lexical_features(domain_series)
        self.scaler_ = StandardScaler()
        X_lex_scaled = self.scaler_.fit_transform(X_lex).astype(np.float32)

        feature_blocks = [X_tfidf, X_lex_scaled]
        if self.use_graph:
            X_graph = extract_graph_features(X)
            self.graph_scaler_ = StandardScaler()
            X_graph_scaled = self.graph_scaler_.fit_transform(X_graph).astype(np.float32)
            feature_blocks.append(X_graph_scaled)

        X_combined = hstack(feature_blocks, format='csr', dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        N = X_combined.shape[0]
        D = X_combined.shape[1]
        X_mean = np.asarray(X_combined.mean(axis=0), dtype=np.float32).ravel()
        Y_mean = np.asarray(y_arr.mean(axis=0), dtype=np.float32).ravel()
        A = X_combined.T.dot(X_combined).toarray().astype(np.float32)
        A -= (N * np.outer(X_mean, X_mean)).astype(np.float32)
        A = 0.5 * (A + A.T)
        A[np.diag_indices(D)] += np.float32(self.alpha)
        B = X_combined.T.dot(y_arr).astype(np.float32)
        B -= (N * np.outer(X_mean, Y_mean)).astype(np.float32)
        try:
            W = scipy.linalg.solve(A, B, assume_a='pos')
        except Exception:
            W = scipy.linalg.solve(A, B)
        b = Y_mean.reshape(1, -1) - X_mean.reshape(1, -1) @ W
        self.coef_ = W.astype(np.float32)
        self.intercept_ = b.astype(np.float32)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and 'domain' in X.columns:
            domain_series = X['domain'].fillna('').astype(str)
        else:
            domain_series = pd.Series(X).fillna('').astype(str)
        X_tfidf = self.tfidf_.transform(domain_series)
        X_lex = extract_lexical_features(domain_series)
        X_lex_scaled = self.scaler_.transform(X_lex).astype(np.float32)

        feature_blocks = [X_tfidf, X_lex_scaled]
        if self.use_graph:
            X_graph = extract_graph_features(X)
            X_graph_scaled = self.graph_scaler_.transform(X_graph).astype(np.float32)
            feature_blocks.append(X_graph_scaled)

        X_combined = hstack(feature_blocks, format='csr', dtype=np.float32)
        scores = X_combined.dot(self.coef_) + self.intercept_
        return np.asarray(scores, dtype=np.float32)

    def predict_proba(self, X):
        return self.predict(X)

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).predict(X)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    link_graph = skrub.as_data_op(LINK_GRAPH_PATH).skb.apply_func(
        pd.read_parquet, columns=['source_domain_id', 'target_domain_id']
    )
    in_counts = link_graph.groupby('target_domain_id', as_index=False).agg(
        {'source_domain_id': 'count'}
    ).rename(columns={'target_domain_id': 'domain_id', 'source_domain_id': 'in_degree'})
    out_counts = link_graph.groupby('source_domain_id', as_index=False).agg(
        {'target_domain_id': 'count'}
    ).rename(columns={'source_domain_id': 'domain_id', 'target_domain_id': 'out_degree'})

    X_graph = X.merge(in_counts, on='domain_id', how='left').merge(out_counts, on='domain_id', how='left')

    models = {
        'ridge_alpha100_nograph': MultiOutputRidgeTracker(alpha=100.0, use_graph=False, max_features=12000),
        'ridge_alpha100_graph': MultiOutputRidgeTracker(alpha=100.0, use_graph=True, max_features=12000),
        'ridge_alpha300_graph': MultiOutputRidgeTracker(alpha=300.0, use_graph=True, max_features=12000),
        'ridge_alpha1000_graph': MultiOutputRidgeTracker(alpha=1000.0, use_graph=True, max_features=12000),
    }
    model = skrub.choose_from(models, name='model_variant')
    pred = X_graph.skb.apply(model, y=y)
    return {'pred': pred, 'scoring': setup['scoring']}