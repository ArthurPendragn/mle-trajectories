import random
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import skrub
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

TRACKING_GRAPH_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/tracking_graph_train.parquet'
DOMAINS_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/domains.parquet'


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


class MLPModule(nn.Module):
    def __init__(self, in_features, hidden_dim_1, hidden_dim_2, n_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class TrackerMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_dim_1=512,
        hidden_dim_2=256,
        dropout=0.1,
        lr=1e-3,
        epochs=12,
        batch_size=256,
        weight_decay=1e-4,
        max_features=10000,
        random_state=42,
    ):
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.max_features = max_features
        self.random_state = random_state

    def _extract_lexical(self, domains_series):
        if not isinstance(domains_series, pd.Series):
            domains_series = pd.Series(domains_series)
        lens = domains_series.str.len().fillna(0).astype(np.float32).values[:, None]
        dots = domains_series.str.count(r'\.').fillna(0).astype(np.float32).values[:, None]
        hyphens = domains_series.str.count(r'-').fillna(0).astype(np.float32).values[:, None]
        digits = domains_series.str.count(r'\d').fillna(0).astype(np.float32).values[:, None]
        return np.hstack([lens, dots, hyphens, digits])

    def fit(self, X, y):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        if isinstance(X, pd.DataFrame):
            domains = X['domain'].fillna('').astype(str)
        elif isinstance(X, pd.Series):
            domains = X.fillna('').astype(str)
        else:
            domains = pd.Series(X).fillna('').astype(str)

        lex_mat = self._extract_lexical(domains)
        self.scaler_ = StandardScaler()
        scaled_lex = self.scaler_.fit_transform(lex_mat)

        self.tfidf_ = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=2,
        )
        tfidf_mat = self.tfidf_.fit_transform(domains)

        combined_mat = hstack([tfidf_mat, scaled_lex], format='csr')
        X_dense = combined_mat.toarray().astype(np.float32)

        if isinstance(y, pd.DataFrame):
            y_mat = y.values.astype(np.float32)
        else:
            y_mat = np.asarray(y, dtype=np.float32)

        n_samples, in_features = X_dense.shape
        n_classes = y_mat.shape[1]
        self.classes_ = np.arange(n_classes)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ = device
        self.net_ = MLPModule(
            in_features=in_features,
            hidden_dim_1=self.hidden_dim_1,
            hidden_dim_2=self.hidden_dim_2,
            n_classes=n_classes,
            dropout=self.dropout,
        ).to(device)

        dataset = TensorDataset(
            torch.from_numpy(X_dense),
            torch.from_numpy(y_mat),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = optim.AdamW(
            self.net_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = max(1, self.epochs * len(loader))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-5,
        )
        criterion = nn.BCEWithLogitsLoss()

        self.net_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = self.net_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                num_batches += 1
            if (epoch + 1) % 4 == 0 or epoch == self.epochs - 1:
                avg_loss = total_loss / max(1, num_batches)
                print(f"Epoch {epoch+1}/{self.epochs} - BCE Loss: {avg_loss:.4f}")

        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            domains = X['domain'].fillna('').astype(str)
        elif isinstance(X, pd.Series):
            domains = X.fillna('').astype(str)
        else:
            domains = pd.Series(X).fillna('').astype(str)

        lex_mat = self._extract_lexical(domains)
        scaled_lex = self.scaler_.transform(lex_mat)
        tfidf_mat = self.tfidf_.transform(domains)

        combined_mat = hstack([tfidf_mat, scaled_lex], format='csr')
        X_dense = combined_mat.toarray().astype(np.float32)

        device = self.device_ if hasattr(self, 'device_') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net_.eval()
        n_samples = X_dense.shape[0]
        eval_batch_size = 1024
        all_probs = []

        with torch.no_grad():
            for i in range(0, n_samples, eval_batch_size):
                batch_arr = X_dense[i:i + eval_batch_size]
                batch_tensor = torch.from_numpy(batch_arr).to(device)
                logits = self.net_(batch_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        if len(all_probs) > 0:
            return np.vstack(all_probs)
        return np.empty((0, len(self.classes_)), dtype=np.float32)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.int8)

    def transform(self, X):
        return self.predict_proba(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).predict_proba(X)


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    model = skrub.choose_from(
        {
            'mlp_512_256_drop0.1': TrackerMLPClassifier(
                hidden_dim_1=512, hidden_dim_2=256, dropout=0.1, lr=1e-3, epochs=12, batch_size=256
            ),
            'mlp_512_256_drop0.2': TrackerMLPClassifier(
                hidden_dim_1=512, hidden_dim_2=256, dropout=0.2, lr=1e-3, epochs=12, batch_size=256
            ),
            'mlp_256_128_drop0.1': TrackerMLPClassifier(
                hidden_dim_1=256, hidden_dim_2=128, dropout=0.1, lr=1e-3, epochs=12, batch_size=256
            ),
            'mlp_256_128_drop0.2': TrackerMLPClassifier(
                hidden_dim_1=256, hidden_dim_2=128, dropout=0.2, lr=1e-3, epochs=12, batch_size=256
            ),
        },
        name='model_variant',
    )

    pred = X.skb.apply(model, y=y)
    return {'pred': pred, 'scoring': scoring}