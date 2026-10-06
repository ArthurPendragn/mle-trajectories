import numpy as np
import pandas as pd
import scipy.linalg
from scipy.sparse import hstack
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

TRACKING_GRAPH_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/tracking_graph_train.parquet'
DOMAINS_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/domains.parquet'
LINK_GRAPH_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/link-graph.parquet'
TRACKERS_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/trackers.tsv'
URL_CLASSIFICATION_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/url-classification.csv'
FREEDOM_OF_THE_PRESS_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/freedom-of-the-press.csv'
TARGET_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/target.tsv'


class UrlClassificationStudyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        target_df = df[df['split'] == 'target'].drop_duplicates(subset=['domain_id'])
        train_df = df[df['split'] == 'train'].drop_duplicates(subset=['domain_id'])
        n_target = len(target_df)
        n_train = len(train_df)
        n_target_matched = int(target_df['category'].notna().sum())
        n_train_matched = int(train_df['category'].notna().sum())
        target_cov = 100.0 * n_target_matched / max(n_target, 1)
        train_cov = 100.0 * n_train_matched / max(n_train, 1)

        target_cats = target_df['category'].dropna().value_counts(normalize=True)
        train_cats = train_df['category'].dropna().value_counts(normalize=True)
        all_cats = sorted(list(set(target_cats.index).union(set(train_cats.index))))

        train_with_cat_and_tracker = df[(df['split'] == 'train') & df['category'].notna() & (df['tracker_id'] >= 0)]
        cat_tracker_assoc = {}
        for cat in all_cats:
            sub = train_with_cat_and_tracker[train_with_cat_and_tracker['category'] == cat]
            if len(sub) > 0:
                top_trackers = sub['tracker_id'].value_counts().head(3)
                cat_tracker_assoc[cat] = [f"t_{t}:{cnt}" for t, cnt in top_trackers.items()]
            else:
                cat_tracker_assoc[cat] = []

        rows = [
            {
                'metric': 'total_domains',
                'target_val': f"{n_target}",
                'train_val': f"{n_train}",
                'notes': 'Candidate population domain count',
            },
            {
                'metric': 'url_classification_matched_count',
                'target_val': f"{n_target_matched}",
                'train_val': f"{n_train_matched}",
                'notes': 'Domains matched to url-classification.csv categories',
            },
            {
                'metric': 'url_classification_coverage_pct',
                'target_val': f"{target_cov:.2f}%",
                'train_val': f"{train_cov:.2f}%",
                'notes': 'Percentage of domains with content category',
            },
        ]
        for cat in all_cats:
            t_pct = 100.0 * target_cats.get(cat, 0.0)
            tr_pct = 100.0 * train_cats.get(cat, 0.0)
            assoc_str = ", ".join(cat_tracker_assoc.get(cat, []))
            rows.append({
                'metric': f"cat_share_{cat}",
                'target_val': f"{t_pct:.2f}%",
                'train_val': f"{tr_pct:.2f}%",
                'notes': f"Top trackers in train: {assoc_str}" if assoc_str else "No matched trackers",
            })
        return pd.DataFrame(rows)


class TldPressFreedomStudyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        target_df = df[df['split'] == 'target'].drop_duplicates(subset=['domain_id'])
        train_df = df[df['split'] == 'train'].drop_duplicates(subset=['domain_id'])
        n_target = len(target_df)
        n_train = len(train_df)

        t_fotp = pd.to_numeric(target_df['freedom_of_the_press'].astype(str).str.replace(',', '.'), errors='coerce').dropna()
        tr_fotp = pd.to_numeric(train_df['freedom_of_the_press'].astype(str).str.replace(',', '.'), errors='coerce').dropna()
        t_join_pct = 100.0 * len(t_fotp) / max(n_target, 1)
        tr_join_pct = 100.0 * len(tr_fotp) / max(n_train, 1)

        t_mean = float(t_fotp.mean()) if len(t_fotp) > 0 else 0.0
        tr_mean = float(tr_fotp.mean()) if len(tr_fotp) > 0 else 0.0
        t_med = float(t_fotp.median()) if len(t_fotp) > 0 else 0.0
        tr_med = float(tr_fotp.median()) if len(tr_fotp) > 0 else 0.0
        t_std = float(t_fotp.std()) if len(t_fotp) > 1 else 0.0
        tr_std = float(tr_fotp.std()) if len(tr_fotp) > 1 else 0.0

        t_tlds = target_df['tld'].value_counts(normalize=True)
        tr_tlds = train_df['tld'].value_counts(normalize=True)
        top_tlds = list(tr_tlds.head(10).index)

        train_trackers = df[(df['split'] == 'train') & (df['tracker_id'] >= 0)]
        tld_tracker_signals = {}
        for tld in ['ru', 'de', 'uk', 'fr', 'jp', 'cn', 'pl', 'it', 'br', 'com']:
            sub = train_trackers[train_trackers['tld'] == tld]
            if len(sub) > 0:
                top_tr = sub['tracker_id'].value_counts().head(3)
                tld_tracker_signals[tld] = [f"t_{t}:{cnt}" for t, cnt in top_tr.items()]
            else:
                tld_tracker_signals[tld] = []

        rows = [
            {
                'metric': 'freedom_of_the_press_join_pct',
                'target_val': f"{t_join_pct:.2f}%",
                'train_val': f"{tr_join_pct:.2f}%",
                'notes': 'Percentage of domains with country press freedom score',
            },
            {
                'metric': 'freedom_score_mean_std',
                'target_val': f"{t_mean:.2f} +/- {t_std:.2f}",
                'train_val': f"{tr_mean:.2f} +/- {tr_std:.2f}",
                'notes': 'Country press freedom index (lower = more free)',
            },
            {
                'metric': 'freedom_score_median',
                'target_val': f"{t_med:.2f}",
                'train_val': f"{tr_med:.2f}",
                'notes': 'Median press freedom score',
            },
        ]
        for tld in top_tlds:
            t_pct = 100.0 * t_tlds.get(tld, 0.0)
            tr_pct = 100.0 * tr_tlds.get(tld, 0.0)
            sig_str = ", ".join(tld_tracker_signals.get(tld, []))
            rows.append({
                'metric': f"tld_share_{tld}",
                'target_val': f"{t_pct:.2f}%",
                'train_val': f"{tr_pct:.2f}%",
                'notes': f"Top trackers on .{tld}: {sig_str}" if sig_str else "No distinctive signal",
            })
        return pd.DataFrame(rows)


class TrackerHyperlinksStudyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        train_links = df[df['split'] == 'train']
        target_links = df[df['split'] == 'target']

        n_train_links = len(train_links)
        n_target_links = len(target_links)

        train_doms_with_links = int(train_links['domain_id'].nunique())
        target_doms_with_links = int(target_links['domain_id'].nunique())

        train_trackers_linked = int(train_links['tracker_id'].nunique())
        target_trackers_linked = int(target_links['tracker_id'].nunique())

        tp_links = int((train_links['is_true_tracker'] == 1).sum()) if n_train_links > 0 else 0
        total_links = max(n_train_links, 1)
        precision_overall = 100.0 * tp_links / total_links

        top_linked = train_links['tracker_id'].value_counts().head(8)
        tracker_precision_breakdown = []
        for tr_id, cnt in top_linked.items():
            sub = train_links[train_links['tracker_id'] == tr_id]
            prec = 100.0 * (sub['is_true_tracker'] == 1).sum() / max(len(sub), 1)
            comp = sub['company'].iloc[0] if 'company' in sub.columns and pd.notna(sub['company'].iloc[0]) else "Unknown"
            tracker_precision_breakdown.append({
                'tracker_id': tr_id,
                'company': comp,
                'link_count': cnt,
                'precision_pct': f"{prec:.1f}%",
            })

        rows = [
            {
                'metric': 'domains_with_tracker_hyperlinks',
                'target_val': f"{target_doms_with_links} / 50000 ({100.0 * target_doms_with_links / 50000:.2f}%)",
                'train_val': f"{train_doms_with_links} / 50000 ({100.0 * train_doms_with_links / 50000:.2f}%)",
                'notes': 'Domains linking directly to any of the 355 tracker domains',
            },
            {
                'metric': 'total_direct_tracker_hyperlinks',
                'target_val': f"{n_target_links}",
                'train_val': f"{n_train_links}",
                'notes': 'Total (domain, tracker) hyperlink edges found in link-graph',
            },
            {
                'metric': 'unique_tracker_domains_linked',
                'target_val': f"{target_trackers_linked} / 355",
                'train_val': f"{train_trackers_linked} / 355",
                'notes': 'Number of tracker domains receiving direct hyperlinks',
            },
            {
                'metric': 'overall_hyperlink_precision',
                'target_val': 'N/A (unlabeled)',
                'train_val': f"{precision_overall:.2f}% ({tp_links} / {n_train_links})",
                'notes': 'Precision: fraction of tracker hyperlinks that are true trackers on domain',
            },
        ]
        for tr_info in tracker_precision_breakdown:
            rows.append({
                'metric': f"tracker_{tr_info['tracker_id']}_{tr_info['company']}",
                'target_val': 'N/A',
                'train_val': f"Links: {tr_info['link_count']}, Precision: {tr_info['precision_pct']}",
                'notes': f"Direct hyperlink precision for tracker {tr_info['tracker_id']}",
            })
        return pd.DataFrame(rows)


class TrackerStrataCalibrationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=100.0, max_features=10000):
        self.alpha = alpha
        self.max_features = max_features

    def _extract_lexical(self, domain_series):
        s = domain_series.astype(str)
        length = s.str.len().to_numpy(dtype=np.float32)[:, None]
        dots = s.str.count('\\.').to_numpy(dtype=np.float32)[:, None]
        hyphens = s.str.count('-').to_numpy(dtype=np.float32)[:, None]
        digits = s.str.count('\\d').to_numpy(dtype=np.float32)[:, None]
        digit_ratio = digits / np.maximum(length, 1.0)
        return np.hstack([length, dots, hyphens, digit_ratio])

    def _extract_graph(self, X_df):
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

    def _compute_metrics(self, preds, y_val, head_mask, mid_mask, tail_mask):
        n_val = len(preds)
        top10 = np.argpartition(-preds, 10, axis=1)[:, :10]
        hits_matrix = np.zeros_like(y_val, dtype=bool)
        row_indices = np.arange(n_val)[:, None]
        hits_matrix[row_indices, top10] = (y_val[row_indices, top10] == 1)

        overall_hits = hits_matrix.sum(axis=1)
        overall_true = np.maximum(y_val.sum(axis=1), 1)
        rec_overall = float(np.mean(overall_hits / overall_true))

        head_true = y_val[:, head_mask].sum(axis=1)
        head_hits = hits_matrix[:, head_mask].sum(axis=1)
        has_head = head_true > 0
        rec_head = float(np.mean(head_hits[has_head] / head_true[has_head])) if has_head.any() else 0.0

        mid_true = y_val[:, mid_mask].sum(axis=1)
        mid_hits = hits_matrix[:, mid_mask].sum(axis=1)
        has_mid = mid_true > 0
        rec_mid = float(np.mean(mid_hits[has_mid] / mid_true[has_mid])) if has_mid.any() else 0.0

        tail_true = y_val[:, tail_mask].sum(axis=1)
        tail_hits = hits_matrix[:, tail_mask].sum(axis=1)
        has_tail = tail_true > 0
        rec_tail = float(np.mean(tail_hits[has_tail] / tail_true[has_tail])) if has_tail.any() else 0.0

        return rec_overall, rec_head, rec_mid, rec_tail

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        y_arr = np.asarray(y, dtype=np.float32)
        N = len(X_df)

        np.random.seed(42)
        indices = np.random.permutation(N)
        train_idx = indices[:int(0.8 * N)]
        val_idx = indices[int(0.8 * N):]

        X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
        y_train = y_arr[train_idx]
        X_val_df = X_df.iloc[val_idx].reset_index(drop=True)
        y_val = y_arr[val_idx]

        priors = np.asarray(y_train.mean(axis=0), dtype=np.float32)

        head_mask = priors >= 0.05
        mid_mask = (priors >= 0.005) & (priors < 0.05)
        tail_mask = priors < 0.005

        n_head = int(head_mask.sum())
        n_mid = int(mid_mask.sum())
        n_tail = int(tail_mask.sum())

        domain_train = X_train_df['domain'].fillna('').astype(str)
        tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=3,
            max_features=self.max_features,
            sublinear_tf=True,
            dtype=np.float32
        )
        X_train_tfidf = tfidf.fit_transform(domain_train)
        X_train_lex = self._extract_lexical(domain_train)
        scaler_lex = StandardScaler()
        X_train_lex_scaled = scaler_lex.fit_transform(X_train_lex).astype(np.float32)
        X_train_graph = self._extract_graph(X_train_df)
        scaler_graph = StandardScaler()
        X_train_graph_scaled = scaler_graph.fit_transform(X_train_graph).astype(np.float32)
        X_train_combined = hstack([X_train_tfidf, X_train_lex_scaled, X_train_graph_scaled], format='csr', dtype=np.float32)

        N_tr = X_train_combined.shape[0]
        D = X_train_combined.shape[1]
        X_mean = np.asarray(X_train_combined.mean(axis=0), dtype=np.float32).ravel()
        Y_mean = np.asarray(y_train.mean(axis=0), dtype=np.float32).ravel()
        A = X_train_combined.T.dot(X_train_combined).toarray().astype(np.float32)
        A -= (N_tr * np.outer(X_mean, X_mean)).astype(np.float32)
        A = 0.5 * (A + A.T)
        A[np.diag_indices(D)] += np.float32(self.alpha)
        B = X_train_combined.T.dot(y_train).astype(np.float32)
        B -= (N_tr * np.outer(X_mean, Y_mean)).astype(np.float32)
        try:
            W = scipy.linalg.solve(A, B, assume_a='pos').astype(np.float32)
        except Exception:
            W = scipy.linalg.solve(A, B).astype(np.float32)
        b = (Y_mean.reshape(1, -1) - X_mean.reshape(1, -1) @ W).astype(np.float32)

        domain_val = X_val_df['domain'].fillna('').astype(str)
        X_val_tfidf = tfidf.transform(domain_val)
        X_val_lex = self._extract_lexical(domain_val)
        X_val_lex_scaled = scaler_lex.transform(X_val_lex).astype(np.float32)
        X_val_graph = self._extract_graph(X_val_df)
        X_val_graph_scaled = scaler_graph.transform(X_val_graph).astype(np.float32)
        X_val_combined = hstack([X_val_tfidf, X_val_lex_scaled, X_val_graph_scaled], format='csr', dtype=np.float32)
        P_val = np.asarray(X_val_combined.dot(W) + b, dtype=np.float32)

        b_overall, b_head, b_mid, b_tail = self._compute_metrics(P_val, y_val, head_mask, mid_mask, tail_mask)

        results = [
            {
                'configuration': 'uncalibrated_baseline (gamma=0.0)',
                'overall_recall_at_10': f"{b_overall:.4f}",
                'head_recall': f"{b_head:.4f}",
                'mid_recall': f"{b_mid:.4f}",
                'tail_recall': f"{b_tail:.4f}",
                'notes': f"Strata counts: {n_head} head, {n_mid} mid, {n_tail} tail",
            }
        ]

        for gamma in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            P_calib = P_val - gamma * priors.reshape(1, -1)
            c_overall, c_head, c_mid, c_tail = self._compute_metrics(P_calib, y_val, head_mask, mid_mask, tail_mask)
            results.append({
                'configuration': f'subtraction_gamma_{gamma:.2f}',
                'overall_recall_at_10': f"{c_overall:.4f}",
                'head_recall': f"{c_head:.4f}",
                'mid_recall': f"{c_mid:.4f}",
                'tail_recall': f"{c_tail:.4f}",
                'notes': f"Recall diff vs baseline: {c_overall - b_overall:+.4f}",
            })

        for gamma in [0.05, 0.10, 0.15, 0.20]:
            scale = np.power(np.maximum(priors, 1e-5), gamma).reshape(1, -1)
            P_scaled = P_val / scale
            s_overall, s_head, s_mid, s_tail = self._compute_metrics(P_scaled, y_val, head_mask, mid_mask, tail_mask)
            results.append({
                'configuration': f'power_scale_gamma_{gamma:.2f}',
                'overall_recall_at_10': f"{s_overall:.4f}",
                'head_recall': f"{s_head:.4f}",
                'mid_recall': f"{s_mid:.4f}",
                'tail_recall': f"{s_tail:.4f}",
                'notes': f"Recall diff vs baseline: {s_overall - b_overall:+.4f}",
            })

        self.summary_df_ = pd.DataFrame(results)
        return self

    def transform(self, X):
        return self.summary_df_


def build():
    tracking_graph = skrub.as_data_op(TRACKING_GRAPH_PATH).skb.apply_func(pd.read_parquet, columns=['domain_id', 'tracker_id'])
    domains = skrub.as_data_op(DOMAINS_PATH).skb.apply_func(pd.read_parquet)
    link_graph = skrub.as_data_op(LINK_GRAPH_PATH).skb.apply_func(pd.read_parquet, columns=['source_domain_id', 'target_domain_id'])
    trackers = skrub.as_data_op(TRACKERS_PATH).skb.apply_func(pd.read_csv, sep='\t')
    target_domains = skrub.as_data_op(TARGET_PATH).skb.apply_func(pd.read_csv, sep='\t')
    url_classification = skrub.as_data_op(URL_CLASSIFICATION_PATH).skb.apply_func(pd.read_csv, usecols=['url', 'category'])
    freedom_of_the_press = skrub.as_data_op(FREEDOM_OF_THE_PRESS_PATH).skb.apply_func(
        pd.read_csv, sep=None, engine='python', encoding='utf-8-sig'
    )

    tracker_counts = tracking_graph.groupby('domain_id', as_index=False).agg({'tracker_id': 'count'}).rename(columns={'tracker_id': 'tracker_count'})
    valid_domains = tracker_counts[(tracker_counts['tracker_count'] >= 2) & (tracker_counts['tracker_count'] <= 20)]
    sampled_domains = valid_domains.sample(n=50000, random_state=42).sort_values('domain_id').reset_index(drop=True)
    sampled_train = sampled_domains[['domain_id']].merge(domains, on='domain_id', how='left').sort_values('domain_id').reset_index(drop=True)
    sampled_tracking = sampled_domains[['domain_id']].merge(tracking_graph, on='domain_id', how='inner')
    target_df = target_domains[['domain_id']].merge(domains, on='domain_id', how='left')

    url_clean = url_classification.assign(
        clean_domain=url_classification['url'].str.replace('^https?://', '', regex=True).str.split('/').str[0].str.split(':').str[0].str.replace('^www\\.', '', regex=True).str.lower()
    )
    url_clean = url_clean[['clean_domain', 'category']].drop_duplicates(subset=['clean_domain'])

    target_doms = target_df[['domain_id', 'domain']].assign(clean_domain=target_df['domain'].str.lower())
    train_doms = sampled_train[['domain_id', 'domain']].assign(clean_domain=sampled_train['domain'].str.lower())

    target_url_joined = target_doms.merge(
        url_clean, on='clean_domain', how='left'
    )[['domain_id', 'category']].assign(split='target', tracker_id=-1)

    train_url_joined = train_doms.merge(
        url_clean, on='clean_domain', how='left'
    )[['domain_id', 'category']].assign(split='train')

    train_url_with_trackers = train_url_joined.merge(
        sampled_tracking[['domain_id', 'tracker_id']], on='domain_id', how='left'
    )
    train_url_with_trackers = train_url_with_trackers.assign(
        tracker_id=train_url_with_trackers['tracker_id'].fillna(-1)
    )

    combined_url = train_url_with_trackers[
        ['domain_id', 'category', 'split', 'tracker_id']
    ].skb.concat([
        target_url_joined[['domain_id', 'category', 'split', 'tracker_id']]
    ], axis=0).reset_index(drop=True)

    url_study_op = combined_url.skb.apply(UrlClassificationStudyTransformer())

    fotp_renamed = freedom_of_the_press.rename(columns={
        'TLD': 'tld',
        ' tld': 'tld',
        'tld ': 'tld',
        '\ufefftld': 'tld',
        'Country': 'country',
        ' country': 'country',
        'country ': 'country',
        'Freedom of the Press': 'freedom_of_the_press',
        'Freedom of the press': 'freedom_of_the_press',
        'Freedom_of_the_press': 'freedom_of_the_press',
        'Freedom_of_the_Press': 'freedom_of_the_press',
        'freedom of the press': 'freedom_of_the_press',
        ' freedom_of_the_press': 'freedom_of_the_press',
        'freedom_of_the_press ': 'freedom_of_the_press',
        'Score': 'freedom_of_the_press',
        'score': 'freedom_of_the_press',
    })
    fotp_clean = fotp_renamed.assign(
        tld=fotp_renamed['tld'].astype('string').str.lstrip('.').str.lower().str.strip()
    ).drop_duplicates(subset=['tld'])

    target_tld = target_df[['domain_id', 'domain']].assign(
        tld=target_df['domain'].str.split('.').str[-1].str.lower()
    )
    train_tld = sampled_train[['domain_id', 'domain']].assign(
        tld=sampled_train['domain'].str.split('.').str[-1].str.lower()
    )

    target_tld_joined = target_tld[['domain_id', 'tld']].merge(
        fotp_clean[['tld', 'freedom_of_the_press']], on='tld', how='left'
    ).assign(split='target', tracker_id=-1)

    train_tld_joined = train_tld[['domain_id', 'tld']].merge(
        fotp_clean[['tld', 'freedom_of_the_press']], on='tld', how='left'
    ).assign(split='train')

    train_tld_trackers = train_tld_joined.merge(
        sampled_tracking[['domain_id', 'tracker_id']], on='domain_id', how='left'
    )
    train_tld_trackers = train_tld_trackers.assign(
        tracker_id=train_tld_trackers['tracker_id'].fillna(-1)
    )

    combined_tld = train_tld_trackers[
        ['domain_id', 'tld', 'freedom_of_the_press', 'split', 'tracker_id']
    ].skb.concat([
        target_tld_joined[['domain_id', 'tld', 'freedom_of_the_press', 'split', 'tracker_id']]
    ], axis=0).reset_index(drop=True)

    tld_study_op = combined_tld.skb.apply(TldPressFreedomStudyTransformer())

    tracker_edges = link_graph.merge(
        trackers[['tracking_domain_id', 'tracker_id', 'company']],
        left_on='target_domain_id',
        right_on='tracking_domain_id',
        how='inner'
    )
    target_tracker_links = target_df[['domain_id']].merge(
        tracker_edges[['source_domain_id', 'tracker_id', 'company']],
        left_on='domain_id',
        right_on='source_domain_id',
        how='inner'
    ).assign(split='target', is_true_tracker=-1)

    train_tracker_links = sampled_train[['domain_id']].merge(
        tracker_edges[['source_domain_id', 'tracker_id', 'company']],
        left_on='domain_id',
        right_on='source_domain_id',
        how='inner'
    ).assign(split='train')
    train_tracker_links = train_tracker_links.merge(
        sampled_tracking.assign(is_true_tracker=1)[['domain_id', 'tracker_id', 'is_true_tracker']],
        on=['domain_id', 'tracker_id'],
        how='left'
    )
    train_tracker_links = train_tracker_links.assign(
        is_true_tracker=train_tracker_links['is_true_tracker'].fillna(0)
    )

    combined_links = train_tracker_links[
        ['domain_id', 'tracker_id', 'company', 'split', 'is_true_tracker']
    ].skb.concat([
        target_tracker_links[['domain_id', 'tracker_id', 'company', 'split', 'is_true_tracker']]
    ], axis=0).reset_index(drop=True)

    links_study_op = combined_links.skb.apply(TrackerHyperlinksStudyTransformer())

    in_counts = link_graph.groupby('target_domain_id', as_index=False).agg({'source_domain_id': 'count'}).rename(columns={'target_domain_id': 'domain_id', 'source_domain_id': 'in_degree'})
    out_counts = link_graph.groupby('source_domain_id', as_index=False).agg({'target_domain_id': 'count'}).rename(columns={'source_domain_id': 'domain_id', 'target_domain_id': 'out_degree'})
    X_graph = sampled_train.merge(in_counts, on='domain_id', how='left').merge(out_counts, on='domain_id', how='left').sort_values('domain_id').reset_index(drop=True)

    sampled_tracking_val = sampled_tracking.assign(val=1)
    pivot = sampled_tracking_val.pivot_table(index='domain_id', columns='tracker_id', values='val', fill_value=0)
    all_tracker_ids = list(range(355))
    pivot = pivot.reindex(columns=all_tracker_ids, fill_value=0)
    pivot = pivot.rename(columns={i: f't_{i}' for i in range(355)})
    y_matrix = pivot.reset_index(drop=True).astype('int8')

    strata_study_op = X_graph.skb.apply(TrackerStrataCalibrationTransformer(alpha=100.0, max_features=10000), y=y_matrix)

    return {
        'url_classification_study': url_study_op,
        'tld_and_press_freedom_study': tld_study_op,
        'tracker_hyperlinks_study': links_study_op,
        'tracker_strata_and_calibration_study': strata_study_op,
    }