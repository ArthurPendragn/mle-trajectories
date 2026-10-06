import numpy as np
import pandas as pd
import scipy.sparse as sp
import skrub
from sklearn.base import BaseEstimator, TransformerMixin

TRACKING_GRAPH_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/tracking_graph_train.parquet'
DOMAINS_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/domains.parquet'
LINK_GRAPH_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/link-graph.parquet'
TRACKERS_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/trackers.tsv'
TARGET_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/target.tsv'


class OneHopNeighborStudyTransformer(BaseEstimator, TransformerMixin):
    """
    Evaluates 1-hop web graph neighbor tracker profiles constructed from provably
    disjoint training domains (out-neighbors, in-neighbors, and combined).
    Computes domain coverage on target vs train, distribution parity, standalone
    precision and Recall@10 overall and on the 86.1% unlinked domain subset.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        N = len(df)
        is_target = df['is_target'].to_numpy().astype(bool)
        is_train = ~is_target

        direct_cols = [f'tl_{i}' for i in range(355)]
        out_cols = [f'out_{i}' for i in range(355)]
        in_cols = [f'in_{i}' for i in range(355)]
        comb_cols = [f'comb_{i}' for i in range(355)]
        true_cols = [f't_{i}' for i in range(355)]

        direct_arr = df[direct_cols].fillna(0.0).to_numpy(dtype=np.float32)
        out_arr = df[out_cols].fillna(0.0).to_numpy(dtype=np.float32)
        in_arr = df[in_cols].fillna(0.0).to_numpy(dtype=np.float32)
        comb_arr = df[comb_cols].fillna(0.0).to_numpy(dtype=np.float32)
        true_arr = df[true_cols].fillna(0.0).to_numpy(dtype=np.float32)

        y_train = true_arr[is_train]
        true_counts_train = np.maximum(y_train.sum(axis=1), 1.0)
        global_priors = y_train.mean(axis=0)

        has_direct = direct_arr.sum(axis=1) > 0
        has_out = out_arr.sum(axis=1) > 0
        has_in = in_arr.sum(axis=1) > 0
        has_comb = comb_arr.sum(axis=1) > 0

        target_total = is_target.sum()
        train_total = is_train.sum()

        target_cov_direct = float(has_direct[is_target].sum() / target_total)
        train_cov_direct = float(has_direct[is_train].sum() / train_total)

        target_cov_out = float(has_out[is_target].sum() / target_total)
        train_cov_out = float(has_out[is_train].sum() / train_total)

        target_cov_in = float(has_in[is_target].sum() / target_total)
        train_cov_in = float(has_in[is_train].sum() / train_total)

        target_cov_comb = float(has_comb[is_target].sum() / target_total)
        train_cov_comb = float(has_comb[is_train].sum() / train_total)

        # Coverage on unlinked domains
        unlinked_target = is_target & (~has_direct)
        unlinked_train = is_train & (~has_direct)
        target_unlinked_total = unlinked_target.sum()
        train_unlinked_total = unlinked_train.sum()

        target_unlinked_cov_comb = float(has_comb[unlinked_target].sum() / target_unlinked_total)
        train_unlinked_cov_comb = float(has_comb[unlinked_train].sum() / train_unlinked_total)

        # Precision calculation
        def get_precision(signal_sub, y_sub):
            pred_mask = signal_sub > 0
            hits = (pred_mask & (y_sub > 0)).sum()
            total_preds = pred_mask.sum()
            return float(hits / total_preds) if total_preds > 0 else 0.0

        prec_direct = get_precision(direct_arr[is_train], y_train)
        prec_out = get_precision(out_arr[is_train], y_train)
        prec_in = get_precision(in_arr[is_train], y_train)
        prec_comb = get_precision(comb_arr[is_train], y_train)

        # Precision on unlinked domains
        prec_out_unlinked = get_precision(out_arr[unlinked_train], true_arr[unlinked_train])
        prec_in_unlinked = get_precision(in_arr[unlinked_train], true_arr[unlinked_train])
        prec_comb_unlinked = get_precision(comb_arr[unlinked_train], true_arr[unlinked_train])

        # Standalone Recall@10 with prior backfill
        def get_recall(signal_sub, y_sub):
            n_sub = len(signal_sub)
            if n_sub == 0:
                return 0.0
            scores = signal_sub + 1e-4 * global_priors[None, :]
            top10 = np.argpartition(-scores, 10, axis=1)[:, :10]
            hits = y_sub[np.arange(n_sub)[:, None], top10].sum(axis=1)
            t_counts = np.maximum(y_sub.sum(axis=1), 1.0)
            return float(np.mean(hits / t_counts))

        prior_only_scores = np.tile(global_priors, (train_total, 1))
        top10_prior = np.argpartition(-prior_only_scores, 10, axis=1)[:, :10]
        hits_prior = y_train[np.arange(train_total)[:, None], top10_prior].sum(axis=1)
        rec_prior = float(np.mean(hits_prior / true_counts_train))

        rec_direct = get_recall(direct_arr[is_train], y_train)
        rec_out = get_recall(out_arr[is_train], y_train)
        rec_in = get_recall(in_arr[is_train], y_train)
        rec_comb = get_recall(comb_arr[is_train], y_train)

        # Recall on unlinked vs linked subsets
        y_unlinked_train = true_arr[unlinked_train]
        y_linked_train = true_arr[is_train & has_direct]

        rec_prior_unlinked = get_recall(np.zeros_like(y_unlinked_train), y_unlinked_train)
        rec_out_unlinked = get_recall(out_arr[unlinked_train], y_unlinked_train)
        rec_in_unlinked = get_recall(in_arr[unlinked_train], y_unlinked_train)
        rec_comb_unlinked = get_recall(comb_arr[unlinked_train], y_unlinked_train)

        records = [
            {'metric': 'direct_tracker_links_coverage', 'target_val': f'{target_cov_direct:.2%}', 'train_val': f'{train_cov_direct:.2%}', 'notes': 'Direct links Domain -> Tracker Hostname'},
            {'metric': 'out_neighbor_profile_coverage', 'target_val': f'{target_cov_out:.2%}', 'train_val': f'{train_cov_out:.2%}', 'notes': 'Disjoint out-neighbors with known trackers'},
            {'metric': 'in_neighbor_profile_coverage', 'target_val': f'{target_cov_in:.2%}', 'train_val': f'{train_cov_in:.2%}', 'notes': 'Disjoint in-neighbors with known trackers'},
            {'metric': 'combined_neighbor_coverage_overall', 'target_val': f'{target_cov_comb:.2%}', 'train_val': f'{train_cov_comb:.2%}', 'notes': 'Either out or in disjoint neighbor coverage'},
            {'metric': 'combined_neighbor_coverage_on_unlinked', 'target_val': f'{target_unlinked_cov_comb:.2%}', 'train_val': f'{train_unlinked_cov_comb:.2%}', 'notes': 'Coverage on domains lacking direct tracker links (86.1%)'},
            {'metric': 'empirical_precision_direct_links', 'target_val': 'N/A (unlabeled)', 'train_val': f'{prec_direct:.2%}', 'notes': 'Precision of direct tracker hyperlinks'},
            {'metric': 'empirical_precision_out_neighbors', 'target_val': 'N/A (unlabeled)', 'train_val': f'{prec_out:.2%}', 'notes': 'Precision of out-neighbor candidate trackers'},
            {'metric': 'empirical_precision_in_neighbors', 'target_val': 'N/A (unlabeled)', 'train_val': f'{prec_in:.2%}', 'notes': 'Precision of in-neighbor candidate trackers'},
            {'metric': 'empirical_precision_combined_neighbors', 'target_val': 'N/A (unlabeled)', 'train_val': f'{prec_comb:.2%}', 'notes': 'Precision of combined neighbor candidate trackers'},
            {'metric': 'empirical_precision_comb_on_unlinked', 'target_val': 'N/A (unlabeled)', 'train_val': f'{prec_comb_unlinked:.2%}', 'notes': 'Precision on 86.1% unlinked domain subset'},
            {'metric': 'overall_recall_at_10_prior_baseline', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_prior:.4f}', 'notes': 'Global frequency prior baseline Recall@10'},
            {'metric': 'overall_recall_at_10_direct_links', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_direct:.4f}', 'notes': 'Direct links + prior Recall@10 across all 50k domains'},
            {'metric': 'overall_recall_at_10_out_neighbors', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_out:.4f}', 'notes': 'Out-neighbor profile + prior Recall@10'},
            {'metric': 'overall_recall_at_10_in_neighbors', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_in:.4f}', 'notes': 'In-neighbor profile + prior Recall@10'},
            {'metric': 'overall_recall_at_10_combined_neighbors', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_comb:.4f}', 'notes': 'Combined neighbor profile + prior Recall@10'},
            {'metric': 'unlinked_domains_recall_prior_baseline', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_prior_unlinked:.4f}', 'notes': 'Prior Recall@10 on 86.1% unlinked domains'},
            {'metric': 'unlinked_domains_recall_out_neighbors', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_out_unlinked:.4f}', 'notes': 'Out-neighbors Recall@10 on unlinked domains'},
            {'metric': 'unlinked_domains_recall_in_neighbors', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_in_unlinked:.4f}', 'notes': 'In-neighbors Recall@10 on unlinked domains'},
            {'metric': 'unlinked_domains_recall_comb_neighbors', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_comb_unlinked:.4f}', 'notes': 'Combined neighbors Recall@10 on unlinked domains'},
        ]
        res_df = pd.DataFrame(records)
        return res_df.reindex(range(N))


class TwoHopTrackerLinkStudyTransformer(BaseEstimator, TransformerMixin):
    """
    Evaluates 2-hop hyperlink paths to tracker domains (Domain -> Intermediate -> Tracker).
    Computes reach on target vs train, reach specifically on domains lacking 1-hop links,
    empirical path precision, and standalone Recall@10.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        N = len(df)
        is_target = df['is_target'].to_numpy().astype(bool)
        is_train = ~is_target

        direct_cols = [f'tl_{i}' for i in range(355)]
        two_hop_cols = [f'th_{i}' for i in range(355)]
        true_cols = [f't_{i}' for i in range(355)]

        direct_arr = df[direct_cols].fillna(0.0).to_numpy(dtype=np.float32)
        th_arr = df[two_hop_cols].fillna(0.0).to_numpy(dtype=np.float32)
        true_arr = df[true_cols].fillna(0.0).to_numpy(dtype=np.float32)

        y_train = true_arr[is_train]
        global_priors = y_train.mean(axis=0)

        has_direct = direct_arr.sum(axis=1) > 0
        has_th = th_arr.sum(axis=1) > 0

        target_total = is_target.sum()
        train_total = is_train.sum()

        target_th_cov = float(has_th[is_target].sum() / target_total)
        train_th_cov = float(has_th[is_train].sum() / train_total)

        unlinked_target = is_target & (~has_direct)
        unlinked_train = is_train & (~has_direct)
        target_unlinked_total = unlinked_target.sum()
        train_unlinked_total = unlinked_train.sum()

        target_unlinked_th_cov = float(has_th[unlinked_target].sum() / target_unlinked_total)
        train_unlinked_th_cov = float(has_th[unlinked_train].sum() / train_unlinked_total)

        th_mask_train = th_arr[is_train] > 0
        hits_th = (th_mask_train & (y_train > 0)).sum()
        total_th_preds = th_mask_train.sum()
        prec_th = float(hits_th / total_th_preds) if total_th_preds > 0 else 0.0

        th_mask_unlinked = th_arr[unlinked_train] > 0
        hits_unlinked = (th_mask_unlinked & (true_arr[unlinked_train] > 0)).sum()
        total_unlinked = th_mask_unlinked.sum()
        prec_th_unlinked = float(hits_unlinked / total_unlinked) if total_unlinked > 0 else 0.0

        def get_recall(signal_sub, y_sub):
            n_sub = len(signal_sub)
            if n_sub == 0:
                return 0.0
            scores = signal_sub + 1e-4 * global_priors[None, :]
            top10 = np.argpartition(-scores, 10, axis=1)[:, :10]
            hits = y_sub[np.arange(n_sub)[:, None], top10].sum(axis=1)
            t_counts = np.maximum(y_sub.sum(axis=1), 1.0)
            return float(np.mean(hits / t_counts))

        rec_th_overall = get_recall(th_arr[is_train], y_train)
        rec_th_unlinked = get_recall(th_arr[unlinked_train], true_arr[unlinked_train])

        # Combined 1-hop + 2-hop signals
        comb_direct_th = direct_arr + 0.3 * th_arr
        rec_direct_plus_th = get_recall(comb_direct_th[is_train], y_train)

        records = [
            {'metric': 'two_hop_reach_overall', 'target_val': f'{target_th_cov:.2%}', 'train_val': f'{train_th_cov:.2%}', 'notes': 'Domains with >= 1 2-hop path (D -> M -> Tracker)'},
            {'metric': 'two_hop_reach_on_unlinked_domains', 'target_val': f'{target_unlinked_th_cov:.2%}', 'train_val': f'{train_unlinked_th_cov:.2%}', 'notes': '2-hop reach on domains lacking direct 1-hop links'},
            {'metric': 'two_hop_empirical_precision_overall', 'target_val': 'N/A (unlabeled)', 'train_val': f'{prec_th:.2%}', 'notes': 'Fraction of 2-hop tracker paths that are true trackers'},
            {'metric': 'two_hop_empirical_precision_unlinked', 'target_val': 'N/A (unlabeled)', 'train_val': f'{prec_th_unlinked:.2%}', 'notes': '2-hop precision on unlinked domain subset'},
            {'metric': 'two_hop_overall_recall_at_10', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_th_overall:.4f}', 'notes': '2-hop paths + prior Recall@10 on all domains'},
            {'metric': 'two_hop_unlinked_recall_at_10', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_th_unlinked:.4f}', 'notes': '2-hop paths + prior Recall@10 on unlinked domains'},
            {'metric': 'direct_plus_two_hop_recall_at_10', 'target_val': 'N/A (unlabeled)', 'train_val': f'{rec_direct_plus_th:.4f}', 'notes': 'Combined 1-hop + 2-hop path Recall@10'},
        ]
        res_df = pd.DataFrame(records)
        return res_df.reindex(range(N))


class TrackerCooccurrenceStudyTransformer(BaseEstimator, TransformerMixin):
    """
    Computes tracker co-occurrence transition statistics across all 355 trackers,
    including pairwise support, conditional probabilities, and lift ratios.
    Identifies high-lift tracker syndicates (e.g. Russian web stack, Google suite).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        N_rows = len(df)
        is_train = ~df['is_target'].to_numpy().astype(bool)

        true_cols = [f't_{i}' for i in range(355)]
        y_train = df[true_cols].fillna(0.0).to_numpy(dtype=np.float32)[is_train]
        N_train = len(y_train)

        # Co-occurrence matrix C = Y^T Y (355 x 355)
        C = y_train.T.dot(y_train)
        diag = np.diag(C)

        pairs = []
        for i in range(355):
            for j in range(i + 1, 355):
                n_ij = C[i, j]
                if n_ij >= 100:  # Support threshold
                    n_i = diag[i]
                    n_j = diag[j]
                    lift = (n_ij * N_train) / (n_i * n_j)
                    p_j_given_i = n_ij / n_i
                    p_i_given_j = n_ij / n_j
                    pairs.append({
                        'tracker_pair': f't_{i} <-> t_{j}',
                        'support': int(n_ij),
                        'lift': float(lift),
                        'p_cond_max': float(max(p_j_given_i, p_i_given_j)),
                        'notes': f'Support: {int(n_ij)}, Lift: {lift:.2f}x'
                    })

        pairs_df = pd.DataFrame(pairs).sort_values('lift', ascending=False).reset_index(drop=True)
        top_pairs = pairs_df.head(20)
        return top_pairs.reindex(range(N_rows))


class SubgroupPerformanceStudyTransformer(BaseEstimator, TransformerMixin):
    """
    Synthesizes empirical performance of all candidate signals across subgroups:
    - All domains (50,000)
    - Linked domains (6,982, 13.9%)
    - Unlinked domains (43,018, 86.1%)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        N = len(df)
        is_target = df['is_target'].to_numpy().astype(bool)
        is_train = ~is_target

        direct_cols = [f'tl_{i}' for i in range(355)]
        out_cols = [f'out_{i}' for i in range(355)]
        in_cols = [f'in_{i}' for i in range(355)]
        comb_cols = [f'comb_{i}' for i in range(355)]
        th_cols = [f'th_{i}' for i in range(355)]
        true_cols = [f't_{i}' for i in range(355)]

        direct_arr = df[direct_cols].fillna(0.0).to_numpy(dtype=np.float32)
        out_arr = df[out_cols].fillna(0.0).to_numpy(dtype=np.float32)
        in_arr = df[in_cols].fillna(0.0).to_numpy(dtype=np.float32)
        comb_arr = df[comb_cols].fillna(0.0).to_numpy(dtype=np.float32)
        th_arr = df[th_cols].fillna(0.0).to_numpy(dtype=np.float32)
        true_arr = df[true_cols].fillna(0.0).to_numpy(dtype=np.float32)

        y_train = true_arr[is_train]
        global_priors = y_train.mean(axis=0)

        has_direct = direct_arr.sum(axis=1) > 0
        unlinked_train = is_train & (~has_direct)
        linked_train = is_train & has_direct

        y_unlinked = true_arr[unlinked_train]
        y_linked = true_arr[linked_train]

        def eval_signal(signal_full):
            # Overall recall
            scores_all = signal_full[is_train] + 1e-4 * global_priors[None, :]
            top10_all = np.argpartition(-scores_all, 10, axis=1)[:, :10]
            hits_all = y_train[np.arange(len(y_train))[:, None], top10_all].sum(axis=1)
            rec_all = float(np.mean(hits_all / np.maximum(y_train.sum(axis=1), 1.0)))

            # Unlinked recall
            scores_unlinked = signal_full[unlinked_train] + 1e-4 * global_priors[None, :]
            top10_unlinked = np.argpartition(-scores_unlinked, 10, axis=1)[:, :10]
            hits_unlinked = y_unlinked[np.arange(len(y_unlinked))[:, None], top10_unlinked].sum(axis=1)
            rec_unlinked = float(np.mean(hits_unlinked / np.maximum(y_unlinked.sum(axis=1), 1.0)))

            # Linked recall
            scores_linked = signal_full[linked_train] + 1e-4 * global_priors[None, :]
            top10_linked = np.argpartition(-scores_linked, 10, axis=1)[:, :10]
            hits_linked = y_linked[np.arange(len(y_linked))[:, None], top10_linked].sum(axis=1)
            rec_linked = float(np.mean(hits_linked / np.maximum(y_linked.sum(axis=1), 1.0)))

            # Coverage
            has_sig = signal_full.sum(axis=1) > 0
            cov_target = float(has_sig[is_target].sum() / is_target.sum())
            cov_train = float(has_sig[is_train].sum() / is_train.sum())

            return cov_target, cov_train, rec_all, rec_unlinked, rec_linked

        # 1. Global prior baseline
        cov_tgt_p, cov_trn_p, rec_all_p, rec_unl_p, rec_lnk_p = eval_signal(np.zeros_like(direct_arr))

        # 2. Direct 1-hop links
        cov_tgt_d, cov_trn_d, rec_all_d, rec_unl_d, rec_lnk_d = eval_signal(direct_arr)

        # 3. 1-hop out-neighbors
        cov_tgt_o, cov_trn_o, rec_all_o, rec_unl_o, rec_lnk_o = eval_signal(out_arr)

        # 4. 1-hop in-neighbors
        cov_tgt_i, cov_trn_i, rec_all_i, rec_unl_i, rec_lnk_i = eval_signal(in_arr)

        # 5. 1-hop combined neighbors
        cov_tgt_c, cov_trn_c, rec_all_c, rec_unl_c, rec_lnk_c = eval_signal(comb_arr)

        # 6. 2-hop paths
        cov_tgt_th, cov_trn_th, rec_all_th, rec_unl_th, rec_lnk_th = eval_signal(th_arr)

        # 7. Hybrid: Direct + Combined Neighbors
        hybrid_1 = direct_arr + 0.1 * comb_arr
        cov_tgt_h1, cov_trn_h1, rec_all_h1, rec_unl_h1, rec_lnk_h1 = eval_signal(hybrid_1)

        # 8. Full Hybrid: Direct + Combined Neighbors + 2-Hop
        hybrid_full = direct_arr + 0.1 * comb_arr + 0.05 * th_arr
        cov_tgt_hf, cov_trn_hf, rec_all_hf, rec_unl_hf, rec_lnk_hf = eval_signal(hybrid_full)

        records = [
            {'signal_name': 'global_prior_baseline', 'target_cov': f'{cov_tgt_p:.1%}', 'train_cov': f'{cov_trn_p:.1%}', 'all_recall_10': f'{rec_all_p:.4f}', 'unlinked_recall_10': f'{rec_unl_p:.4f}', 'linked_recall_10': f'{rec_lnk_p:.4f}', 'notes': 'Frequency prior baseline'},
            {'signal_name': 'direct_tracker_links', 'target_cov': f'{cov_tgt_d:.1%}', 'train_cov': f'{cov_trn_d:.1%}', 'all_recall_10': f'{rec_all_d:.4f}', 'unlinked_recall_10': f'{rec_unl_d:.4f}', 'linked_recall_10': f'{rec_lnk_d:.4f}', 'notes': 'Direct 1-hop hyperlinks'},
            {'signal_name': '1hop_out_neighbors', 'target_cov': f'{cov_tgt_o:.1%}', 'train_cov': f'{cov_trn_o:.1%}', 'all_recall_10': f'{rec_all_o:.4f}', 'unlinked_recall_10': f'{rec_unl_o:.4f}', 'linked_recall_10': f'{rec_lnk_o:.4f}', 'notes': 'Disjoint out-neighbor trackers'},
            {'signal_name': '1hop_in_neighbors', 'target_cov': f'{cov_tgt_i:.1%}', 'train_cov': f'{cov_trn_i:.1%}', 'all_recall_10': f'{rec_all_i:.4f}', 'unlinked_recall_10': f'{rec_unl_i:.4f}', 'linked_recall_10': f'{rec_lnk_i:.4f}', 'notes': 'Disjoint in-neighbor trackers'},
            {'signal_name': '1hop_combined_neighbors', 'target_cov': f'{cov_tgt_c:.1%}', 'train_cov': f'{cov_trn_c:.1%}', 'all_recall_10': f'{rec_all_c:.4f}', 'unlinked_recall_10': f'{rec_unl_c:.4f}', 'linked_recall_10': f'{rec_lnk_c:.4f}', 'notes': 'Disjoint out + in neighbors'},
            {'signal_name': '2hop_tracker_paths', 'target_cov': f'{cov_tgt_th:.1%}', 'train_cov': f'{cov_trn_th:.1%}', 'all_recall_10': f'{rec_all_th:.4f}', 'unlinked_recall_10': f'{rec_unl_th:.4f}', 'linked_recall_10': f'{rec_lnk_th:.4f}', 'notes': '2-hop paths (D -> M -> Tracker)'},
            {'signal_name': 'hybrid_direct_plus_neighbors', 'target_cov': f'{cov_tgt_h1:.1%}', 'train_cov': f'{cov_trn_h1:.1%}', 'all_recall_10': f'{rec_all_h1:.4f}', 'unlinked_recall_10': f'{rec_unl_h1:.4f}', 'linked_recall_10': f'{rec_lnk_h1:.4f}', 'notes': 'Direct links + 1-hop neighbors'},
            {'signal_name': 'full_hybrid_direct_neighbors_2hop', 'target_cov': f'{cov_tgt_hf:.1%}', 'train_cov': f'{cov_trn_hf:.1%}', 'all_recall_10': f'{rec_all_hf:.4f}', 'unlinked_recall_10': f'{rec_unl_hf:.4f}', 'linked_recall_10': f'{rec_lnk_hf:.4f}', 'notes': 'Direct links + neighbors + 2-hop'},
        ]
        res_df = pd.DataFrame(records)
        return res_df.reindex(range(N))


def build():
    # 1. Base tracking graph and population
    tracking_graph = skrub.as_data_op(TRACKING_GRAPH_PATH).skb.apply_func(
        pd.read_parquet, columns=['domain_id', 'tracker_id']
    )
    tracker_counts = (
        tracking_graph.groupby('domain_id', as_index=False)
        .agg({'tracker_id': 'count'})
        .rename(columns={'tracker_id': 'tracker_count'})
    )
    valid_domains = tracker_counts[
        (tracker_counts['tracker_count'] >= 2) & (tracker_counts['tracker_count'] <= 20)
    ]
    sampled_domains = (
        valid_domains.sample(n=50000, random_state=42)
        .sort_values('domain_id')
        .reset_index(drop=True)
    )

    target_domains = skrub.as_data_op(TARGET_PATH).skb.apply_func(pd.read_csv, sep='\t')

    train_eval = sampled_domains[['domain_id']].assign(is_target=0)
    target_eval = target_domains[['domain_id']].assign(is_target=1)
    eval_domains = train_eval.skb.concat([target_eval], axis=0).reset_index(drop=True)

    # 2. Trackers metadata
    trackers = skrub.as_data_op(TRACKERS_PATH).skb.apply_func(
        pd.read_csv,
        sep='\t',
        usecols=['tracking_domain_id', 'tracker_id'],
        dtype={'tracking_domain_id': 'int64', 'tracker_id': 'int32'},
    )

    # 3. Link graph reads
    link_graph = skrub.as_data_op(LINK_GRAPH_PATH).skb.apply_func(
        pd.read_parquet, columns=['source_domain_id', 'target_domain_id']
    )

    # 4. Direct 1-hop tracker links: Domain -> Tracker Domain
    tracker_links = (
        link_graph.merge(
            trackers, left_on='target_domain_id', right_on='tracking_domain_id', how='inner'
        )[['source_domain_id', 'tracker_id']]
        .drop_duplicates()
        .rename(columns={'source_domain_id': 'domain_id'})
    )

    domain_direct_links = eval_domains[['domain_id']].merge(tracker_links, on='domain_id', how='inner').assign(val=1)
    all_tracker_ids = list(range(355))
    tl_rename = {i: f'tl_{i}' for i in range(355)}
    pivot_direct = (
        domain_direct_links.pivot_table(index='domain_id', columns='tracker_id', values='val', fill_value=0)
        .reindex(columns=all_tracker_ids, fill_value=0)
        .rename(columns=tl_rename)
        .reset_index()
    )

    # 5. Out-edges and In-edges for eval_domains
    eval_out = (
        eval_domains[['domain_id']]
        .merge(link_graph, left_on='domain_id', right_on='source_domain_id', how='inner')[
            ['domain_id', 'target_domain_id']
        ]
        .rename(columns={'target_domain_id': 'neighbor_id'})
    )

    eval_in = (
        eval_domains[['domain_id']]
        .merge(link_graph, left_on='domain_id', right_on='target_domain_id', how='inner')[
            ['domain_id', 'source_domain_id']
        ]
        .rename(columns={'source_domain_id': 'neighbor_id'})
    )

    # 6. Disjoint neighbors (exclude self-loops and domains in locked 50k train sample)
    locked_train_indicator = sampled_domains[['domain_id']].assign(in_locked=1)

    eval_out_clean = eval_out[eval_out['domain_id'] != eval_out['neighbor_id']]
    eval_out_clean = eval_out_clean.merge(locked_train_indicator, left_on='neighbor_id', right_on='domain_id', how='left')
    eval_out_disjoint = eval_out_clean[eval_out_clean['in_locked'].isna()][['domain_id_x', 'neighbor_id']].rename(
        columns={'domain_id_x': 'domain_id'}
    )

    eval_in_clean = eval_in[eval_in['domain_id'] != eval_in['neighbor_id']]
    eval_in_clean = eval_in_clean.merge(locked_train_indicator, left_on='neighbor_id', right_on='domain_id', how='left')
    eval_in_disjoint = eval_in_clean[eval_in_clean['in_locked'].isna()][['domain_id_x', 'neighbor_id']].rename(
        columns={'domain_id_x': 'domain_id'}
    )

    # Lookup neighbor trackers in tracking_graph
    out_neighbor_trackers = eval_out_disjoint.merge(
        tracking_graph, left_on='neighbor_id', right_on='domain_id', how='inner'
    )[['domain_id_x', 'tracker_id']].rename(columns={'domain_id_x': 'domain_id'})

    in_neighbor_trackers = eval_in_disjoint.merge(
        tracking_graph, left_on='neighbor_id', right_on='domain_id', how='inner'
    )[['domain_id_x', 'tracker_id']].rename(columns={'domain_id_x': 'domain_id'})

    # Pivot out and in neighbor counts
    out_rename = {i: f'out_{i}' for i in range(355)}
    pivot_out = (
        out_neighbor_trackers.assign(val=1)
        .pivot_table(index='domain_id', columns='tracker_id', values='val', aggfunc='sum', fill_value=0)
        .reindex(columns=all_tracker_ids, fill_value=0)
        .rename(columns=out_rename)
        .reset_index()
    )

    in_rename = {i: f'in_{i}' for i in range(355)}
    pivot_in = (
        in_neighbor_trackers.assign(val=1)
        .pivot_table(index='domain_id', columns='tracker_id', values='val', aggfunc='sum', fill_value=0)
        .reindex(columns=all_tracker_ids, fill_value=0)
        .rename(columns=in_rename)
        .reset_index()
    )

    # Combined neighbors
    all_neighbor_trackers = out_neighbor_trackers.skb.concat([in_neighbor_trackers], axis=0)
    comb_rename = {i: f'comb_{i}' for i in range(355)}
    pivot_comb = (
        all_neighbor_trackers.assign(val=1)
        .pivot_table(index='domain_id', columns='tracker_id', values='val', aggfunc='sum', fill_value=0)
        .reindex(columns=all_tracker_ids, fill_value=0)
        .rename(columns=comb_rename)
        .reset_index()
    )

    # 7. 2-Hop tracker links: Domain -> Intermediate Neighbor -> Tracker
    eval_two_hop = (
        eval_out.merge(tracker_links, left_on='neighbor_id', right_on='domain_id', how='inner')[
            ['domain_id_x', 'tracker_id']
        ]
        .drop_duplicates()
        .rename(columns={'domain_id_x': 'domain_id'})
    )

    th_rename = {i: f'th_{i}' for i in range(355)}
    pivot_th = (
        eval_two_hop.assign(val=1)
        .pivot_table(index='domain_id', columns='tracker_id', values='val', fill_value=0)
        .reindex(columns=all_tracker_ids, fill_value=0)
        .rename(columns=th_rename)
        .reset_index()
    )

    # 8. True labels y for locked train domains
    sampled_tracking = sampled_domains[['domain_id']].merge(tracking_graph, on='domain_id', how='inner').assign(val=1)
    true_rename = {i: f't_{i}' for i in range(355)}
    pivot_true = (
        sampled_tracking.pivot_table(index='domain_id', columns='tracker_id', values='val', fill_value=0)
        .reindex(columns=all_tracker_ids, fill_value=0)
        .rename(columns=true_rename)
        .reset_index()
    )

    # 9. Merge all tables onto eval_domains
    X_study = (
        eval_domains.merge(pivot_direct, on='domain_id', how='left')
        .merge(pivot_out, on='domain_id', how='left')
        .merge(pivot_in, on='domain_id', how='left')
        .merge(pivot_comb, on='domain_id', how='left')
        .merge(pivot_th, on='domain_id', how='left')
        .merge(pivot_true, on='domain_id', how='left')
    )

    # 10. Instantiate study transformers and apply
    out_one_hop = X_study.skb.apply(OneHopNeighborStudyTransformer())
    out_two_hop = X_study.skb.apply(TwoHopTrackerLinkStudyTransformer())
    out_cooccurrence = X_study.skb.apply(TrackerCooccurrenceStudyTransformer())
    out_subgroups = X_study.skb.apply(SubgroupPerformanceStudyTransformer())

    return {
        'one_hop_neighbor_study': out_one_hop,
        'two_hop_tracker_link_study': out_two_hop,
        'tracker_cooccurrence_study': out_cooccurrence,
        'subgroup_performance_comparison': out_subgroups,
    }