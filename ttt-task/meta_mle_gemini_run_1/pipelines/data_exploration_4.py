"""
Exploration: Subgroup Error Distribution, Hub-Damped Authority Weighting,
Lexical/Regional Indicators on Graph-Isolated Domains, and Tracker Co-Occurrence Transitions.

Evaluates using fine-grained DataOps:
1. Recall@10, true tracker distributions, and domain counts across graph-isolated vs connected domains.
2. Empirical precision of unweighted (uniform count) vs hub-damped (inverse degree) neighbor tracker evidence.
3. Distribution and regional TLD profiles of graph-isolated domains.
4. Tracker co-occurrence pairs, conditional transition probabilities, lift, and excess probabilities under thresholding.
"""
import numpy as np
import pandas as pd
import skrub

TRACKING_GRAPH_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/tracking_graph_train.parquet'
DOMAINS_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/domains.parquet'
LINK_GRAPH_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/link-graph.parquet'
TRACKERS_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/trackers.tsv'
FREEDOM_PRESS_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/freedom-of-the-press.csv'
TARGET_PATH = '/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/target.tsv'

def build():
    tracking_graph = skrub.as_data_op(TRACKING_GRAPH_PATH).skb.apply_func(pd.read_parquet, columns=['domain_id', 'tracker_id'])
    tracker_counts = tracking_graph.groupby('domain_id', as_index=False).agg({'tracker_id': 'count'}).rename(columns={'tracker_id': 'tracker_count'})
    valid_train = tracker_counts[(tracker_counts['tracker_count'] >= 2) & (tracker_counts['tracker_count'] <= 20)]
    sampled_train = valid_train.sample(n=50000, random_state=42).sort_values('domain_id').reset_index(drop=True)

    link_graph = skrub.as_data_op(LINK_GRAPH_PATH).skb.apply_func(pd.read_parquet, columns=['source_domain_id', 'target_domain_id'])
    trackers = skrub.as_data_op(TRACKERS_PATH).skb.apply_func(pd.read_csv, sep='\t', usecols=['tracking_domain_id', 'tracker_id'], dtype={'tracking_domain_id': 'int64', 'tracker_id': 'int32'})

    tracker_links = link_graph.merge(trackers, left_on='target_domain_id', right_on='tracking_domain_id', how='inner')[['source_domain_id', 'tracker_id']].drop_duplicates().rename(columns={'source_domain_id': 'domain_id'})
    train_with_links = sampled_train.merge(tracker_links[['domain_id']].drop_duplicates().assign(has_direct_link=1.0), on='domain_id', how='left')
    train_with_links = train_with_links.assign(has_direct_link=train_with_links['has_direct_link'].fillna(0.0))

    train_out = link_graph.merge(sampled_train[['domain_id']], left_on='source_domain_id', right_on='domain_id', how='inner')[['domain_id', 'target_domain_id']].drop_duplicates()
    train_out_anti = train_out.merge(sampled_train[['domain_id']].assign(is_sample=1), left_on='target_domain_id', right_on='domain_id', how='left')
    disjoint_train_out = train_out_anti[train_out_anti['is_sample'].isna()][['domain_id_x', 'target_domain_id']].rename(columns={'domain_id_x': 'domain_id'})
    train_out_with_trackers = disjoint_train_out.merge(tracking_graph, left_on='target_domain_id', right_on='domain_id', how='inner')[['domain_id_x']].drop_duplicates().rename(columns={'domain_id_x': 'domain_id'}).assign(has_out_neighbor=1.0)

    train_in = link_graph.merge(sampled_train[['domain_id']], left_on='target_domain_id', right_on='domain_id', how='inner')[['domain_id', 'source_domain_id']].drop_duplicates()
    train_in_anti = train_in.merge(sampled_train[['domain_id']].assign(is_sample=1), left_on='source_domain_id', right_on='domain_id', how='left')
    disjoint_train_in = train_in_anti[train_in_anti['is_sample'].isna()][['domain_id_x', 'source_domain_id']].rename(columns={'domain_id_x': 'domain_id'})
    train_in_with_trackers = disjoint_train_in.merge(tracking_graph, left_on='source_domain_id', right_on='domain_id', how='inner')[['domain_id_x']].drop_duplicates().rename(columns={'domain_id_x': 'domain_id'}).assign(has_in_neighbor=1.0)

    train_profile = train_with_links.merge(train_out_with_trackers, on='domain_id', how='left')
    train_profile = train_profile.assign(has_out_neighbor=train_profile['has_out_neighbor'].fillna(0.0))
    train_profile = train_profile.merge(train_in_with_trackers, on='domain_id', how='left')
    train_profile = train_profile.assign(has_in_neighbor=train_profile['has_in_neighbor'].fillna(0.0))

    is_conn = (train_profile['has_out_neighbor'] > 0.0) | (train_profile['has_in_neighbor'] > 0.0)
    train_profile = train_profile.assign(is_connected=is_conn.astype('int32'))
    train_profile = train_profile.assign(is_isolated=(1 - is_conn.astype('int32')))

    sampled_tracking = sampled_train[['domain_id']].merge(tracking_graph, on='domain_id', how='inner')
    top10_trackers = [129, 131, 104, 105, 292, 51, 8, 133, 274, 320]
    hits_tracking = sampled_tracking[sampled_tracking['tracker_id'].isin(top10_trackers)]
    hits_per_domain = hits_tracking.groupby('domain_id', as_index=False).agg({'tracker_id': 'count'}).rename(columns={'tracker_id': 'top10_hits'})

    train_profile = train_profile.merge(hits_per_domain, on='domain_id', how='left')
    train_profile = train_profile.assign(top10_hits=train_profile['top10_hits'].fillna(0.0))
    train_profile = train_profile.assign(prior_recall_10=train_profile['top10_hits'] / train_profile['tracker_count'])

    subgroup_stats = train_profile.groupby('is_connected', as_index=False).agg({
        'domain_id': 'count',
        'tracker_count': 'mean',
        'top10_hits': 'mean',
        'prior_recall_10': 'mean'
    }).rename(columns={'domain_id': 'domain_count', 'tracker_count': 'mean_true_trackers', 'prior_recall_10': 'mean_prior_recall_10'})
    subgroup_stats = subgroup_stats.assign(subgroup=subgroup_stats['is_connected'].map({1: 'connected_domains_with_neighbors', 0: 'graph_isolated_domains'}))

    link_stats = train_profile.groupby('has_direct_link', as_index=False).agg({
        'domain_id': 'count',
        'tracker_count': 'mean',
        'top10_hits': 'mean',
        'prior_recall_10': 'mean'
    }).rename(columns={'domain_id': 'domain_count', 'tracker_count': 'mean_true_trackers', 'prior_recall_10': 'mean_prior_recall_10'})
    link_stats = link_stats.assign(subgroup=link_stats['has_direct_link'].map({1.0: 'domains_with_direct_links', 0.0: 'domains_without_direct_links'}))

    cols_out1 = ['subgroup', 'domain_count', 'mean_true_trackers', 'top10_hits', 'mean_prior_recall_10']
    out1 = subgroup_stats[cols_out1].skb.concat([link_stats[cols_out1]], axis=0).reset_index(drop=True)

    in_deg = link_graph.groupby('target_domain_id', as_index=False).agg({'source_domain_id': 'count'}).rename(columns={'target_domain_id': 'domain_id', 'source_domain_id': 'in_degree'})
    out_deg = link_graph.groupby('source_domain_id', as_index=False).agg({'target_domain_id': 'count'}).rename(columns={'source_domain_id': 'domain_id', 'target_domain_id': 'out_degree'})
    deg_df = in_deg.merge(out_deg, on='domain_id', how='outer')
    deg_df = deg_df.assign(total_degree=deg_df['in_degree'].fillna(0.0) + deg_df['out_degree'].fillna(0.0))

    disjoint_edges_deg = disjoint_train_out.merge(deg_df[['domain_id', 'total_degree']], left_on='target_domain_id', right_on='domain_id', how='left')
    disjoint_edges_deg = disjoint_edges_deg.assign(total_degree=disjoint_edges_deg['total_degree'].fillna(0.0))
    disjoint_edges_deg = disjoint_edges_deg.assign(w_uniform=1.0)
    disjoint_edges_deg = disjoint_edges_deg.assign(w_inv_deg=1.0 / (disjoint_edges_deg['total_degree'] + 1.0))

    neighbor_tracker_edges = disjoint_edges_deg.merge(tracking_graph, left_on='target_domain_id', right_on='domain_id', how='inner')[['domain_id_x', 'tracker_id', 'w_uniform', 'w_inv_deg']].rename(columns={'domain_id_x': 'domain_id'})
    neighbor_tracker_profiles = neighbor_tracker_edges.groupby(['domain_id', 'tracker_id'], as_index=False).agg({'w_uniform': 'sum', 'w_inv_deg': 'sum'})

    joined_eval = neighbor_tracker_profiles.merge(sampled_tracking[['domain_id', 'tracker_id']].assign(is_true_tracker=1.0), on=['domain_id', 'tracker_id'], how='left')
    joined_eval = joined_eval.assign(is_true_tracker=joined_eval['is_true_tracker'].fillna(0.0))

    mask_multi = (joined_eval['w_uniform'] >= 2.0)
    joined_eval = joined_eval.assign(is_multi_uniform=mask_multi.astype('int32'))
    eval_by_multi = joined_eval.groupby('is_multi_uniform', as_index=False).agg({'domain_id': 'count', 'is_true_tracker': 'mean'}).rename(columns={'domain_id': 'edge_count', 'is_true_tracker': 'precision'})
    eval_by_multi = eval_by_multi.assign(weighting_scheme=eval_by_multi['is_multi_uniform'].map({1: 'uniform_multi_neighbor (>=2)', 0: 'uniform_single_neighbor (=1)'}))

    mask_inv_high = (joined_eval['w_inv_deg'] >= 0.05)
    joined_eval = joined_eval.assign(is_high_inv=mask_inv_high.astype('int32'))
    eval_by_inv = joined_eval.groupby('is_high_inv', as_index=False).agg({'domain_id': 'count', 'is_true_tracker': 'mean'}).rename(columns={'domain_id': 'edge_count', 'is_true_tracker': 'precision'})
    eval_by_inv = eval_by_inv.assign(weighting_scheme=eval_by_inv['is_high_inv'].map({1: 'inv_degree_high_authority (>=0.05)', 0: 'inv_degree_low_authority (<0.05)'}))

    cols_out2 = ['weighting_scheme', 'edge_count', 'precision']
    out2 = eval_by_multi[cols_out2].skb.concat([eval_by_inv[cols_out2]], axis=0).reset_index(drop=True)

    domains = skrub.as_data_op(DOMAINS_PATH).skb.apply_func(pd.read_parquet)
    train_domains_with_host = train_profile.merge(domains, on='domain_id', how='left')
    tld_col = train_domains_with_host['domain'].astype(str).str.rsplit('.', n=1).str.get(-1).str.lower()
    train_domains_with_host = train_domains_with_host.assign(tld=tld_col)

    isolated_domains = train_domains_with_host[train_domains_with_host['is_isolated'] == 1]
    connected_domains = train_domains_with_host[train_domains_with_host['is_connected'] == 1]

    iso_tld_counts = isolated_domains.groupby('tld', as_index=False).agg({'domain_id': 'count'}).rename(columns={'domain_id': 'iso_domain_count'}).sort_values('iso_domain_count', ascending=False)
    conn_tld_counts = connected_domains.groupby('tld', as_index=False).agg({'domain_id': 'count'}).rename(columns={'domain_id': 'conn_domain_count'})
    tld_comparison = iso_tld_counts.head(20).merge(conn_tld_counts, on='tld', how='left')
    tld_comparison = tld_comparison.assign(conn_domain_count=tld_comparison['conn_domain_count'].fillna(0.0))
    tld_comparison = tld_comparison.assign(iso_share_pct=(tld_comparison['iso_domain_count'] / 4480.0) * 100.0)
    tld_comparison = tld_comparison.assign(conn_share_pct=(tld_comparison['conn_domain_count'] / 45520.0) * 100.0)
    out3 = tld_comparison[['tld', 'iso_domain_count', 'conn_domain_count', 'iso_share_pct', 'conn_share_pct']].reset_index(drop=True)

    pairs = sampled_tracking.merge(sampled_tracking, on='domain_id', how='inner')
    pairs = pairs[pairs['tracker_id_x'] != pairs['tracker_id_y']]
    pair_counts = pairs.groupby(['tracker_id_x', 'tracker_id_y'], as_index=False).agg({'domain_id': 'count'}).rename(columns={'domain_id': 'support'})

    t_freq = sampled_tracking.groupby('tracker_id', as_index=False).agg({'domain_id': 'count'}).rename(columns={'domain_id': 'n_x'})
    pair_counts = pair_counts.merge(t_freq, left_on='tracker_id_x', right_on='tracker_id', how='left').drop(columns=['tracker_id'])
    t_freq_y = t_freq.rename(columns={'n_x': 'n_y', 'tracker_id': 'tracker_id'})
    pair_counts = pair_counts.merge(t_freq_y, left_on='tracker_id_y', right_on='tracker_id', how='left').drop(columns=['tracker_id'])

    pair_counts = pair_counts.assign(cond_prob=pair_counts['support'] / pair_counts['n_x'])
    pair_counts = pair_counts.assign(lift=(pair_counts['support'] * 50000.0) / (pair_counts['n_x'] * pair_counts['n_y']))
    pair_counts = pair_counts.assign(excess_prob=pair_counts['cond_prob'] - (pair_counts['n_y'] / 50000.0))

    top_pairs = pair_counts[pair_counts['support'] >= 50].sort_values('lift', ascending=False).head(25)
    top_pairs = top_pairs.assign(pair_label='t_' + top_pairs['tracker_id_x'].astype(str) + ' -> t_' + top_pairs['tracker_id_y'].astype(str))
    out4 = top_pairs[['pair_label', 'support', 'cond_prob', 'lift', 'excess_prob']].reset_index(drop=True)

    return {
        'subgroup_recall_breakdown': out1,
        'hub_damped_neighbors_study': out2,
        'lexical_knn_isolated_study': out3,
        'cooccurrence_diffusion_study': out4
    }