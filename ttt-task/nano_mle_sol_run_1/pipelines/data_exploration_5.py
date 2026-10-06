import time
import resource
import numpy as np
import pandas as pd
import scipy.sparse as sp
import skrub

BASE = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/"
N_POOL = 1206000
N_RESERVED = 6000
N_TRACKERS = 355
FIRST_CAP = 16
TERMINAL_CAP = 64


def matrix(values, rows, cols, shape):
    specification = skrub.as_data_op((values, (rows, cols)))
    return specification.skb.apply_func(sp.csr_matrix, shape=shape, dtype=np.float32)


def dense_frame(array, columns):
    return array.skb.apply_func(pd.DataFrame, columns=columns)


def bounded_profile(links, relevant, auxiliary, external_ids, n_domains,
                    focal, hub, name):
    first = links[links[focal].isin(relevant["domain_id"])]
    first = first[first[focal] != first[hub]].drop_duplicates([focal, hub])
    first_counts = first.groupby(focal).size().rename("first_before").reset_index()
    # Source-order bounds are deterministic, but not uniform random sampling.
    first = first.groupby(focal, sort=False).head(FIRST_CAP)
    hubs = first[[hub]].drop_duplicates().reset_index(drop=True)
    hubs = hubs.reset_index().rename(columns={"index": "hub_row"})

    terminal = links[links[hub].isin(hubs[hub])]
    terminal = terminal[terminal[focal].isin(external_ids["domain_id"])]
    terminal = terminal[terminal[focal] != terminal[hub]]
    terminal = terminal.drop_duplicates([hub, focal])
    terminal_counts = terminal.groupby(hub).size().rename("terminal_before").reset_index()
    terminal = terminal.groupby(hub, sort=False).head(TERMINAL_CAP)
    retained_counts = terminal.groupby(hub).size().rename("terminal_retained").reset_index()
    terminal = terminal.merge(hubs, on=hub, how="inner", sort=False)
    terminal = terminal.merge(retained_counts, on=hub, how="left", sort=False)

    first = first.merge(hubs, on=hub, how="inner", sort=False)
    first = first.merge(
        relevant[["domain_id", "profile_row"]].rename(columns={"domain_id": focal}),
        on=focal, how="inner", sort=False
    )
    first = first.merge(retained_counts, on=hub, how="left", sort=False)
    first = first.assign(terminal_retained=first["terminal_retained"].fillna(0))
    covered = first[first["terminal_retained"] > 0]

    # Timer covers sparse assembly and profile propagation, not preceding joins.
    assembly_start = (
        (covered.shape[0] * 0 + time.CLOCK_MONOTONIC)
        .skb.apply_func(int).skb.apply_func(time.clock_gettime)
    )
    a = matrix(
        covered[focal].astype("float32").to_numpy() * 0 + 1 + assembly_start * 0,
        covered["profile_row"].to_numpy(), covered["hub_row"].to_numpy(),
        (relevant.shape[0], hubs.shape[0])
    )
    b = matrix(
        (1.0 / terminal["terminal_retained"]).astype("float32").to_numpy(),
        terminal["hub_row"].to_numpy(), terminal[focal].to_numpy(),
        (hubs.shape[0], n_domains)
    )
    hub_profiles = b.dot(auxiliary)
    sums = a.dot(hub_profiles)
    count = a.sum(axis=1).skb.apply_func(np.asarray).ravel()
    profiles = dense_frame(sums.toarray(), list(range(N_TRACKERS)))
    profiles = profiles.div(count.skb.apply_func(pd.Series).replace(0, np.nan), axis=0).fillna(0)
    assembly_end = (
        (profiles.shape[0] * 0 + time.CLOCK_MONOTONIC)
        .skb.apply_func(int).skb.apply_func(time.clock_gettime)
    )
    elapsed = assembly_end - assembly_start

    coverage = relevant[["domain_id", "population"]].assign(
        covered_hubs=count,
        profile_mass=profiles.sum(axis=1),
        profile_max=profiles.max(axis=1),
        profile_nonzero=(profiles > 0).sum(axis=1),
        covered=(count > 0).astype("int64")
    )
    coverage = coverage.merge(
        first_counts.rename(columns={focal: "domain_id"}),
        on="domain_id", how="left", sort=False
    ).fillna({"first_before": 0})
    coverage = coverage.assign(
        first_retained=coverage["first_before"].clip(upper=FIRST_CAP),
        first_truncated=(coverage["first_before"] > FIRST_CAP).astype("int64")
    )
    hub_audit = hubs.merge(terminal_counts, on=hub, how="left", sort=False)
    hub_audit = hub_audit.merge(retained_counts, on=hub, how="left", sort=False).fillna(0)
    hub_audit = hub_audit.assign(
        terminal_truncated=(hub_audit["terminal_before"] > TERMINAL_CAP).astype("int64")
    )
    totals = skrub.as_data_op({"block": name})
    totals = totals.skb.apply_func(pd.DataFrame, index=[0])
    totals = totals.assign(
        retained_first_edges=first.shape[0],
        intermediaries=hubs.shape[0],
        retained_terminal_edges=terminal.shape[0],
        terminals_used=terminal[[focal]].drop_duplicates().shape[0],
        sparse_assembly_propagation_seconds=elapsed,
        process_peak_rss_gib=(
            (profiles.shape[0] * 0 + resource.RUSAGE_SELF)
            .skb.apply_func(int).skb.apply_func(resource.getrusage).ru_maxrss / (1024.0 ** 2)
        ),
        terminal_label_pool_overlap=terminal[[focal]].rename(
            columns={focal: "domain_id"}
        ).merge(relevant[relevant["population"] != "prediction"][["domain_id"]],
                on="domain_id", how="inner").shape[0]
    )
    hub_summary = hub_audit[["terminal_before", "terminal_retained", "terminal_truncated"]].agg(
        ["count", "mean", "sum", "max"]
    ).reset_index()
    return profiles, coverage, totals, hub_summary


def one_hop(links, relevant, auxiliary, n_domains, focal, neighbour):
    selected = links[links[focal].isin(relevant["domain_id"])]
    selected = selected[selected[focal] != selected[neighbour]]
    selected = selected.drop_duplicates([focal, neighbour])
    selected = selected.merge(
        relevant[["domain_id", "profile_row"]].rename(columns={"domain_id": focal}),
        on=focal, how="inner", sort=False
    )
    adjacency = matrix(
        selected[focal].astype("float32").to_numpy() * 0 + 1,
        selected["profile_row"].to_numpy(), selected[neighbour].to_numpy(),
        (relevant.shape[0], n_domains)
    )
    sums = adjacency.dot(auxiliary)
    counts = sums.sum(axis=1).skb.apply_func(np.asarray).ravel()
    frame = dense_frame(sums.toarray(), list(range(N_TRACKERS)))
    return frame.div(counts.skb.apply_func(pd.Series).replace(0, np.nan), axis=0).fillna(0)


def diagnostics(profiles, coverage, truth, one_guesses, one_hits):
    scores = profiles.iloc[:N_RESERVED].to_numpy()
    guesses = (-scores).skb.apply_func(np.argsort, axis=1, kind="stable")[:, :10]
    hit = truth.skb.apply_func(np.take_along_axis, guesses, axis=1).sum(axis=1)
    cardinality = truth.sum(axis=1)
    # Turn guesses into explicit binary tables without a deferred custom function.
    guessed = dense_frame(guesses, list(range(10)))
    one = dense_frame(one_guesses, list(range(10)))
    new_hit = hit * 0
    overlap = hit * 0
    for rank in range(10):
        tracker = guessed[rank].to_numpy()
        matches = one.eq(guessed[rank], axis=0).any(axis=1).to_numpy()
        positive = truth.skb.apply_func(np.take_along_axis, tracker.reshape(-1, 1), axis=1).ravel()
        overlap = overlap + matches
        new_hit = new_hit + positive * (~matches)
    rows = coverage.iloc[:N_RESERVED].reset_index(drop=True)
    rows = rows.assign(
        known_trackers=cardinality,
        recall=hit / cardinality,
        one_hop_recall=one_hits / cardinality,
        top_ten_overlap=overlap,
        additional_recall_outside_one_hop=new_hit / cardinality,
        near_perfect=(hit / cardinality >= 0.99).astype("int64")
    )
    all_summary = rows.groupby("population").agg(
        domains=("domain_id", "size"),
        coverage=("covered", "mean")
    ).reset_index()
    covered = rows[rows["covered"] > 0]
    covered_summary = covered.groupby("population").agg(
        covered_domains=("domain_id", "size"),
        mean_known_trackers=("known_trackers", "mean"),
        mean_recall=("recall", "mean"),
        mean_one_hop_recall=("one_hop_recall", "mean"),
        mean_top_ten_overlap=("top_ten_overlap", "mean"),
        additional_recall_outside_one_hop=("additional_recall_outside_one_hop", "mean"),
        near_perfect_fraction=("near_perfect", "mean")
    ).reset_index()
    return all_summary.merge(covered_summary, on="population", how="left"), rows.head(12)


def build():
    start = skrub.as_data_op(time.CLOCK_MONOTONIC).skb.apply_func(int).skb.apply_func(time.clock_gettime)
    # Force the overall timer to precede the recorded source reads.
    suffix = (start * 0).skb.apply_func(str).replace("0.0", "")
    labels = skrub.as_data_op(BASE + "tracking_graph_train.parquet" + suffix).skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"]
    ).drop_duplicates(["domain_id", "tracker_id"])
    counts = labels.groupby("domain_id")["tracker_id"].nunique().rename("known_count").reset_index()
    eligible = counts[counts["known_count"] >= 2]
    pool = eligible[["domain_id"]].sort_values("domain_id").sample(
        n=N_POOL, replace=False, random_state=42
    ).reset_index(drop=True)
    indexed = pool.reset_index().rename(columns={"index": "pool_row"})
    indexed = indexed.assign(
        population=(indexed["pool_row"] // 2000).map(
            {0: "validation_0", 1: "validation_1", 2: "validation_2"}
        ).fillna("training")
    )
    target = skrub.as_data_op(BASE + "target.tsv").skb.apply_func(
        pd.read_csv, sep="\t", usecols=["domain_id"]
    ).assign(population="prediction")
    relevant = indexed[["domain_id", "population"]].skb.concat([target], axis=0)
    relevant = relevant.reset_index(drop=True).reset_index().rename(columns={"index": "profile_row"})
    domains = skrub.as_data_op(BASE + "domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id"]
    )
    n_domains = domains["domain_id"].max() + 1
    external = labels[~labels["domain_id"].isin(pool["domain_id"])]
    cardinality = external.groupby("domain_id")["tracker_id"].nunique().rename("cardinality").reset_index()
    external = external.merge(cardinality, on="domain_id", how="inner", sort=False)
    auxiliary = matrix(
        (1.0 / external["cardinality"]).astype("float32").to_numpy(),
        external["domain_id"].to_numpy(), external["tracker_id"].to_numpy(),
        (n_domains, N_TRACKERS)
    )
    links = skrub.as_data_op(BASE + "link-graph.parquet").skb.apply_func(
        pd.read_parquet, columns=["source_domain_id", "target_domain_id"]
    )
    validation = pool.iloc[:N_RESERVED]
    truth = labels.merge(validation, on="domain_id", how="inner", sort=False).assign(present=1)
    truth = truth.pivot_table(
        index="domain_id", columns="tracker_id", values="present", aggfunc="max", fill_value=0
    ).reindex(index=validation["domain_id"], columns=list(range(N_TRACKERS)), fill_value=0)
    truth = truth.fillna(0).astype("uint8").to_numpy()

    out = one_hop(links, relevant, auxiliary, n_domains, "source_domain_id", "target_domain_id")
    inc = one_hop(links, relevant, auxiliary, n_domains, "target_domain_id", "source_domain_id")
    baseline = (out + inc).iloc[:N_RESERVED].to_numpy()
    one_guesses = (-baseline).skb.apply_func(np.argsort, axis=1, kind="stable")[:, :10]
    one_hits = truth.skb.apply_func(np.take_along_axis, one_guesses, axis=1).sum(axis=1)
    outputs = {}
    endings = []
    for name, focal, hub in [
        ("shared_destination", "source_domain_id", "target_domain_id"),
        ("shared_source", "target_domain_id", "source_domain_id")
    ]:
        profiles, coverage, totals, hub_summary = bounded_profile(
            links, relevant, auxiliary, cardinality[["domain_id"]], n_domains, focal, hub, name
        )
        standalone, preview = diagnostics(profiles, coverage, truth, one_guesses, one_hits)
        outputs[name + "_construction"] = totals
        outputs[name + "_intermediary_bounds"] = hub_summary
        outputs[name + "_population_means"] = coverage.groupby("population")[
            ["covered", "covered_hubs", "profile_mass", "profile_max",
             "profile_nonzero", "first_before", "first_retained", "first_truncated"]
        ].mean().reset_index()
        outputs[name + "_population_quantiles"] = coverage.groupby("population")[
            ["covered_hubs", "profile_mass", "profile_max", "profile_nonzero"]
        ].quantile([0.1, 0.5, 0.9, 0.99]).reset_index()
        outputs[name + "_standalone_validation"] = standalone
        outputs[name + "_validation_preview"] = preview
        endings.append(profiles.sum().sum() * 0 + standalone.shape[0] * 0)
    finish = (
        (endings[0] + endings[1] + time.CLOCK_MONOTONIC)
        .skb.apply_func(int).skb.apply_func(time.clock_gettime)
    )
    elapsed = finish - start
    outputs["complete_cost_estimate"] = skrub.as_data_op({"protocol": "both blocks plus diagnostic one-hop construction"}).skb.apply_func(
        pd.DataFrame, index=[0]
    ).assign(
        measured_complete_exploration_seconds=elapsed,
        three_fold_upper_planning_seconds=3 * (elapsed + 465),
        peak_process_rss_gib=(endings[0] + endings[1] + resource.RUSAGE_SELF).skb.apply_func(
            int
        ).skb.apply_func(resource.getrusage).ru_maxrss / (1024.0 ** 2)
    )
    outputs["label_source_checks"] = external[["domain_id"]].drop_duplicates().merge(
        pool, on="domain_id", how="inner"
    ).shape
    outputs["interpretation"] = skrub.as_data_op({
        "bounds": "16 first-hop intermediaries per focal domain; 64 distinct externally labelled terminal domains per intermediary. Bounds use parquet row order, not random sampling.",
        "label_safety": "All 1,206,000 pool domains are excluded before terminal selection and auxiliary matrix construction. A focal-return path cannot obtain its label because the terminal label matrix has no pool labels. Targets also have no supplied tracker labels.",
        "normalization": "Each terminal contributes inverse-cardinality label mass; average terminals within each intermediary, then average covered intermediaries per domain.",
        "baseline": "Equal-weight sum of normalized incoming and outgoing one-hop profiles. This is not the learned parent's prediction.",
        "timing_limitations": "Per-block times cover sparse assembly and propagation only, excluding upstream selections and joins. Overall measured time includes source reads, both blocks, one-hop diagnostics and validation diagnostics. Its threefold replication plus 465 seconds per fold is deliberately conservative, not an exact candidate runtime.",
        "memory": "Linux process peak RSS, cumulative across the exploration; not isolated block allocations.",
        "signal": "Covered-domain recall and additional recall outside baseline guesses are descriptive feature diagnostics. Coverage must be considered separately. Near-perfect fractions flag rankings for scrutiny but do not establish leakage.",
        "recommendation_rule": "Prioritize a block only if it has substantial target coverage comparable to training/validation, meaningful additional true-label recovery outside one-hop guesses, and an affordable complete cost. Test both only if their separate evidence supports it; standalone results cannot establish a learned-model improvement.",
        "unfinished": "No isolated selection/join timing, no isolated per-block peak memory, no learned-parent error analysis, and no unrestricted two-hop graph or new model fit."
    })
    return outputs