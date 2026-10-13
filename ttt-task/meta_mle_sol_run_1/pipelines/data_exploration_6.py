import numpy as np
import pandas as pd
import skrub
from scipy.stats import entropy

BASE = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/"
N_POOL = 1206000
N_RESERVED = 6000
N_TRACKERS = 355


def read_parquet(name, columns):
    return skrub.as_data_op(BASE + name).skb.apply_func(
        pd.read_parquet, columns=columns
    )


def category_profiles(edges, relevant, memberships, categories, focal, neighbour):
    selected = edges[edges[focal].isin(relevant["domain_id"])]
    selected = selected[selected[focal] != selected[neighbour]]
    selected = selected.drop_duplicates([focal, neighbour])
    degree = selected.groupby(focal).size().rename("degree").reset_index()
    degree = degree.rename(columns={focal: "domain_id"})

    classified = selected[selected[neighbour].isin(memberships["domain_id"])]
    count = classified.groupby(focal).size().rename("classified_neighbours").reset_index()
    count = count.rename(columns={focal: "domain_id"})
    joined = classified.merge(
        memberships.rename(columns={"domain_id": neighbour}),
        on=neighbour, how="inner", sort=False
    )
    mass = joined.groupby([focal, "category"])["membership_weight"].sum().reset_index()
    wide = mass.pivot_table(
        index=focal, columns="category", values="membership_weight",
        aggfunc="sum", fill_value=0
    )
    wide = wide.reindex(
        index=relevant["domain_id"], columns=categories, fill_value=0
    ).fillna(0).reset_index(drop=True)
    totals = wide.sum(axis=1).replace(0, np.nan)
    profile = wide.div(totals, axis=0).fillna(0)

    stats = relevant.merge(degree, on="domain_id", how="left", sort=False)
    stats = stats.merge(count, on="domain_id", how="left", sort=False)
    stats = stats.fillna({"degree": 0, "classified_neighbours": 0})
    stats = stats.assign(
        covered=(stats["classified_neighbours"] > 0).astype("float32"),
        classified_fraction=stats["classified_neighbours"] / stats["degree"].replace(0, np.nan),
        category_entropy=profile.to_numpy().skb.apply_func(entropy, axis=1),
        distinct_categories=(profile > 0).sum(axis=1).to_numpy(),
        maximum_category_mass=profile.max(axis=1).to_numpy()
    )
    stats = stats.fillna({"classified_fraction": 0, "category_entropy": 0})
    return profile, stats, selected.shape, classified.shape


def population_summaries(stats):
    return {
        "means": stats.groupby("population").agg(
            domains=("domain_id", "size"),
            coverage=("covered", "mean"),
            mean_degree=("degree", "mean"),
            mean_classified_neighbours=("classified_neighbours", "mean"),
            mean_classified_fraction=("classified_fraction", "mean"),
            mean_category_entropy=("category_entropy", "mean"),
            mean_distinct_categories=("distinct_categories", "mean"),
            mean_maximum_category_mass=("maximum_category_mass", "mean")
        ).reset_index(),
        "quantiles": stats.groupby("population")[
            ["degree", "classified_neighbours", "classified_fraction",
             "category_entropy", "distinct_categories", "maximum_category_mass"]
        ].quantile([0.1, 0.5, 0.9, 0.99]).reset_index()
    }


def ranking_diagnostics(profile, stats, relevance, truth):
    scores = profile.iloc[:N_RESERVED].to_numpy().dot(relevance.to_numpy())
    guesses = (-scores).skb.apply_func(np.argsort, axis=1, kind="stable")[:, :10]
    hits = truth.skb.apply_func(np.take_along_axis, guesses, axis=1).sum(axis=1)
    cardinality = truth.sum(axis=1)
    report = stats.iloc[:N_RESERVED].assign(
        known_trackers=cardinality,
        recall=hits / cardinality,
        exact_recovery=(hits == cardinality).astype("float32")
    )
    covered = report[report["covered"] > 0]
    return {
        "coverage": report.groupby("population").agg(
            domains=("domain_id", "size"),
            covered_domains=("covered", "sum"),
            coverage=("covered", "mean")
        ).reset_index(),
        "covered_rankings": covered.groupby("population").agg(
            domains=("domain_id", "size"),
            mean_known_trackers=("known_trackers", "mean"),
            recall_at_10=("recall", "mean"),
            exact_recovery_fraction=("exact_recovery", "mean")
        ).reset_index(),
        "preview": report.head(12)
    }


def build():
    labels = read_parquet(
        "tracking_graph_train.parquet", ["domain_id", "tracker_id"]
    ).drop_duplicates(["domain_id", "tracker_id"])
    counts = labels.groupby("domain_id")["tracker_id"].nunique().rename(
        "known_tracker_count"
    ).reset_index()
    eligible = counts[counts["known_tracker_count"] >= 2]
    pool = eligible[["domain_id"]].sort_values("domain_id").sample(
        n=N_POOL, replace=False, random_state=42
    ).reset_index(drop=True)
    targets = skrub.as_data_op(BASE + "target.tsv").skb.apply_func(
        pd.read_csv, sep="\t", usecols=["domain_id"]
    )
    relevant = pool.skb.concat([targets], axis=0).reset_index(drop=True)
    relevant = relevant.reset_index().rename(columns={"index": "profile_row"})
    relevant = relevant.assign(population="training")
    relevant = relevant.assign(
        population=relevant["population"].mask(
            relevant["profile_row"] >= N_POOL, "prediction"
        )
    )
    for fold in range(3):
        mask = (relevant["profile_row"] >= fold * 2000) & (
            relevant["profile_row"] < (fold + 1) * 2000
        )
        relevant = relevant.assign(
            population=relevant["population"].mask(mask, "validation_" + str(fold))
        )

    raw = skrub.as_data_op(BASE + "url-classification.csv").skb.apply_func(
        pd.read_csv, usecols=["url", "category"]
    )
    host = raw["url"].fillna("").str.lower()
    host = host.str.replace("^https?://", "", regex=True)
    host = host.str.split("/").str[0].str.replace("^www\\.", "", regex=True)
    parsed = raw.assign(host=host)
    parsed = parsed[
        (parsed["host"] != "") & parsed["category"].notna()
    ]
    pairs = parsed[["host", "category"]].drop_duplicates()
    categories = pairs["category"].drop_duplicates().sort_values().reset_index(drop=True)
    host_counts = pairs.groupby("host")["category"].nunique().rename(
        "category_count"
    ).reset_index()
    conflicts = host_counts[host_counts["category_count"] > 1]
    pairs = pairs.merge(host_counts, on="host", how="inner", sort=False)
    pairs = pairs.assign(membership_weight=1.0 / pairs["category_count"])

    domains = read_parquet("domains.parquet", ["domain_id", "domain"])
    domain_host = domains["domain"].fillna("").str.lower().str.replace(
        "^www\\.", "", regex=True
    )
    lookup = domains.assign(host=domain_host)
    classified_lookup = lookup[lookup["host"].isin(pairs["host"])]
    memberships = classified_lookup[["domain_id", "host"]].merge(
        pairs, on="host", how="inner", sort=False
    )[["domain_id", "category", "membership_weight"]]
    memberships = memberships.drop_duplicates(["domain_id", "category"])

    external = labels[~labels["domain_id"].isin(pool["domain_id"])]
    external_counts = external.groupby("domain_id")["tracker_id"].nunique().rename(
        "external_cardinality"
    ).reset_index()
    external = external.merge(
        external_counts, on="domain_id", how="inner", sort=False
    )
    # Fractional category membership prevents multi-category domains from
    # contributing multiple full units of domain relevance.
    semantic_labels = external[
        external["domain_id"].isin(memberships["domain_id"])
    ].merge(memberships, on="domain_id", how="inner", sort=False)
    semantic_labels = semantic_labels.assign(
        relevance_weight=semantic_labels["membership_weight"]
        / semantic_labels["external_cardinality"]
    )
    support_rows = semantic_labels[
        ["domain_id", "category", "membership_weight"]
    ].drop_duplicates(["domain_id", "category"])
    support = support_rows.groupby("category")["membership_weight"].sum()
    relevance = semantic_labels.groupby(["category", "tracker_id"])[
        "relevance_weight"
    ].sum().reset_index()
    relevance = relevance.pivot_table(
        index="category", columns="tracker_id", values="relevance_weight",
        aggfunc="sum", fill_value=0
    ).reindex(index=categories, columns=list(range(N_TRACKERS)), fill_value=0).fillna(0)
    relevance = relevance.div(support.reindex(categories).replace(0, np.nan), axis=0).fillna(0)

    direct_rows = memberships[memberships["domain_id"].isin(relevant["domain_id"])]
    direct = direct_rows.pivot_table(
        index="domain_id", columns="category", values="membership_weight",
        aggfunc="sum", fill_value=0
    ).reindex(index=relevant["domain_id"], columns=categories, fill_value=0).fillna(0)
    direct = direct.reset_index(drop=True)
    direct_stats = relevant.assign(
        degree=1,
        classified_neighbours=(direct.sum(axis=1) > 0).astype("float32").to_numpy(),
        covered=(direct.sum(axis=1) > 0).astype("float32").to_numpy(),
        classified_fraction=(direct.sum(axis=1) > 0).astype("float32").to_numpy(),
        category_entropy=direct.to_numpy().skb.apply_func(entropy, axis=1),
        distinct_categories=(direct > 0).sum(axis=1).to_numpy(),
        maximum_category_mass=direct.max(axis=1).to_numpy()
    ).fillna({"category_entropy": 0})

    links = read_parquet(
        "link-graph.parquet", ["source_domain_id", "target_domain_id"]
    )
    touching = links[
        links["source_domain_id"].isin(relevant["domain_id"])
        | links["target_domain_id"].isin(relevant["domain_id"])
    ]
    outgoing, out_stats, out_shape, out_classified_shape = category_profiles(
        touching, relevant, memberships, categories,
        "source_domain_id", "target_domain_id"
    )
    incoming, in_stats, in_shape, in_classified_shape = category_profiles(
        touching, relevant, memberships, categories,
        "target_domain_id", "source_domain_id"
    )

    validation_keys = pool.iloc[:N_RESERVED]
    validation_labels = labels.merge(
        validation_keys, on="domain_id", how="inner", sort=False
    )
    truth = validation_labels.assign(present=1).pivot_table(
        index="domain_id", columns="tracker_id", values="present",
        aggfunc="max", fill_value=0
    ).reindex(
        index=validation_keys["domain_id"],
        columns=list(range(N_TRACKERS)), fill_value=0
    ).fillna(0).astype("uint8").to_numpy()

    outputs = {
        "pool_shape": pool.shape,
        "classification_source_shape": raw.shape,
        "normalized_host_category_shape": pairs.shape,
        "conflicting_host_count": conflicts.shape,
        "category_count_distribution": host_counts.groupby("category_count").size().rename(
            "hosts"
        ).reset_index(),
        "conflict_preview": pairs[pairs["host"].isin(conflicts["host"])].head(20),
        "category_support": support.rename("fractional_external_domains").reset_index(),
        "auxiliary_pool_overlap": support_rows[["domain_id"]].drop_duplicates().merge(
            pool, on="domain_id", how="inner"
        ).shape,
        "touching_link_shape": touching.shape,
        "outgoing_distinct_edge_shape": out_shape,
        "incoming_distinct_edge_shape": in_shape,
        "outgoing_classified_edge_shape": out_classified_shape,
        "incoming_classified_edge_shape": in_classified_shape,
        "budget_assessment": skrub.as_data_op({
            "remaining_seconds_at_request": 2781,
            "parent_mean_fit_seconds": 652.4624671141306,
            "parent_mean_score_seconds": 396.9476758639018,
            "parent_three_fold_fit_score_seconds": 3148.230428934097,
            "decision": "Do not launch the augmented full-fold experiment: even the parent's measured three-fold fit-and-score cost exceeds the remaining budget before adding semantic construction or a safety margin.",
            "timing_measurement": "Use the harness eval_outputs wall time as complete construction-plus-diagnostic cost. No parent fitting is performed. Internal isolated semantic-build timing is unfinished.",
            "memory_measurement": "Peak process memory is not instrumented in this graph; report it as unfinished unless the harness supplies it.",
            "interpretation": "Standalone category-conditioned rankings are feature diagnostics, not evidence of complementarity to the learned parent.",
            "label_safety": "All locked-pool domains are excluded before auxiliary category-label joins and aggregations. Validation labels appear only in diagnostic scoring.",
            "normalization": "Each classified neighbour contributes unit mass split equally over its distinct categories. Profiles average these masses over classified neighbours.",
            "unfinished": "Isolated semantic-feature wall time and peak-memory instrumentation; no augmented learned-model score."
        })
    }
    for name, profile, stats in [
        ("direct", direct, direct_stats),
        ("outgoing", outgoing, out_stats),
        ("incoming", incoming, in_stats)
    ]:
        for key, value in population_summaries(stats).items():
            outputs[name + "_" + key] = value
        for key, value in ranking_diagnostics(profile, stats, relevance, truth).items():
            outputs[name + "_" + key] = value
    return outputs