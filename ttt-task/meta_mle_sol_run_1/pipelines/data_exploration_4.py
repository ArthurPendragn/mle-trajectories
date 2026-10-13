import numpy as np
import pandas as pd
import skrub

BASE = "/home/estrauss-ldap/repos/mle-claude/tasks/trackthetrackers-task/"
N_POOL = 1206000
N_RESERVED = 6000


def coverage_table(rows):
    return rows.groupby(["population", "coverage"], observed=True).agg(
        domains=("domain_id", "size"),
        mean_out_degree=("out_degree", "mean"),
        mean_in_degree=("in_degree", "mean"),
    ).reset_index()


def build():
    availability = skrub.as_data_op({
        "status": "partial_analysis_predictions_unavailable",
        "requested_probe": "probe_9c46b8b65931",
        "reason": (
            "The input contains a truncated prediction preview, not the complete "
            "6000-row out-of-fold predictions or a recorded DataOp holding them. "
            "The listed task sources contain no predictions. Reading an earlier "
            "probe output file is prohibited by the plan contract, and refitting "
            "is prohibited by this exploration's intent."
        ),
        "not_computed": [
            "Recall@10 by coverage, geography, cardinality, or tracker frequency",
            "Missed-tracker ranks and domain-normalized loss contributions",
            "Representative failures with actual top-ten guesses",
            "A model-error-based priority for the next experiment",
        ],
        "provided_instead": (
            "Reconstructed validation-domain metadata, label cardinality, the "
            "mathematical Recall@10 ceiling, and tracker training-frequency "
            "diagnostics. None are estimates of model performance."
        ),
        "required_to_complete": (
            "Provide the complete probe predictions as an authorized in-memory "
            "DataOp/input, or explicitly revise the source-read contract to "
            "authorize the probe artifact. No new model fit is needed."
        ),
    })

    labels = skrub.as_data_op(
        BASE + "tracking_graph_train.parquet"
    ).skb.apply_func(pd.read_parquet, columns=["domain_id", "tracker_id"])
    labels = labels.drop_duplicates(["domain_id", "tracker_id"])
    counts = labels.groupby("domain_id")["tracker_id"].nunique().rename(
        "known_tracker_count"
    ).reset_index()
    eligible = counts[counts["known_tracker_count"] >= 2]
    pool = eligible[["domain_id"]].sort_values("domain_id").sample(
        n=N_POOL, replace=False, random_state=42
    ).reset_index(drop=True)

    validation = pool.iloc[:N_RESERVED].copy()
    validation = validation.assign(
        fold=validation.index // 2000,
        population="validation",
    )
    training = pool.iloc[N_RESERVED:][["domain_id"]]
    targets = skrub.as_data_op(BASE + "target.tsv").skb.apply_func(
        pd.read_csv, sep="\t", usecols=["domain_id"]
    )
    targets = targets.assign(population="prediction", fold=-1)
    observed = validation.skb.concat([targets], axis=0).reset_index(drop=True)

    links = skrub.as_data_op(BASE + "link-graph.parquet").skb.apply_func(
        pd.read_parquet,
        columns=["source_domain_id", "target_domain_id"],
    )
    outgoing = links[
        links["source_domain_id"].isin(observed["domain_id"])
    ]
    incoming = links[
        links["target_domain_id"].isin(observed["domain_id"])
    ]
    out_degree = outgoing.groupby("source_domain_id").size().rename(
        "out_degree"
    ).reset_index().rename(columns={"source_domain_id": "domain_id"})
    in_degree = incoming.groupby("target_domain_id").size().rename(
        "in_degree"
    ).reset_index().rename(columns={"target_domain_id": "domain_id"})
    observed = observed.merge(
        out_degree, on="domain_id", how="left", sort=False
    ).merge(in_degree, on="domain_id", how="left", sort=False)
    observed = observed.assign(
        out_degree=observed["out_degree"].fillna(0),
        in_degree=observed["in_degree"].fillna(0),
    )
    observed = observed.assign(
        coverage_code=(
            (observed["out_degree"] > 0).astype("int64")
            + 2 * (observed["in_degree"] > 0).astype("int64")
        )
    )
    observed = observed.assign(
        coverage=observed["coverage_code"].map({
            0: "neither_direction",
            1: "outgoing_only",
            2: "incoming_only",
            3: "both_directions",
        })
    )

    domains = skrub.as_data_op(BASE + "domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "domain"]
    )
    domains = domains[domains["domain_id"].isin(observed["domain_id"])]
    observed = observed.merge(
        domains, on="domain_id", how="left", sort=False
    )
    observed = observed.assign(
        tld=observed["domain"].fillna("").str.lower().str.rsplit(".").str[-1]
    )
    validation_metadata = observed[
        observed["population"] == "validation"
    ].merge(counts, on="domain_id", how="left", sort=False)
    validation_metadata = validation_metadata.assign(
        oracle_recall_ceiling=(
            10.0 / validation_metadata["known_tracker_count"]
        ).clip(upper=1.0)
    )
    validation_metadata = validation_metadata.assign(
        unavoidable_recall_loss=(
            1.0 - validation_metadata["oracle_recall_ceiling"]
        )
    )
    cardinality = validation_metadata.groupby(
        ["fold", "known_tracker_count"]
    ).agg(
        domains=("domain_id", "size"),
        mean_oracle_recall_ceiling=("oracle_recall_ceiling", "mean"),
        mean_unavoidable_loss=("unavoidable_recall_loss", "mean"),
    ).reset_index()
    ceiling = validation_metadata.groupby("fold").agg(
        domains=("domain_id", "size"),
        mean_oracle_recall_ceiling=("oracle_recall_ceiling", "mean"),
        mean_unavoidable_loss=("unavoidable_recall_loss", "mean"),
    ).reset_index()

    training_labels = labels.merge(
        training, on="domain_id", how="inner", sort=False
    )
    frequency = training_labels.groupby("tracker_id").size().rename(
        "training_domain_count"
    ).reset_index()
    validation_labels = labels.merge(
        validation[["domain_id", "fold"]],
        on="domain_id", how="inner", sort=False,
    )
    tracker_validation = validation_labels.groupby(
        ["fold", "tracker_id"]
    ).size().rename("validation_positive_domains").reset_index()
    tracker_validation = tracker_validation.merge(
        frequency, on="tracker_id", how="left", sort=False
    )
    tracker_validation = tracker_validation.assign(
        training_domain_count=tracker_validation[
            "training_domain_count"
        ].fillna(0)
    )
    tracker_validation = tracker_validation.assign(
        frequency_band=tracker_validation[
            "training_domain_count"
        ].skb.apply_func(
            pd.cut,
            bins=[-1, 0, 100, 1000, 10000, 100000, np.inf],
            labels=[
                "unseen", "1_to_100", "101_to_1000",
                "1001_to_10000", "10001_to_100000", "over_100000",
            ],
        )
    )
    frequency_summary = tracker_validation.groupby(
        ["fold", "frequency_band"], observed=True
    ).agg(
        trackers=("tracker_id", "size"),
        validation_positive_edges=("validation_positive_domains", "sum"),
    ).reset_index()

    geography = observed.groupby(
        ["population", "tld"], observed=True
    ).size().rename("domains").reset_index().sort_values(
        ["population", "domains"], ascending=[True, False]
    )

    return {
        "analysis_availability": availability,
        "coverage_populations": coverage_table(observed),
        "validation_coverage_by_fold": validation_metadata.groupby(
            ["fold", "coverage"], observed=True
        ).agg(
            domains=("domain_id", "size"),
            mean_known_trackers=("known_tracker_count", "mean"),
        ).reset_index(),
        "cardinality_oracle_ceiling": cardinality,
        "unavoidable_loss_by_fold": ceiling,
        "tracker_frequency_exposure": frequency_summary,
        "geography_population_counts": geography.head(40),
        "validation_metadata_preview": validation_metadata.head(20),
        "interpretation": skrub.as_data_op({
            "coverage": "Hyperlink degree only; not externally labelled-neighbour coverage.",
            "oracle_ceiling": "min(10, known_tracker_count) / known_tracker_count.",
            "unavoidable_loss": (
                "The excess-cardinality ceiling is computable without predictions. "
                "Recoverable model loss requires actual per-domain predictions."
            ),
            "frequency": (
                "Counts use only the 1,200,000 common training domains; all "
                "6,000 reserved validation domains are excluded."
            ),
            "priority": "No model-error-based recommendation is justified without full predictions.",
        }),
    }