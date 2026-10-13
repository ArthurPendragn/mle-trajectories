import pandas as pd
import skrub

ROOT = "gs://mle-trajectories-data/ttt-task/input/"

def build():
    labels = skrub.as_data_op(ROOT + "tracking_graph_train.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "tracker_id"]
    ).drop_duplicates(["domain_id", "tracker_id"])
    counts = labels.groupby("domain_id").size().rename("tracker_count").reset_index()
    eligible = counts[counts["tracker_count"] >= 2][["domain_id"]]
    pool = eligible.sample(n=169000, random_state=42).reset_index(drop=True)
    folds = []
    for fold in range(3):
        folds.append(
            pool.iloc[160000 + 3000 * fold:160000 + 3000 * (fold + 1)]
            .assign(population="validation", fold=fold)
        )
    validation = folds[0].skb.concat(folds[1:], axis=0).reset_index(drop=True)
    targets = skrub.as_data_op(ROOT + "target.tsv").skb.apply_func(
        pd.read_csv, sep="\t", usecols=["domain_id"]
    ).assign(population="target", fold=-1)
    rows = validation.skb.concat([targets], axis=0).reset_index(drop=True)
    domains = skrub.as_data_op(ROOT + "domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain_id", "domain"]
    )
    domains = domains[domains["domain_id"].isin(rows["domain_id"])]
    rows = rows.merge(domains, on="domain_id", how="left")
    rows = rows.assign(tld=rows["domain"].str.lower().str.split(".").str[-1].fillna("unknown"))

    categories = skrub.as_data_op(ROOT + "url-classification.csv").skb.apply_func(
        pd.read_csv, usecols=["url", "category"]
    )
    categories = categories.assign(
        domain=categories["url"].str.lower()
        .str.replace(r"^https?://", "", regex=True)
        .str.split("/").str[0]
        .str.replace(r"^www\.", "", regex=True)
    )
    categories = categories[["domain"]].drop_duplicates().assign(has_content=1)
    rows = rows.assign(
        content_domain=rows["domain"].str.lower().str.replace(r"^www\.", "", regex=True)
    )
    rows = rows.merge(
        categories.rename(columns={"domain": "content_domain"}),
        on="content_domain", how="left"
    )
    rows = rows.assign(has_content=rows["has_content"].fillna(0).astype("int32"))

    links = skrub.as_data_op(ROOT + "link-graph.parquet").skb.apply_func(
        pd.read_parquet, columns=["source_domain_id", "target_domain_id"]
    )
    disjoint = counts[~counts["domain_id"].isin(pool["domain_id"])][["domain_id"]]
    disjoint = disjoint.rename(columns={"domain_id": "neighbor_id"}).assign(labelled_neighbor=1)
    for direction, focal, neighbor in [
        ("out", "source_domain_id", "target_domain_id"),
        ("in", "target_domain_id", "source_domain_id"),
    ]:
        incident = links[links[focal].isin(rows["domain_id"])]
        incident = incident.rename(columns={focal: "domain_id", neighbor: "neighbor_id"})
        incident = incident.drop_duplicates(["domain_id", "neighbor_id"])
        incident = incident.merge(disjoint, on="neighbor_id", how="left")
        incident = incident.assign(labelled_neighbor=incident["labelled_neighbor"].fillna(0))
        degree = incident.groupby("domain_id").agg(
            degree=("neighbor_id", "size"),
            covered=("labelled_neighbor", "sum")
        ).reset_index().rename(columns={
            "degree": direction + "_degree",
            "covered": direction + "_covered"
        })
        rows = rows.merge(degree, on="domain_id", how="left")
        rows = rows.assign(**{
            direction + "_degree": rows[direction + "_degree"].fillna(0),
            direction + "_covered": rows[direction + "_covered"].fillna(0)
        })
    rows = rows.assign(
        connected=(rows["out_degree"] + rows["in_degree"] > 0),
        labelled_neighbor_available=(rows["out_covered"] + rows["in_covered"] > 0)
    )
    totals = rows.groupby(["population", "fold"]).size().rename("total").reset_index()
    outputs = {}
    for feature in ["tld", "has_content", "connected", "labelled_neighbor_available"]:
        summary = rows.groupby(["population", "fold", feature]).size().rename("rows").reset_index()
        summary = summary.merge(totals, on=["population", "fold"], how="left")
        summary = summary.assign(fraction=summary["rows"] / summary["total"])
        outputs[feature + "_frequencies"] = summary.sort_values(
            ["population", "fold", "rows"], ascending=[True, True, False]
        ).head(20)

    outputs["connectivity_summary"] = rows.groupby(["population", "fold"]).agg(
        rows=("domain_id", "size"),
        mean_out_degree=("out_degree", "mean"),
        mean_in_degree=("in_degree", "mean"),
        mean_out_labelled_neighbors=("out_covered", "mean"),
        mean_in_labelled_neighbors=("in_covered", "mean")
    ).reset_index()
    outputs["prediction_access_limitation"] = skrub.as_data_op({
        "status": "Requested prediction-based error analysis cannot be computed under the source-read contract.",
        "reason": "Stored out-of-fold artifacts are not permitted plan inputs. No candidate predictions are supplied as source DataOps; refitting is explicitly excluded by the request.",
        "not_computed": [
            "per-domain recall", "paired candidate differences", "missed recall by tracker",
            "subgroup error scores", "fold-level candidate comparisons"
        ],
        "provided": "Recomputed label-free validation/target frequencies and disjoint labelled-neighbor coverage; no prediction arrays, refitting, or population-selection study.",
        "coverage_definition": "Distinct incident neighbors with recorded labels outside the entire 169000-domain locked pool.",
        "interpretation": "These are availability audits, not evidence of model error concentrations."
    })
    return outputs