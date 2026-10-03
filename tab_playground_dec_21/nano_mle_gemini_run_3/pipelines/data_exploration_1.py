import pandas as pd
import skrub

TRAIN_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"
TEST_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/test.csv"

CONTINUOUS_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]


def build():
    train_df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    test_df = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    # 1. Target distribution: Cover_Type frequencies and rare classes
    cover_type_counts = (
        train_df[["Cover_Type", "Id"]]
        .groupby("Cover_Type")
        .count()
        .rename(columns={"Id": "count"})
    )
    cover_type_distribution = cover_type_counts.assign(
        percentage=(cover_type_counts["count"] / 4000000.0) * 100.0
    )
    rare_cover_types = cover_type_distribution[cover_type_distribution["count"] < 1000]

    # 2. Missing value counts across train and test
    train_nulls = train_df.isna().sum().to_frame(name="train_null_count")
    test_nulls = test_df.isna().sum().to_frame(name="test_null_count")
    missing_values_summary = train_nulls.join(test_nulls, how="outer")

    mask_train_null = (missing_values_summary["train_null_count"] > 0).fillna(False)
    mask_test_null = (missing_values_summary["test_null_count"] > 0).fillna(False)
    columns_with_missing_values = missing_values_summary[
        mask_train_null | mask_test_null
    ]

    # 3. Summary statistics and zero-variance / constant column detection
    train_features = train_df.drop(columns=["Cover_Type", "Id"])
    test_features = test_df.drop(columns=["Id"])

    train_summary = train_features.describe().transpose()
    test_summary = test_features.describe().transpose()

    train_constant_columns = train_summary[train_summary["min"] == train_summary["max"]][
        ["count", "mean", "std", "min", "max"]
    ]
    test_constant_columns = test_summary[test_summary["min"] == test_summary["max"]][
        ["count", "mean", "std", "min", "max"]
    ]

    # 4. Feature distribution comparison between train and test
    train_stats = train_summary[["mean", "std", "min", "max"]].rename(
        columns={
            "mean": "train_mean",
            "std": "train_std",
            "min": "train_min",
            "max": "train_max",
        }
    )
    test_stats = test_summary[["mean", "std", "min", "max"]].rename(
        columns={
            "mean": "test_mean",
            "std": "test_std",
            "min": "test_min",
            "max": "test_max",
        }
    )
    feature_comparison = train_stats.join(test_stats, how="inner")
    feature_comparison = feature_comparison.assign(
        abs_mean_diff=(
            feature_comparison["train_mean"] - feature_comparison["test_mean"]
        ).abs(),
        abs_std_diff=(
            feature_comparison["train_std"] - feature_comparison["test_std"]
        ).abs(),
    )
    feature_comparison_top_drift = feature_comparison.sort_values(
        by="abs_mean_diff", ascending=False
    )
    continuous_features_comparison = feature_comparison.loc[CONTINUOUS_COLS]

    return {
        "cover_type_distribution": cover_type_distribution,
        "rare_cover_types": rare_cover_types,
        "missing_values_summary": missing_values_summary,
        "columns_with_missing_values": columns_with_missing_values,
        "train_constant_columns": train_constant_columns,
        "test_constant_columns": test_constant_columns,
        "continuous_features_comparison": continuous_features_comparison,
        "feature_comparison_top_drift": feature_comparison_top_drift,
    }