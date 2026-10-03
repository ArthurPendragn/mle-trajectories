import pandas as pd
import numpy as np
import skrub

PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"
NUMERIC = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]
WILDERNESS = ["Wilderness_Area%d" % i for i in range(1, 5)]
SOIL = ["Soil_Type%d" % i for i in range(1, 41)]

def build():
    df = skrub.as_data_op(PATH).skb.apply_func(pd.read_csv)
    features = NUMERIC + WILDERNESS + SOIL
    hashes = df[features].skb.apply_func(pd.util.hash_pandas_object, index=False)
    blocks = (df["Id"].rank(method="first", pct=True) * 5).astype("int64").clip(upper=4)
    annotated = df.assign(block=blocks, feature_hash=hashes)
    grouped = annotated.groupby("feature_hash").agg(
        rows=("Cover_Type", "size"),
        labels=("Cover_Type", "nunique")
    )
    indicator_checks = df.assign(
        wilderness_sum=df[WILDERNESS].sum(axis=1),
        soil_sum=df[SOIL].sum(axis=1)
    )
    sample = annotated[(annotated["Id"] % 10) == 0]
    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "missing": df.isna().sum(),
        "class_counts": df["Cover_Type"].value_counts().sort_index(),
        "summary": df[["Id"] + NUMERIC].describe(),
        "id_unique": df["Id"].nunique(),
        "id_monotonic": df["Id"].is_monotonic_increasing,
        "duplicate_rows": df[features].duplicated(keep=False).sum(),
        "duplicate_groups": grouped[grouped["rows"] > 1].describe(),
        "conflicting_groups": grouped[grouped["labels"] > 1].shape,
        "indicator_range": df[WILDERNESS + SOIL].agg(["min", "max"]),
        "wilderness_sums": indicator_checks["wilderness_sum"].value_counts(),
        "soil_sums": indicator_checks["soil_sum"].value_counts(),
        "indicator_prevalence": df[WILDERNESS + SOIL].mean(),
        "block_class_counts": annotated.groupby(["block", "Cover_Type"]).size(),
        "block_numeric_means": annotated.groupby("block")[NUMERIC].mean(),
        "block_indicator_means": annotated.groupby("block")[WILDERNESS + SOIL].mean(),
        "sample_shape": sample.shape,
        "sample_class_counts": sample["Cover_Type"].value_counts().sort_index(),
        "sample_numeric_summary": sample[NUMERIC].describe(),
        "sample_indicator_prevalence": sample[WILDERNESS + SOIL].mean(),
    }