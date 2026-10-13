import pandas as pd
import skrub

PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"
NUMERIC = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
INDICATORS = ["Wilderness_Area%d" % i for i in range(1, 5)] + [
    "Soil_Type%d" % i for i in range(1, 41)
]
COUNTS = {1: 146814, 2: 226209, 3: 19571, 4: 38, 5: 1, 6: 1143, 7: 6226}


def build():
    full = skrub.as_data_op(PATH).skb.apply_func(pd.read_csv)
    pieces = []
    for label, n in COUNTS.items():
        rows = full[full["Cover_Type"] == label]
        pieces.append(rows.sample(n=n, random_state=42))
    sample = pieces[0].skb.concat(pieces[1:], axis=0).sort_values("Id").reset_index(drop=True)
    full_counts = full["Cover_Type"].value_counts().sort_index()
    sample_counts = sample["Cover_Type"].value_counts().sort_index()
    class_difference = sample_counts / sample.shape[0] - full_counts / full.shape[0]
    prevalence_difference = sample[INDICATORS].mean() - full[INDICATORS].mean()
    full_quantiles = full[NUMERIC].quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    sample_quantiles = sample[NUMERIC].quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    full_blocks = full.assign(inspection_block=full["Id"] // 800000)
    sample_blocks = sample.assign(inspection_block=sample["Id"] // 800000)
    full_table = full_blocks.groupby(["Cover_Type", "inspection_block"]).size().unstack(fill_value=0)
    sample_table = sample_blocks.groupby(["Cover_Type", "inspection_block"]).size().unstack(fill_value=0)
    full_distribution = full_table.div(full_table.sum(axis=1), axis=0)
    sample_distribution = sample_table.div(sample_table.sum(axis=1), axis=0)
    block_difference = sample_distribution - full_distribution
    return {
        "source_columns": full.columns,
        "sample_columns": sample.columns,
        "sample_shape": sample.shape,
        "sample_counts": sample_counts,
        "full_counts": full_counts,
        "class_proportion_difference": class_difference,
        "largest_class_discrepancy": class_difference.abs().max(),
        "sample_indicator_prevalence": sample[INDICATORS].mean(),
        "indicator_prevalence_difference": prevalence_difference,
        "largest_indicator_discrepancies": prevalence_difference.abs().sort_values(ascending=False),
        "full_numeric_quantiles": full_quantiles,
        "sample_numeric_quantiles": sample_quantiles,
        "numeric_quantile_difference": sample_quantiles - full_quantiles,
        "full_within_class_block_proportions": full_distribution,
        "sample_within_class_block_proportions": sample_distribution,
        "within_class_block_difference": block_difference,
        "largest_block_discrepancy": block_difference.abs().max().max(),
        "full_memory_bytes": full.memory_usage(index=True, deep=True).sum(),
        "sample_memory_bytes": sample.memory_usage(index=True, deep=True).sum(),
        "full_feature_float32_bytes": full.shape[0] * 54 * 4,
        "sample_feature_float32_bytes": sample.shape[0] * 54 * 4,
        "sample_id_bounds": sample["Id"].agg(["min", "max"]),
    }