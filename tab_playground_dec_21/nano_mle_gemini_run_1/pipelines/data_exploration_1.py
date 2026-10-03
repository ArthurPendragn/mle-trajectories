import pandas as pd
import skrub

TRAIN_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"
TEST_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/test.csv"


def build():
    train = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    test = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    # 1. Target distribution
    cover_type_counts = (
        train.groupby("Cover_Type")
        .agg(count=("Id", "count"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    # 2. Row counts
    train_row_count = (
        train[["Id"]].count().to_frame(name="row_count").reset_index(drop=True)
    )
    test_row_count = (
        test[["Id"]].count().to_frame(name="row_count").reset_index(drop=True)
    )

    # 3. Missing values
    train_missing = (
        train.isna()
        .sum()
        .to_frame(name="missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    train_missing_columns = train_missing[train_missing["missing_count"] > 0]
    train_total_missing = (
        train_missing.agg({"missing_count": "sum"})
        .to_frame(name="total_missing")
        .reset_index(drop=True)
    )

    test_missing = (
        test.isna()
        .sum()
        .to_frame(name="missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    test_missing_columns = test_missing[test_missing["missing_count"] > 0]
    test_total_missing = (
        test_missing.agg({"missing_count": "sum"})
        .to_frame(name="total_missing")
        .reset_index(drop=True)
    )

    # 4. Constant columns and ranges
    train_min = train.min().to_frame(name="min_val")
    train_max = train.max().to_frame(name="max_val")
    train_col_stats = (
        train_min.join(train_max)
        .reset_index()
        .rename(columns={"index": "column"})
    )
    train_constant_columns = train_col_stats[
        train_col_stats["min_val"] == train_col_stats["max_val"]
    ]

    test_min = test.min().to_frame(name="min_val")
    test_max = test.max().to_frame(name="max_val")
    test_col_stats = (
        test_min.join(test_max)
        .reset_index()
        .rename(columns={"index": "column"})
    )
    test_constant_columns = test_col_stats[
        test_col_stats["min_val"] == test_col_stats["max_val"]
    ]

    # 5. Memory usage and dtypes
    train_dtypes = (
        train.dtypes.astype(str)
        .to_frame(name="dtype")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    train_mem = (
        train.memory_usage(deep=True)
        .to_frame(name="bytes")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    train_mem_merged = train_mem.merge(train_dtypes, on="column", how="inner")
    train_memory_by_dtype = (
        train_mem_merged.groupby("dtype")
        .agg(num_columns=("column", "count"), total_bytes=("bytes", "sum"))
        .reset_index()
    )
    train_total_memory_bytes = (
        train_mem.agg({"bytes": "sum"})
        .to_frame(name="total_memory_bytes")
        .reset_index(drop=True)
    )

    test_dtypes = (
        test.dtypes.astype(str)
        .to_frame(name="dtype")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    test_mem = (
        test.memory_usage(deep=True)
        .to_frame(name="bytes")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    test_mem_merged = test_mem.merge(test_dtypes, on="column", how="inner")
    test_memory_by_dtype = (
        test_mem_merged.groupby("dtype")
        .agg(num_columns=("column", "count"), total_bytes=("bytes", "sum"))
        .reset_index()
    )
    test_total_memory_bytes = (
        test_mem.agg({"bytes": "sum"})
        .to_frame(name="total_memory_bytes")
        .reset_index(drop=True)
    )

    return {
        "cover_type_counts": cover_type_counts,
        "train_row_count": train_row_count,
        "test_row_count": test_row_count,
        "train_total_missing": train_total_missing,
        "test_total_missing": test_total_missing,
        "train_missing_columns": train_missing_columns,
        "test_missing_columns": test_missing_columns,
        "train_constant_columns": train_constant_columns,
        "test_constant_columns": test_constant_columns,
        "train_memory_by_dtype": train_memory_by_dtype,
        "test_memory_by_dtype": test_memory_by_dtype,
        "train_total_memory_bytes": train_total_memory_bytes,
        "test_total_memory_bytes": test_total_memory_bytes,
        "train_col_ranges": train_col_stats,
        "test_col_ranges": test_col_stats,
    }