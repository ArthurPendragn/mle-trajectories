import pandas as pd
import skrub

TRAIN_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"
TEST_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/test.csv"
SAMPLE_SUBMISSION_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/sample_submission.csv"


def build():
    train_op = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    test_op = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)
    sub_op = skrub.as_data_op(SAMPLE_SUBMISSION_PATH).skb.apply_func(pd.read_csv)

    # 1. Target distribution in train
    target_counts = (
        train_op.groupby("Cover_Type", dropna=False)
        .size()
        .to_frame(name="count")
        .reset_index()
        .sort_values(by="count", ascending=False)
    )

    # 2. Row counts
    train_row_count = train_op[["Id"]].count().to_frame(name="count").reset_index()
    test_row_count = test_op[["Id"]].count().to_frame(name="count").reset_index()
    sub_row_count = sub_op[["Id"]].count().to_frame(name="count").reset_index()

    # 3. Missing values check
    train_nulls = train_op.isna().sum().to_frame(name="null_count").reset_index()
    train_null_summary = train_nulls[["null_count"]].sum().to_frame(name="total_nulls").reset_index()
    train_nulls_non_zero = train_nulls[train_nulls["null_count"] > 0]

    test_nulls = test_op.isna().sum().to_frame(name="null_count").reset_index()
    test_null_summary = test_nulls[["null_count"]].sum().to_frame(name="total_nulls").reset_index()
    test_nulls_non_zero = test_nulls[test_nulls["null_count"] > 0]

    # 4. Column types
    train_dtypes = train_op.dtypes.astype(str).to_frame(name="dtype").reset_index()
    train_dtype_counts = train_dtypes.groupby("dtype").size().to_frame(name="column_count").reset_index()

    test_dtypes = test_op.dtypes.astype(str).to_frame(name="dtype").reset_index()
    test_dtype_counts = test_dtypes.groupby("dtype").size().to_frame(name="column_count").reset_index()

    # 5. Memory usage
    train_memory = train_op.memory_usage(deep=True).to_frame(name="memory_bytes").reset_index()
    train_total_memory = train_memory[["memory_bytes"]].sum().to_frame(name="total_memory_bytes").reset_index()

    test_memory = test_op.memory_usage(deep=True).to_frame(name="memory_bytes").reset_index()
    test_total_memory = test_memory[["memory_bytes"]].sum().to_frame(name="total_memory_bytes").reset_index()

    # 6. Min/Max inspection to check for constant columns or anomalous ranges
    train_min = train_op.min().to_frame(name="min").reset_index()
    train_max = train_op.max().to_frame(name="max").reset_index()
    train_stats = train_min.merge(train_max, on="index")
    train_constant_cols = train_stats[train_stats["min"] == train_stats["max"]]

    test_min = test_op.min().to_frame(name="min").reset_index()
    test_max = test_op.max().to_frame(name="max").reset_index()
    test_stats = test_min.merge(test_max, on="index")
    test_constant_cols = test_stats[test_stats["min"] == test_stats["max"]]

    # 7. Sample submission preview
    sub_head = sub_op.head(5)

    return {
        "train_target_counts": target_counts,
        "train_row_count": train_row_count,
        "train_null_summary": train_null_summary,
        "train_nulls_non_zero": train_nulls_non_zero,
        "train_dtype_counts": train_dtype_counts,
        "train_total_memory": train_total_memory,
        "train_constant_cols": train_constant_cols,
        "test_row_count": test_row_count,
        "test_null_summary": test_null_summary,
        "test_nulls_non_zero": test_nulls_non_zero,
        "test_dtype_counts": test_dtype_counts,
        "test_total_memory": test_total_memory,
        "test_constant_cols": test_constant_cols,
        "sample_submission_head": sub_head,
        "sample_submission_row_count": sub_row_count,
    }