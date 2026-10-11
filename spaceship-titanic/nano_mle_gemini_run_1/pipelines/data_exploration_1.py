import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import make_pipeline
import skrub
from skrub import TableVectorizer

TRAIN_PATH = "/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv"
TEST_PATH = "/home/estrauss-ldap/datasets/space-ship-titanic/input/test.csv"


def build():
    train = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    test = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    train_head = train.head(10)
    test_head = test.head(10)

    train_counts = train.count()
    test_counts = test.count()

    target_counts = train["Transported"].value_counts(dropna=False)
    target_distribution = train["Transported"].value_counts(
        normalize=True, dropna=False
    )

    train_null_counts = train.isna().sum()
    test_null_counts = test.isna().sum()
    train_null_fraction = train.isna().mean()
    test_null_fraction = test.isna().mean()

    train_numeric_summary = train.describe()
    test_numeric_summary = test.describe()

    train_groups = (
        train[["PassengerId"]]
        .assign(group=train["PassengerId"].str.slice(0, 4))[["group"]]
        .drop_duplicates()
    )
    test_groups = (
        test[["PassengerId"]]
        .assign(group=test["PassengerId"].str.slice(0, 4))[["group"]]
        .drop_duplicates()
    )
    common_groups = train_groups.merge(test_groups, on="group", how="inner")

    train_unique_groups = train_groups.count()
    test_unique_groups = test_groups.count()
    common_unique_groups = common_groups.count()

    train_passengers_per_group = (
        train["PassengerId"].str.slice(0, 4).value_counts().describe()
    )
    test_passengers_per_group = (
        test["PassengerId"].str.slice(0, 4).value_counts().describe()
    )

    train_adv_features = train.drop(
        columns=["Transported", "PassengerId", "Name"], errors="ignore"
    ).assign(is_test=0)
    test_adv_features = test.drop(
        columns=["PassengerId", "Name"], errors="ignore"
    ).assign(is_test=1)
    adv_combined = train_adv_features.skb.concat(
        [test_adv_features], axis=0
    ).reset_index(drop=True)

    X_adv = adv_combined.drop(columns=["is_test"])
    y_adv = adv_combined["is_test"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    adv_pipe = make_pipeline(
        TableVectorizer(cardinality_threshold=40),
        HistGradientBoostingClassifier(random_state=42),
    )
    adversarial_cv = skrub.as_data_op(adv_pipe).skb.apply_func(
        cross_validate, X_adv, y_adv, cv=cv, scoring="roc_auc"
    )

    y_train = train["Transported"].astype(int)
    X_train = train.drop(
        columns=["Transported", "PassengerId", "Name"], errors="ignore"
    )

    baseline_pipe = make_pipeline(
        TableVectorizer(cardinality_threshold=40),
        HistGradientBoostingClassifier(random_state=42),
    )
    baseline_cv = skrub.as_data_op(baseline_pipe).skb.apply_func(
        cross_validate, X_train, y_train, cv=cv, scoring="accuracy"
    )

    curve = skrub.as_data_op(baseline_pipe).skb.apply_func(
        learning_curve,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        train_sizes=[0.25, 0.5, 0.75, 1.0],
        return_times=True,
    )

    return {
        "train_head": train_head,
        "test_head": test_head,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "target_counts": target_counts,
        "target_distribution": target_distribution,
        "train_null_counts": train_null_counts,
        "test_null_counts": test_null_counts,
        "train_null_fraction": train_null_fraction,
        "test_null_fraction": test_null_fraction,
        "train_numeric_summary": train_numeric_summary,
        "test_numeric_summary": test_numeric_summary,
        "train_unique_groups": train_unique_groups,
        "test_unique_groups": test_unique_groups,
        "common_unique_groups": common_unique_groups,
        "train_passengers_per_group": train_passengers_per_group,
        "test_passengers_per_group": test_passengers_per_group,
        "adversarial_cv": adversarial_cv,
        "baseline_cv": baseline_cv,
        "learning_curve": curve,
    }