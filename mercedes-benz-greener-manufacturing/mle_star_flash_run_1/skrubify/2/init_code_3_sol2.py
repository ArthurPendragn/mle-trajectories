import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit, train_test_split
from skrub import selectors as s


class CatBoostRegressorCloneable(CatBoostRegressor):
    """Avoid CatBoost's sklearn cloning issues with learning_rate."""

    def __sklearn_clone__(self):
        return CatBoostRegressorCloneable(**self.get_params(deep=False))


class GetXY(TransformerMixin, BaseEstimator):
    """Split only the current fold's training rows for early stopping."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        parts = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the training CSV read.
    # Test loading, its existence guard, and submission generation are omitted:
    # none contributes to the validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and reproduce the original hold-out.
    y = data["y"].skb.mark_as_y()
    X = data.drop(columns=["ID", "y"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — use the original numeric-dtype predicate,
    # including its handling of booleans and other non-object dtypes.
    categorical_selector = s.filter(
        lambda col: not pd.api.types.is_numeric_dtype(col)
    )
    categorical = X.skb.select(categorical_selector)
    numeric = X.skb.select(~categorical_selector)
    categorical_strings = categorical.astype(str)
    features = numeric.skb.concat([categorical_strings], axis=1)
    features = features.reindex(columns=X.columns)

    # 4. Early stopping — the original used its scored validation rows as the
    # eval set. Instead, carve an equally proportioned inner split from this
    # fold's training rows, retaining the 50-round patience and 1200 iterations.
    # The scored hold-out is never used for early stopping.
    parts = features.skb.apply(GetXY(), y=y, no_wrap=True)
    X_fit = parts["X"]
    y_fit = parts.get("y", y)
    X_val = parts["X_val"]
    y_val = parts["y_val"]

    # Dynamic categorical column names are passed to fit rather than the
    # constructor; CatBoost's native categorical handling remains unchanged.
    # Patience is likewise passed through CatBoost's supported fit argument.
    model = CatBoostRegressorCloneable(
        iterations=1200,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.8,
        random_seed=42,
        eval_metric="RMSE",
        verbose=False,
    )
    pred = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "cat_features": categorical.columns.tolist(),
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
            "verbose": False,
        },
    )

    # 5. Score — the splitter on mark_as_X drives the single hold-out evaluation.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="r2"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
