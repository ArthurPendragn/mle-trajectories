import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit, train_test_split


class CatBoostRegressorCloneable(CatBoostRegressor):
    """Bypass CatBoost's constructor-parameter cloning issues."""

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))


class GetXY(TransformerMixin, BaseEstimator):
    """Reserve early-stopping rows from the current fold's training data."""

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
    # 1. Load Data — importing this file only records the CSV read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — mark the raw target and preserve the original hold-out.
    y = data["y"].skb.mark_as_y()
    X = data.drop(columns=["ID", "y"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — only object columns become strings, exactly
    #    as in the original. Restore the original feature order after concat.
    categorical = X.select_dtypes(include=["object"])
    cat_cols = categorical.columns.tolist()
    features = (
        X.select_dtypes(exclude=["object"])
        .skb.concat([categorical.astype(str)], axis=1)
        .reindex(columns=X.columns)
    )

    # The original early-stopped on its scored validation rows. Keep the same
    # patience, but reserve an inner 20% of the fold's own training rows instead;
    # the outer scored rows must not influence early stopping.
    parts = features.skb.apply(GetXY(test_size=0.2, random_state=42), y=y, no_wrap=True)
    X_fit = parts["X"]
    y_fit = parts.get("y", y)
    X_val = parts["X_val"]
    y_val = parts["y_val"]

    # cat_features is supplied at fit time so the recorded, dtype-dependent
    # column list stays dynamic. All original model hyperparameters are kept.
    model = CatBoostRegressorCloneable(
        iterations=1500,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=5.0,
        loss_function="RMSE",
        eval_metric="R2",
        random_seed=42,
        verbose=False,
    )
    pred = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "cat_features": cat_cols,
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
        },
    )

    # 4. Score — the splitter on mark_as_X drives the single hold-out.
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
