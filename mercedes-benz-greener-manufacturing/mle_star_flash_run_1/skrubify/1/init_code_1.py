import numpy as np
import pandas as pd
import stratum as skrub  # Drop-in DataOps API with the faster scheduler.
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.model_selection import ShuffleSplit, train_test_split


class AlignedDummies(TransformerMixin, BaseEstimator):
    """Learn training dummy columns, preserving independent get_dummies semantics."""

    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(include=["object"]).columns.tolist()
        self.columns_ = pd.get_dummies(
            X, columns=self.cat_cols_, drop_first=True
        ).columns
        return self

    def transform(self, X):
        # Like the original, independently drop the first category in this table,
        # then align to training columns. OneHotEncoder would differ when the
        # validation table does not contain the training reference category.
        return pd.get_dummies(
            X, columns=self.cat_cols_, drop_first=True
        ).reindex(columns=self.columns_, fill_value=0)


class GetXY(TransformerMixin, BaseEstimator):
    """Split only the outer fold's training rows for early stopping."""

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
    # 1. Load Data — importing this file records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and features. ShuffleSplit reproduces the original
    #    single train_test_split, including its row indices.
    y = data["y"].skb.mark_as_y()
    X = data.drop(columns=["ID", "y"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — learn dummy-column alignment per training fold.
    encoded = X.skb.apply(AlignedDummies())

    # Separate learned decomposition nodes preserve FeatureUnion's block order.
    pca = encoded.skb.apply(
        PCA(n_components=10, random_state=42).set_output(transform="pandas")
    )
    ica = encoded.skb.apply(
        FastICA(
            n_components=10, random_state=42, max_iter=1000, tol=1e-3
        ).set_output(transform="pandas")
    )
    tsvd = encoded.skb.apply(
        TruncatedSVD(n_components=10, random_state=42).set_output(
            transform="pandas"
        )
    )
    decomposition = pca.skb.concat([ica, tsvd], axis=1)
    features = encoded.astype(np.float32).skb.concat(
        [decomposition.astype(np.float32)], axis=1
    )

    # The original early-stopped on the scored validation rows. Keep its patience
    # and estimator count, but carve a 20% eval set from training rows only;
    # the original leaky validation score is therefore not exactly reproducible.
    pieces = features.skb.apply(
        GetXY(test_size=0.2, random_state=42), y=y, no_wrap=True
    )
    X_fit = pieces["X"]
    y_fit = pieces.get("y", y)
    X_eval = pieces["X_val"]
    y_eval = pieces["y_val"]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.005,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    pred = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={"eval_set": [(X_eval, y_eval)], "verbose": False},
    )

    # 4. Score — the splitter on mark_as_X drives the single hold-out.
    if __name__ == "__main__":
        with skrub.config(scheduler=True):
            search = pred.skb.make_grid_search(
                n_jobs=1, fitted=True, refit=False, scoring="r2"
            )
        results = search.results_
        print(results)
        for variant_score in results["scores"]:
            print(f"Variant score: {variant_score}")
        print(f"Final Validation Performance: {results['scores'][0]}")
