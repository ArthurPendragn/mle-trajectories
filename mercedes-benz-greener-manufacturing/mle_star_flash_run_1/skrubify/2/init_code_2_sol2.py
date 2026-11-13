import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from skrub import selectors as s
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder


class GetXY(TransformerMixin, BaseEstimator):
    """Create an early-stopping holdout from the outer fold's training rows."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return {"X": X_fit, "X_val": X_val, "y": y_fit, "y_val": y_val}

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class NativePredictionXGBRegressor(xgb.XGBRegressor):
    """Match native Booster.predict, which uses all retained boosting rounds."""

    def predict(self, X, **kwargs):
        kwargs["iteration_range"] = (0, self.get_booster().num_boosted_rounds())
        return super().predict(X, **kwargs)


class UnionLabelEncoder(TransformerMixin, BaseEstimator):
    """Learn sorted string vocabularies using training rows and test metadata."""

    def fit(self, X, y=None, reference=None):
        # Each column needs its own vocabulary, just as with LabelEncoder.
        # Only state-dependent encoding is contained in this transformer.
        self.columns_ = list(X.columns)
        self.encoders_ = {}
        for col in self.columns_:
            values = X[[col]]
            if reference is not None and col in reference.columns:
                values = pd.concat(
                    [values, reference[[col]].astype(str)], ignore_index=True
                )
            self.encoders_[col] = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ).fit(values)
        return self

    def fit_transform(self, X, y=None, reference=None):
        return self.fit(X, y=y, reference=reference).transform(X)

    def transform(self, X):
        return pd.DataFrame(
            {
                col: self.encoders_[col].transform(X[[col]]).ravel()
                for col in self.columns_
            },
            index=X.index,
        )


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record reads without accessing files during import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)
    # The environment-dependent existence guard is dropped for the documented
    # test input. Its values remain relevant to the original encoding vocabulary;
    # test prediction, submission writing, and progress output are omitted.
    test_data = skrub.as_data_op("./input/test.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and features. ShuffleSplit reproduces the original
    # train_test_split's exact outer holdout.
    y = data["y"].skb.mark_as_y()
    X = data.drop(columns=["ID", "y"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Preserve dtype-dependent selection, string conversion, sorted label
    # vocabularies, original column order, numeric coercion, and zero filling.
    # Unlike the original, held-out training-table rows do not contribute to
    # vocabularies. External test-table vocabulary remains as originally used.
    # Previously unseen validation strings receive -1 rather than causing an
    # error after this fold-local leakage fix.
    categorical = X.skb.select(
        s.filter(lambda col: not pd.api.types.is_numeric_dtype(col))
    ).astype(str)
    encoded = categorical.skb.apply(
        UnionLabelEncoder(),
        no_wrap=True,
        fit_transform_kwargs={"reference": test_data},
    )
    numeric = X.skb.select(s.numeric())
    features = numeric.skb.concat([encoded], axis=1).reindex(columns=X.columns)
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    # The original early-stopped on its scored holdout. Keep its patience and
    # boosting budget, but carve an equally sized inner holdout from training
    # rows so the outer validation score does not guide training.
    pieces = features.skb.apply(GetXY(), y=y, no_wrap=True)
    X_fit = pieces["X"]
    y_fit = pieces.get("y", y)
    X_val = pieces["X_val"]
    y_val = pieces["y_val"]

    model = NativePredictionXGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.008,
        max_depth=3,
        subsample=0.75,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=5.0,
        random_state=42,
        tree_method="hist",
        n_estimators=1500,
        early_stopping_rounds=50,
    )
    # Native xgb.train retains the patience rounds, and native predict uses
    # them too; the subclass avoids sklearn's automatic best-iteration cutoff.
    pred = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_fit, y_fit), (X_val, y_val)],
            "verbose": False,
        },
    )

    # 4. Score — the splitter on mark_as_X drives the single R² holdout.
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
