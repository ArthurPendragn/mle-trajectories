import lightgbm as lgb
import pandas as pd
import skrub
from skrub import selectors as s
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit, train_test_split


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set out of each fold's training rows."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        parts = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read without accessing the file on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and reproduce the original hold-out.
    y = data["y"].skb.mark_as_y()
    X = data.drop(columns=["ID", "y"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Preserve native categorical handling. Pandas 3 reads text as string
    #    dtype rather than object, so include both representations of CSV text.
    features = X.skb.apply(
        skrub.ToCategorical(),
        cols=s.has_dtype("object") | s.string(),
    )

    # The original early-stopped on the scored validation rows. Avoid that
    # leakage by retaining its 50-round patience and carving a separate 20%
    # eval set out of each outer fold's training rows.
    parts = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y,
        no_wrap=True,
    )
    X_fit = parts["X"]
    y_fit = parts.get("y", y)
    X_val = parts["X_val"]
    y_val = parts["y_val"]

    model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=4,
        num_leaves=15,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=2.0,
        reg_lambda=5.0,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    pred = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "callbacks": [
                lgb.early_stopping(stopping_rounds=50, verbose=False)
            ],
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
