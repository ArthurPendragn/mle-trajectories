import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set from the current fold's training rows."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        return {"X": X_fit, "X_val": X_val, "y": y_fit, "y_val": y_val}

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class XGBClassifier(XGBClassifier):
    """skrub calls predict() on an intermediate model.

    Return the (n_samples, n_classes) probability matrix. The blend takes
    column 1, the positive class, as it does for CatBoost.
    """

    def predict(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)


class LGBMClassifier(LGBMClassifier):
    """Same intermediate predict() contract as XGBClassifier."""

    def predict(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)


def collect_probabilities(cb_probs, xgb_probs, lgb_probs, mode):
    # Predictor nodes hold fitted estimators, not probabilities, in fit mode.
    # Each predict() returns (n_samples, 2); column 1 is the positive class.
    if mode == "fit":
        return None
    return np.column_stack(
        (
            np.asarray(cb_probs, dtype=float)[:, 1],
            np.asarray(xgb_probs, dtype=float)[:, 1],
            np.asarray(lgb_probs, dtype=float)[:, 1],
        )
    )


def optimized_blend_accuracy(y_true, probabilities):
    """Reproduce the original hold-out weight/threshold optimization."""
    cb_probs = probabilities[:, 0]
    xgb_probs = probabilities[:, 1]
    lgb_probs = probabilities[:, 2]

    best_score = 0.0
    best_weights = (1 / 3, 1 / 3, 1 / 3)
    best_threshold = 0.5

    # This is prediction decoding and scoring, not model fitting. Evaluate the
    # original grid on already-computed probabilities, without refitting the
    # three boosters thousands of times.
    for w_cb in np.linspace(0, 1, 11):
        for w_xgb in np.linspace(0, 1 - w_cb, 11):
            w_lgb = 1.0 - w_cb - w_xgb
            if w_lgb < -1e-5:
                continue
            w_lgb = max(0.0, w_lgb)
            blend_probs = (
                w_cb * cb_probs
                + w_xgb * xgb_probs
                + w_lgb * lgb_probs
            )

            for threshold in np.linspace(0.35, 0.65, 31):
                predictions = (blend_probs >= threshold).astype(int)
                score = accuracy_score(y_true, predictions)
                if score > best_score:
                    best_score = score
                    best_weights = (w_cb, w_xgb, w_lgb)
                    best_threshold = threshold

    ensemble_probs = (
        best_weights[0] * cb_probs
        + best_weights[1] * xgb_probs
        + best_weights[2] * lgb_probs
    )
    predictions = (ensemble_probs >= best_threshold).astype(int)
    return accuracy_score(y_true, predictions)


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CAT_FEATURES = ["Sex", "Embarked", "Pclass"]

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- the read is recorded, so importing does not access data.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and original feature matrix.
    # This splitter reproduces the original stratified train_test_split.
    y = data["Survived"].skb.mark_as_y()
    X = data[FEATURES].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and the original three models.
    features = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    # The original early-stopped on the scored hold-out. Preserve patience and
    # iteration limits, but use a stratified 20% of this fold's training rows
    # for early stopping instead. The original leaky score cannot be reproduced
    # exactly under this correction.
    parts = features.skb.apply(GetXY(), y=y, no_wrap=True)
    X_fit = parts["X"]
    y_fit = parts.get("y", y)
    X_eval = parts["X_val"]
    y_eval = parts["y_val"]

    cb_probs = X_fit.skb.apply(
        CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0,
        ),
        y=y_fit,
        fit_kwargs={
            "cat_features": CAT_FEATURES,
            "eval_set": (X_eval, y_eval),
            "early_stopping_rounds": 50,
            "verbose": False,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )

    xgb_probs = X_fit.skb.apply(
        XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            enable_categorical=True,
            tree_method="hist",
            random_state=42,
            eval_metric="logloss",
        ),
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_eval, y_eval)],
            "verbose": False,
        },
    )

    # The original LightGBM callbacks=[] did not enable early stopping.
    # Keep all outer training rows and all 1000 trees. Its eval_set only logged
    # metrics, so omit that logging-only argument rather than use scored rows.
    lgb_probs = features.skb.apply(
        LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
        y=y,
        fit_kwargs={
            "categorical_feature": CAT_FEATURES,
            "callbacks": [],
        },
    )

    pred = cb_probs.skb.apply_func(
        collect_probabilities,
        xgb_probs,
        lgb_probs,
        skrub.eval_mode(),
    )

    # 4. Score once: the original reports ONE optimized ensemble score, not a
    # model-comparison table. Its weight/threshold sweep belongs to the scoring
    # arithmetic and reuses the three fitted models' hold-out probabilities.
    # This retains its optimistic hold-out selection, exact grid, >= decoding,
    # and first-improvement tie handling without thousands of redundant fits.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring=make_scorer(
                optimized_blend_accuracy, response_method="predict"
            ),
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
