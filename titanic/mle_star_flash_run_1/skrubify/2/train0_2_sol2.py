import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


NUM_COLS = ["Age", "Fare", "SibSp", "Parch"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]
FEATURES = NUM_COLS + CAT_COLS


class CatBoostClassifierCloneable(CatBoostClassifier):
    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))


class GetXY(TransformerMixin, BaseEstimator):
    """Provide CatBoost's evaluation rows from the current training fold."""

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
            stratify=y,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class ProbabilityAverageClassifier(ClassifierMixin, BaseEstimator):
    """Keep the original column-1 averaging and >= 0.5 decoding."""

    def __init__(self, cb_model, xgb_model, hgb_model):
        self.cb_model = cb_model
        self.xgb_model = xgb_model
        self.hgb_model = hgb_model

    def fit(self, X, y, cb_X=None, cb_y=None, eval_set=None):
        self.classes_ = np.unique(y)
        self.cb_model_ = clone(self.cb_model)
        self.xgb_model_ = clone(self.xgb_model)
        self.hgb_model_ = clone(self.hgb_model)

        self.cb_model_.fit(
            cb_X, cb_y, eval_set=eval_set, verbose=False
        )
        self.xgb_model_.fit(X, y)
        self.hgb_model_.fit(X, y)
        return self

    def predict(self, X):
        cb_probs = self.cb_model_.predict_proba(X)[:, 1]
        xgb_probs = self.xgb_model_.predict_proba(X)[:, 1]
        hgb_probs = self.hgb_model_.predict_proba(X)[:, 1]
        probabilities = (cb_probs + xgb_probs + hgb_probs) / 3.0
        return (probabilities >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this module records, but does not read, the CSV.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and reproduce the stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data[FEATURES].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — retain the original feature order and dtypes.
    features = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    # CatBoost's original eval_set was the scored hold-out. Even without explicit
    # early-stopping patience, that enables best-iteration selection by default.
    # Keep the eval_set, but carve it from the outer fold's training rows instead;
    # the original leaky selection score cannot be reproduced honestly.
    cb_parts = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y,
        no_wrap=True,
    )

    cb_model = CatBoostClassifierCloneable(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=CAT_COLS,
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0,
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
    )

    hgb_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUM_COLS),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CAT_COLS,
            ),
        ]
    )
    hgb_model = Pipeline(
        [
            ("preprocessor", hgb_preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                ),
            ),
        ]
    )

    # This is one scored ensemble, not three model-comparison variants.
    # Its wrapper preserves the exact probability arithmetic and threshold.
    pred = features.skb.apply(
        ProbabilityAverageClassifier(cb_model, xgb_model, hgb_model),
        y=y,
        fit_kwargs={
            "cb_X": cb_parts["X"],
            "cb_y": cb_parts["y"],
            "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
        },
    )

    # 4. Score — the splitter on mark_as_X owns the outer validation.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
