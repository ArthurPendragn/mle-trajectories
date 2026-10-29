import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


FEATURE_COLS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "CabinDeck",
]


class CatBoostClassifierCloneable(CatBoostClassifier):
    """Bypass CatBoost's sklearn cloning bug without changing parameters."""

    def __sklearn_clone__(self):
        return CatBoostClassifierCloneable(**self.get_params(deep=False))


class GetXY(TransformerMixin, BaseEstimator):
    """Expose an inner split as a pandas table for this skrub version."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        indices = np.arange(len(X))
        train_indices, eval_indices = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        # Preserve the input index here for skrub's dataframe wrapper.
        # Recorded sorting below restores train_test_split's exact row order.
        split_order = np.argsort(
            np.concatenate([train_indices, eval_indices])
        )
        return X.assign(
            __inner_target=np.asarray(y),
            __inner_eval=np.isin(indices, eval_indices),
            __inner_order=split_order,
        )

    def transform(self, X):
        # At prediction time every incoming row goes directly to the model.
        return X.assign(
            __inner_target=None,
            __inner_eval=False,
            __inner_order=np.arange(len(X)),
        )


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read; importing this file reads no data.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and raw feature table.
    # This splitter reproduces the original stratified hold-out exactly.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — preserve feature names and column order.
    features = X.assign(
        Title=X["Name"].str.extract(
            r" ([A-Za-z]+)\.", expand=False
        ).fillna("Missing"),
        FamilySize=X["SibSp"] + X["Parch"] + 1,
        CabinDeck=X["Cabin"].str[0].fillna("Missing"),
        Embarked=X["Embarked"].fillna("Missing"),
    )
    features = features[FEATURE_COLS]

    # Learn Fare's median on each outer fold's training rows instead of the
    # whole dataset, fixing the original imputation leakage. Leave Age alone.
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Fare"]
    )

    # The original early-stopped on its scored hold-out. Keep its iteration
    # budget and patience, but carve a stratified 20% eval set out of the
    # outer fold's training rows instead.
    # This skrub version has no how="no_wrap": return a pandas table and
    # expose the split pieces through recorded filtering and column selection.
    parts = features.skb.apply(GetXY(), y=y)
    fit_rows = parts[~parts["__inner_eval"]].sort_values("__inner_order")
    eval_rows = parts[parts["__inner_eval"]].sort_values("__inner_order")

    X_fit = fit_rows[FEATURE_COLS]
    y_fit = fit_rows["__inner_target"]
    X_val = eval_rows[FEATURE_COLS]
    y_val = eval_rows["__inner_target"]

    model = CatBoostClassifierCloneable(
        iterations=600,
        learning_rate=0.03,
        depth=5,
        cat_features=["Pclass", "Sex", "Embarked", "Title", "CabinDeck"],
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0,
    )
    pred = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
        },
    )

    # 4. Score — the splitter on mark_as_X drives the single hold-out.
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
