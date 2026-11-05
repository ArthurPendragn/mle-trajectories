import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn the Title/Pclass age medians from the training rows."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(
            ["Title", "Pclass"], observed=False
        )["Age"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Title", "Pclass"]])
        medians = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return pd.DataFrame({"Age": X["Age"].fillna(medians)}, index=X.index)


class TicketCountEncoder(TransformerMixin, BaseEstimator):
    """Learn ticket frequencies, retaining the original fallback of one."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"Ticket_Count": X["Ticket"].map(self.counts_).fillna(1)},
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Provide a training-only eval set for CatBoost best-iteration selection."""

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


class CloneableCatBoostClassifier(CatBoostClassifier):
    def __sklearn_clone__(self):
        return CloneableCatBoostClassifier(**self.get_params(deep=False))


class ProbabilityAverageClassifier(ClassifierMixin, BaseEstimator):
    """Keep the original positive-column averaging and >= 0.5 decoding."""

    def __init__(self, cb_model, xgb_model):
        self.cb_model = cb_model
        self.xgb_model = xgb_model

    def fit(self, X, y, cb_X=None, cb_y=None, eval_set=None):
        self.cb_model_ = clone(self.cb_model)
        self.xgb_model_ = clone(self.xgb_model)
        self.cb_model_.fit(
            cb_X, cb_y, eval_set=eval_set, verbose=False
        )
        self.xgb_model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        cb_probs = self.cb_model_.predict_proba(X)[:, 1]
        xgb_probs = self.xgb_model_.predict_proba(X)[:, 1]
        return (((cb_probs + xgb_probs) / 2.0) >= 0.5).astype(int)


CAT_FEATURES = [
    "Pclass", "Sex", "Embarked", "Deck", "Title", "FamilyTier", "Sex_Pclass"
]
FEATURE_COLS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "LogFare", "Embarked",
    "Deck", "Title", "FamilySize", "FamilyTier", "Sex_Pclass", "Ticket_Count"
]


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read; importing this script performs no I/O.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and features. This reproduces the original
    #    stratified 80/20 outer hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    #    Reindex supplies missing optional Cabin/Name/Ticket columns. Their
    #    all-missing values produce the original Unknown/Other/1 fallbacks.
    optional = X.reindex(columns=["Cabin", "Name", "Ticket"])
    cabin = optional["Cabin"]
    cabin_text = cabin.astype(str)
    deck = cabin_text.str[0].where(
        cabin.notna() & cabin_text.ne(""), "Unknown"
    )

    extracted_title = optional["Name"].astype("string").str.extract(
        r" ([A-Za-z]+)\.", expand=False
    )
    title = extracted_title.where(
        extracted_title.isin(["Mr", "Miss", "Mrs", "Master"]), "Other"
    ).fillna("Other")

    features = X.assign(Deck=deck, Title=title)

    # The original learned these statistics before splitting. Separate fitted
    # nodes now learn group medians, the raw-Age global median, Fare median,
    # and ticket frequencies from training rows only.
    group_age = features[["Age", "Title", "Pclass"]].skb.apply(
        GroupAgeImputer(), no_wrap=True
    )
    global_age = X[["Age"]].skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True)
    )
    fare = X[["Fare"]].skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True)
    )
    ticket_count = optional[["Ticket"]].skb.apply(
        TicketCountEncoder(), no_wrap=True
    )

    family_size = X["SibSp"] + X["Parch"] + 1
    # Series.where requires a row-shaped condition, not scalar False.
    family_tier = family_size.astype(str).where(
        family_size.notna() & family_size.isna(), "Alone"
    )
    family_tier = family_tier.where(
        ~((family_size >= 2) & (family_size <= 4)), "SmallFamily"
    )
    family_tier = family_tier.where(
        ~(family_size >= 5), "LargeFamily"
    )

    features = features.assign(
        Age=group_age["Age"].fillna(global_age["Age"]),
        LogFare=fare["Fare"].clip(lower=0).skb.apply_func(np.log1p),
        Embarked=X["Embarked"].fillna("Missing"),
        FamilySize=family_size,
        FamilyTier=family_tier,
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str),
        Ticket_Count=ticket_count["Ticket_Count"],
    )
    features = features[FEATURE_COLS].astype(
        {
            "Pclass": "category",
            "Sex": "category",
            "Embarked": "category",
            "Deck": "category",
            "Title": "category",
            "FamilyTier": "category",
            "Sex_Pclass": "category",
        }
    )

    # Although the original specifies no stopping patience, CatBoost's eval_set
    # enables best-iteration selection by default. Keep that behavior, but use an
    # internal training-only eval set instead of leaking the scored hold-out.
    # XGBoost still fits on all outer-training rows.
    cb_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)

    cb_model = CloneableCatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=CAT_FEATURES,
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
    pred = features.skb.apply(
        ProbabilityAverageClassifier(cb_model, xgb_model),
        y=y,
        fit_kwargs={
            "cb_X": cb_parts["X"],
            "cb_y": cb_parts["y"],
            "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
        },
    )

    # 4. Score the single ensemble, not its components as separate variants.
    #    The splitter declared on mark_as_X drives validation.
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
