import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from joblib import hash as joblib_hash
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# Threshold candidates share identical fitted models for each fold. Keep a
# process-local, in-memory cache so enumerating 61 thresholds does not train
# the same four-model ensemble 61 times. Nothing is read or fitted on import.
# Keys include the complete training matrix, target, and fold-specific seed.
_ENSEMBLE_FIT_CACHE = {}


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"Ticket_Freq": X["Ticket"].map(self.counts_).fillna(1)},
            index=X.index,
        )


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    def __init__(self, column, groups, fallback=None):
        self.column = column
        self.groups = groups
        self.fallback = fallback

    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.groups)[self.column].median()
        return self

    def transform(self, X):
        if len(self.groups) == 1:
            keys = pd.Index(X[self.groups[0]])
        else:
            keys = pd.MultiIndex.from_frame(X[self.groups])
        fill = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        result = X[self.column].fillna(fill)
        if self.fallback is not None:
            result = result.fillna(self.fallback)
        return result.to_frame(self.column)


class CrossFittedGroupSurvival(TransformerMixin, BaseEstimator):
    """Learn one lookup, cross-fitting its training-row encodings."""

    def __init__(self, key, output, filter_column=None):
        self.key = key
        self.output = output
        self.filter_column = filter_column

    def _learn(self, X, y):
        target = pd.Series(np.asarray(y), index=X.index, name="_target")
        table = X[[self.key]].assign(_target=target)
        if self.filter_column is not None:
            table = table[X[self.filter_column] == 1]
        return table.groupby(self.key)["_target"].mean()

    def fit(self, X, y):
        self.lookup_ = self._learn(X, y)
        return self

    def fit_transform(self, X, y, **fit_params):
        # This is the encoder's INNER cross-fitting, not the outer validation
        # loop. It is necessary to retain the original OOF encoding semantics.
        y_array = np.asarray(y)
        inner_cv = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=42
        )
        pieces = []
        for train_idx, val_idx in inner_cv.split(X, y_array):
            lookup = self._learn(X.iloc[train_idx], y_array[train_idx])
            pieces.append(
                X.iloc[val_idx][self.key].map(lookup).fillna(0.5)
            )
        result = pd.concat(pieces).reindex(X.index)
        self.fit(X, y)
        return result.to_frame(self.output)

    def transform(self, X):
        return (
            X[self.key].map(self.lookup_).fillna(0.5).to_frame(self.output)
        )


class FourModelProbabilityAverage(ClassifierMixin, BaseEstimator):
    """Fit the original four models and average positive-class probabilities."""

    def fit(self, X, y, split_y):
        # Full labels are used only to recover the original splitter's fold
        # number and therefore its 42 + fold seed, never to train a model.
        full_y = np.asarray(split_y)
        training_rows = np.sort(np.asarray(X.index, dtype=int))
        original_cv = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=42
        )
        fold_number = None
        for fold, (train_idx, _) in enumerate(
            original_cv.split(np.zeros((len(full_y), 1)), full_y)
        ):
            if np.array_equal(training_rows, train_idx):
                fold_number = fold
                break
        if fold_number is None:
            raise ValueError("Could not identify the original outer CV fold.")

        seed = 42 + fold_number
        self.classes_ = np.unique(y)
        cache_key = joblib_hash((X, np.asarray(y), seed))
        cached = _ENSEMBLE_FIT_CACHE.get(cache_key)
        if cached is not None:
            (
                self.lgb_model_,
                self.cb_model_,
                self.xgb_model_,
                self.lr_model_,
            ) = cached
            return self

        self.lgb_model_ = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
        )
        self.cb_model_ = CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=seed,
            verbose=0,
        )
        self.xgb_model_ = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=seed,
            eval_metric="logloss",
        )
        self.lr_model_ = LogisticRegression(
            penalty="l2",
            C=0.1,
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )

        self.lgb_model_.fit(X, y)
        self.cb_model_.fit(X, y)
        self.xgb_model_.fit(X, y)
        self.lr_model_.fit(X, y)

        _ENSEMBLE_FIT_CACHE[cache_key] = (
            self.lgb_model_,
            self.cb_model_,
            self.xgb_model_,
            self.lr_model_,
        )
        return self

    def predict(self, X):
        return (
            self.lgb_model_.predict_proba(X)[:, 1]
            + self.cb_model_.predict_proba(X)[:, 1]
            + self.xgb_model_.predict_proba(X)[:, 1]
            + self.lr_model_.predict_proba(X)[:, 1]
        ) / 4.0


def threshold_predictions(probabilities, threshold, mode):
    if mode == "fit":
        return probabilities
    return (probabilities >= threshold).astype(int)


TITLE_MAPPING = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Capt": "Officer",
    "Col": "Officer",
    "Don": "Royalty",
    "Dr": "Officer",
    "Major": "Officer",
    "Rev": "Officer",
    "Sir": "Royalty",
    "Jonkheer": "Royalty",
    "Dona": "Royalty",
}

FEATURE_COLS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Fare_Per_Person",
    "Embarked", "Title", "FamilySize", "IsAlone", "Deck", "Ticket_Freq",
    "WCG_Survival", "Ticket_Survival",
]
CAT_COLS = ["Sex", "Embarked", "Title", "Deck", "Pclass"]
NUM_COLS = [
    "Age", "SibSp", "Parch", "Fare", "Fare_Per_Person", "FamilySize",
    "IsAlone", "Ticket_Freq", "WCG_Survival", "Ticket_Survival",
]


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read.
    # Test loading, its existence guard, and submission production are omitted.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and raw design matrix.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing, separate learned lookups, and the model.
    # Counts and medians originally used combined train/test data. Learn them
    # on each outer training fold instead, eliminating that leakage.
    title = (
        X["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .map(TITLE_MAPPING)
        .fillna("Rare")
    )
    features = X.assign(Title=title)
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    features = features.assign(
        IsAlone=(features["FamilySize"] == 1).astype(int)
    )
    features = features.assign(
        Deck=features["Cabin"].astype(str).str[0].where(
            features["Cabin"].notna(), "U"
        )
    )

    frequency = features[["Ticket"]].skb.apply(
        TicketFrequency(), no_wrap=True
    )
    features = features.assign(Ticket_Freq=frequency["Ticket_Freq"])

    fare = features[["Pclass", "Fare"]].skb.apply(
        GroupMedianImputer(column="Fare", groups=["Pclass"]),
        no_wrap=True,
    )
    features = features.assign(Fare=fare["Fare"])
    features = features.assign(
        Fare_Per_Person=features["Fare"] / features["Ticket_Freq"]
    )

    age = features[["Title", "Pclass", "Age"]].skb.apply(
        GroupMedianImputer(
            column="Age", groups=["Title", "Pclass"], fallback=28.0
        ),
        no_wrap=True,
    )
    features = features.assign(Age=age["Age"])
    features = features.assign(Embarked=features["Embarked"].fillna("S"))
    surname = (
        features["Name"]
        .str.extract(r"^([^,]*),", expand=False)
        .str.strip()
        .fillna("Unknown")
    )
    features = features.assign(Surname=surname)
    features = features.assign(
        Is_Woman_Or_Child=(
            (features["Sex"] == "female")
            | (features["Age"] < 16)
            | (features["Title"] == "Master")
        ).astype(int)
    )

    # Preserve 10-fold OOF training encodings, now nested inside outer CV.
    # The original precomputed encodings leaked outer validation labels.
    wcg = features[["Surname", "Is_Woman_Or_Child"]].skb.apply(
        CrossFittedGroupSurvival(
            key="Surname",
            output="WCG_Survival",
            filter_column="Is_Woman_Or_Child",
        ),
        y=y,
        no_wrap=True,
    )
    ticket = features[["Ticket"]].skb.apply(
        CrossFittedGroupSurvival(
            key="Ticket", output="Ticket_Survival"
        ),
        y=y,
        no_wrap=True,
    )
    features = features.assign(
        WCG_Survival=wcg["WCG_Survival"],
        Ticket_Survival=ticket["Ticket_Survival"],
    )
    features = features[FEATURE_COLS]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUM_COLS,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ohe",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                ]),
                CAT_COLS,
            ),
        ]
    )
    transformed = features.skb.apply(preprocessor)
    probabilities = transformed.skb.apply(
        FourModelProbabilityAverage(),
        y=y,
        fit_kwargs={"split_y": data["Survived"]},
        no_wrap=True,
    )

    # Exactly the original 61 thresholds; caching above avoids repeating model
    # training for this prediction-only sweep. Model hyperparameters are intact.
    threshold = skrub.choose_from(
        np.linspace(0.35, 0.65, 61).tolist(),
        name="threshold",
    ).as_data_op()
    pred = probabilities.skb.apply_func(
        threshold_predictions, threshold, skrub.eval_mode()
    )

    # 4. Score -- no cv override.
    # skrub reports mean fold accuracy rather than pooled OOF accuracy.
    # Unequal fold sizes may slightly alter the selected threshold. The
    # leakage fixes also prevent exact reproduction of the original score.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(f"Optimal Threshold: {search.results_['threshold'].iloc[0]:.4f}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
