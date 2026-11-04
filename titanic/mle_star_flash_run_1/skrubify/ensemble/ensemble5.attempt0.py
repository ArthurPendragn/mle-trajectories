import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS
PROB_COLS = ["CB", "XGB", "RF", "NN"]


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    """Learn the grouped medians and the subsequent overall fallback separately."""

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        grouped_age = self._group_fill(X)
        self.fallback_ = grouped_age.median()
        return self

    def _group_fill(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X["Age"].fillna(medians)

    def transform(self, X):
        return self._group_fill(X).fillna(self.fallback_).to_frame("Age")


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class GetXY(TransformerMixin, BaseEstimator):
    """Provide CatBoost an eval set drawn only from its own training rows."""

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
        return type(self)(**self.get_params(deep=False))


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n, d in zip(cat_dims, emb_dims)]
        )
        layers = []
        previous = sum(emb_dims) + num_features
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class TorchClassifier(ClassifierMixin, BaseEstimator):
    """Only the original neural-network construction and training loop."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cat_values = X[CAT_COLS].to_numpy(dtype=np.int64)
        num_values = X[NUM_COLS].to_numpy(dtype=np.float32)
        cat_dims = (cat_values.max(axis=0) + 1).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

        self.model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=num_values.shape[1]
        ).to(self.device_)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )

        t_cat = torch.tensor(cat_values, dtype=torch.long, device=self.device_)
        t_num = torch.tensor(num_values, dtype=torch.float32, device=self.device_)
        t_y = torch.tensor(np.asarray(y), dtype=torch.long, device=self.device_)

        for epoch in range(120):
            self.model_.train()
            permutation = torch.randperm(t_cat.size(0))
            for start in range(0, t_cat.size(0), 32):
                batch = permutation[start : start + 32]
                optimizer.zero_grad()
                loss = criterion(self.model_(t_cat[batch], t_num[batch]), t_y[batch])
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        t_cat = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        t_num = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(t_cat, t_num), dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def make_base_estimator(kind, fold):
    """Build an inspectable DataOps estimator for one inner-fold base model."""
    with skrub.config_context(eager_data_ops=False):
        base_X = skrub.var(
            "base_X", pd.DataFrame(columns=FEATURE_COLS)
        ).skb.mark_as_X()
        base_y = skrub.var(
            "base_y", pd.Series(dtype=np.int64)
        ).skb.mark_as_y()

        if kind == "CB":
            categories = base_X[CAT_COLS].astype(str)
            features = categories.skb.concat([base_X[NUM_COLS]], axis=1)

            # The original early-stopped on its OOF prediction rows. Those rows
            # must not guide fitting. Keep patience=50 and iterations=600, but
            # carve a stratified 20% eval set from this inner fold's TRAIN rows.
            pieces = features.skb.apply(GetXY(), y=base_y, no_wrap=True)
            prediction = pieces["X"].skb.apply(
                CloneableCatBoostClassifier(
                    iterations=600,
                    learning_rate=0.03,
                    depth=5,
                    cat_features=CAT_COLS,
                    eval_metric="Accuracy",
                    random_seed=42 + fold,
                    verbose=0,
                ),
                y=pieces.get("y", base_y),
                fit_kwargs={
                    "eval_set": (pieces["X_val"], pieces["y_val"]),
                    "early_stopping_rounds": 50,
                },
            )
        elif kind == "XGB":
            categories = base_X[CAT_COLS].astype("category")
            features = categories.skb.concat([base_X[NUM_COLS]], axis=1)
            prediction = features.skb.apply(
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    enable_categorical=True,
                    tree_method="hist",
                    random_state=42 + fold,
                    eval_metric="logloss",
                ),
                y=base_y,
            )
        elif kind == "RF":
            categories = base_X[CAT_COLS].astype(str).skb.apply(
                OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            )
            features = categories.skb.concat([base_X[NUM_COLS]], axis=1)
            prediction = features.skb.apply(
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=42 + fold,
                ),
                y=base_y,
            )
        else:
            # OrdinalEncoder uses sorted vocabularies, like the original
            # per-column LabelEncoders; unknown values map to index zero.
            categories = base_X[CAT_COLS].astype(str).skb.apply(
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=0,
                    dtype=np.int64,
                )
            )
            numbers = base_X[NUM_COLS].astype(np.float32).skb.apply(
                StandardScaler()
            )
            features = categories.skb.concat([numbers], axis=1)
            features = features[FEATURE_COLS]
            prediction = features.skb.apply(TorchClassifier(), y=base_y)

        return prediction.skb.make_estimator()


class FoldProbabilities(TransformerMixin, BaseEstimator):
    """OOF probabilities in training; five-model averaged probabilities at scoring."""

    def __init__(self, estimators, output_name, seed_torch=False):
        self.estimators = estimators
        self.output_name = output_name
        self.seed_torch = seed_torch

    def fit(self, X, y):
        # Seed once before the five sequential NN fits, not once per network.
        if self.seed_torch:
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

        y_values = np.asarray(y)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.models_ = []
        oof = np.empty(len(X), dtype=float)

        # This is the stacking algorithm's INTERNAL cross-fitting, not an
        # additional outer validation score. Its preprocessing remains exposed
        # in the individual DataOps estimators.
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y_values)):
            model = clone(self.estimators[fold])
            model.fit(X.iloc[train_idx], y_values[train_idx])
            oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
            self.models_.append(model)

        self.oof_ = pd.DataFrame(
            {self.output_name: oof}, index=X.index
        )
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.oof_

    def transform(self, X):
        probabilities = np.zeros(len(X), dtype=float)
        for model in self.models_:
            probabilities += model.predict_proba(X)[:, 1] / 5
        return pd.DataFrame(
            {self.output_name: probabilities}, index=X.index
        )


class ProbabilityCalibrator(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        self.calibrator_ = LogisticRegression(C=1.0, random_state=42)
        self.calibrator_.fit(X, y)
        self.columns_ = X.columns
        return self

    def transform(self, X):
        return pd.DataFrame(
            self.calibrator_.predict_proba(X)[:, 1],
            columns=self.columns_,
            index=X.index,
        )


class OptimizedThresholdClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        probabilities = self.estimator_.predict_proba(X)[:, 1]

        best_score = -1.0
        self.threshold_ = 0.5
        for threshold in np.linspace(0.1, 0.9, 161):
            score = accuracy_score(y, (probabilities >= threshold).astype(int))
            if score > best_score:
                best_score = score
                self.threshold_ = threshold
        return self

    def predict(self, X):
        probabilities = self.estimator_.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold_).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read; importing this file does not read or fit.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- the only reported score was the stratified 80/20
    #    hold-out. The five inner folds generate stacking features, not five
    #    additional reported variants or outer CV scores.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. The original learned age medians,
    #    ticket frequencies, and the fare median before the hold-out split.
    #    These separate learned steps now fit on the outer training rows only.
    age = X[["Pclass", "Sex", "Age"]].skb.apply(
        GroupedAgeImputer(), no_wrap=True
    )
    ticket_frequency = X[["Ticket"]].skb.apply(
        TicketFrequency(), no_wrap=True
    )
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))

    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = (
        raw_title.map(
            {
                "Mr": "Mr",
                "Miss": "Miss",
                "Mrs": "Mrs",
                "Master": "Master",
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs",
            }
        )
        .fillna("Rare")
        .where(raw_title.notna(), "Missing")
    )
    family_size = X["SibSp"] + X["Parch"] + 1
    family_tier = (
        family_size.map({1: "IsAlone"})
        .where(family_size == 1, "SmallFamily")
        .where(family_size <= 4, "LargeFamily")
    )

    features = X.assign(
        Title=title,
        Age=age["Age"],
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str),
        FamilySize=family_size,
        FamilyTier=family_tier,
        IsAlone=(family_size == 1).astype(int),
        HasCabin=X["Cabin"].notna().astype(int),
        TicketFreq=ticket_frequency["TicketFreq"],
        Embarked=X["Embarked"].fillna("Missing"),
        Fare=fare["Fare"],
    )
    base_features = features[FEATURE_COLS]

    # 4. Internal five-fold stacking. Each base learner has its own recorded
    #    preprocessing plan; OOF assembly and fold-model averaging need state.
    cb_models = tuple(make_base_estimator("CB", fold) for fold in range(5))
    xgb_models = tuple(make_base_estimator("XGB", fold) for fold in range(5))
    rf_models = tuple(make_base_estimator("RF", fold) for fold in range(5))
    nn_models = tuple(make_base_estimator("NN", fold) for fold in range(5))

    cb_probs = base_features.skb.apply(
        FoldProbabilities(cb_models, "CB"), y=y, no_wrap=True
    )
    xgb_probs = base_features.skb.apply(
        FoldProbabilities(xgb_models, "XGB"), y=y, no_wrap=True
    )
    rf_probs = base_features.skb.apply(
        FoldProbabilities(rf_models, "RF"), y=y, no_wrap=True
    )
    nn_probs = base_features.skb.apply(
        FoldProbabilities(nn_models, "NN", seed_torch=True),
        y=y,
        no_wrap=True,
    )

    # 5. Four independent probability calibrators, matching the original.
    cb_calibrated = cb_probs.skb.apply(
        ProbabilityCalibrator(), y=y, no_wrap=True
    )
    xgb_calibrated = xgb_probs.skb.apply(
        ProbabilityCalibrator(), y=y, no_wrap=True
    )
    rf_calibrated = rf_probs.skb.apply(
        ProbabilityCalibrator(), y=y, no_wrap=True
    )
    nn_calibrated = nn_probs.skb.apply(
        ProbabilityCalibrator(), y=y, no_wrap=True
    )
    calibrated = cb_calibrated.skb.concat(
        [xgb_calibrated, rf_calibrated, nn_calibrated], axis=1
    )[PROB_COLS]

    # 6. Recorded meta-features. NumPy's population variance/std use ddof=0.
    meta_probabilities = calibrated.assign(
        prob_mean=calibrated.mean(axis=1),
        prob_variance=calibrated.var(axis=1, ddof=0),
        prob_std=calibrated.std(axis=1, ddof=0),
    )

    # pd.get_dummies left Pclass and IsAlone numeric, followed by Sex and
    # Title dummy columns. Learn that dummy schema on training rows and reuse
    # it, rather than independently discovering a validation-table schema.
    anchor_numbers = features[["Pclass", "IsAlone"]]
    anchor_categories = features[["Sex", "Title"]].skb.apply(
        OneHotEncoder(
            drop="first", handle_unknown="ignore", sparse_output=False
        )
    )
    meta_features = meta_probabilities.skb.concat(
        [anchor_numbers, anchor_categories], axis=1
    )

    # 7. Same regularized meta-learner and same in-training threshold search.
    #    Thresholding stays inside the predictor, including >= and first-tie
    #    selection; there is no target transformation or prediction decoding.
    pred = meta_features.skb.apply(
        OptimizedThresholdClassifier(
            LogisticRegression(
                C=0.5, penalty="l2", random_state=42, max_iter=1000
            )
        ),
        y=y,
    )

    # 8. Score only the original outer hold-out. No cv= here.
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
