import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
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


def inner_indices(X, y, fold):
    """The original stacking algorithm's internal, not outer, cross-fitting."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return list(splitter.split(X, np.asarray(y)))[fold]


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        self.fallback_ = self._group_fill(X).median()
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


class InnerFold(TransformerMixin, BaseEstimator):
    """Expose one internal stacking split as recorded dictionary components."""

    def __init__(self, fold):
        self.fold = fold

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        train_idx, valid_idx = inner_indices(X, y, self.fold)
        target = np.asarray(y)
        return {
            "X": X.iloc[train_idx],
            "target": target[train_idx],
            "X_oof": X.iloc[valid_idx],
        }

    def transform(self, X):
        return {"X": X, "target": None, "X_oof": None}


class GetXY(TransformerMixin, BaseEstimator):
    """Carve CatBoost's early-stopping rows out of its training rows."""

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
        return dict(zip(("X", "X_val", "target", "target_val"), parts))

    def transform(self, X):
        return {
            "X": X,
            "X_val": None,
            "target": None,
            "target_val": None,
        }


class InnerFoldEncoder(TransformerMixin, BaseEstimator):
    """One encoding step, fitted only on this inner fold's training rows."""

    def __init__(self, fold, kind):
        self.fold = fold
        self.kind = kind

    def fit(self, X, y):
        train_idx, _ = inner_indices(X, y, self.fold)
        if self.kind == "onehot":
            self.encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )
        else:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=0,
                dtype=np.int64,
            )
        self.encoder_.fit(X.iloc[train_idx])
        self.output_columns_ = self.encoder_.get_feature_names_out(X.columns)
        return self

    def transform(self, X):
        return pd.DataFrame(
            self.encoder_.transform(X),
            columns=self.output_columns_,
            index=X.index,
        )


class InnerFoldScaler(TransformerMixin, BaseEstimator):
    """The NN scaler is independent of its categorical encoder."""

    def __init__(self, fold):
        self.fold = fold

    def fit(self, X, y):
        train_idx, _ = inner_indices(X, y, self.fold)
        self.scaler_ = StandardScaler().fit(X.iloc[train_idx])
        self.columns_ = X.columns
        return self

    def transform(self, X):
        return pd.DataFrame(
            self.scaler_.transform(X), columns=self.columns_, index=X.index
        )


class CatBoostProbability(CatBoostClassifier):
    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def predict(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)[:, 1]


class XGBProbability(XGBClassifier):
    def predict(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)[:, 1]


class RFProbability(RandomForestClassifier):
    def predict(self, X):
        return self.predict_proba(X)[:, 1]


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


class TorchProbability(ClassifierMixin, BaseEstimator):
    """Only model construction and the original torch training loop."""

    def __init__(self, fold):
        self.fold = fold

    def fit(self, X, y):
        # The original seeded once, before its five sequential NN fits.
        if self.fold == 0:
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

        self.classes_ = np.unique(y)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cats = X[CAT_COLS].to_numpy(dtype=np.int64)
        numbers = X[NUM_COLS].to_numpy(dtype=np.float32)
        cat_dims = (cats.max(axis=0) + 1).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

        self.model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )

        t_cat = torch.tensor(cats, dtype=torch.long, device=self.device_)
        t_num = torch.tensor(numbers, dtype=torch.float32, device=self.device_)
        t_y = torch.tensor(np.asarray(y), dtype=torch.long, device=self.device_)

        for epoch in range(120):
            self.model_.train()
            permutation = torch.randperm(len(X))
            for start in range(0, len(X), 32):
                batch = permutation[start : start + 32]
                optimizer.zero_grad()
                loss = criterion(self.model_(t_cat[batch], t_num[batch]), t_y[batch])
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        cats = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        numbers = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(cats, numbers), dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def fold_probability_frame(prediction, oof_features, source, name, mode):
    """Training gets OOF predictions; scoring gets this fold's probabilities."""
    if mode == "fit":
        # A predictor node is its fitted estimator in fit mode. Predict ONLY
        # this internal fold's withheld rows, then align to the outer train set.
        return pd.DataFrame(
            {name: prediction.predict(oof_features)}, index=oof_features.index
        ).reindex(source.index, fill_value=0.0)
    return pd.DataFrame(
        {name: np.asarray(prediction) / 5.0}, index=source.index
    )


class ProbabilityCalibrator(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        self.calibrator_ = LogisticRegression(C=1.0, random_state=42).fit(X, y)
        self.columns_ = X.columns
        return self

    def transform(self, X):
        return pd.DataFrame(
            self.calibrator_.predict_proba(X)[:, 1],
            columns=self.columns_,
            index=X.index,
        )


class OptimizedThresholdClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.estimator_ = LogisticRegression(
            C=0.5, penalty="l2", random_state=42, max_iter=1000
        ).fit(X, y)
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
        return (
            self.estimator_.predict_proba(X)[:, 1] >= self.threshold_
        ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read; importing this file reads and fits nothing.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the RAW target. The original reported one
    #    stratified 80/20 hold-out score, not a mean of five outer CV scores.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Feature engineering. Separate learned preprocessing steps fix the
    #    original's full-table age/fare imputation and ticket-frequency leakage.
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
    string_categories = base_features[CAT_COLS].astype(str)
    numeric_features = base_features[NUM_COLS]
    cb_features = string_categories.skb.concat([numeric_features], axis=1)
    xgb_features = base_features[CAT_COLS].astype("category").skb.concat(
        [numeric_features], axis=1
    )

    # 4. Internal stacking cross-fitting. This loop BUILDS five recorded
    #    subgraphs; it does not execute an outer validation loop or fit a model.
    #    InnerFold selects unchanged raw labels, not a transformed target domain.
    #    The probability-frame helper explicitly handles fitted estimators in
    #    fit mode, and ordinary probability arrays in prediction mode.
    cb_frames, xgb_frames, rf_frames, nn_frames = [], [], [], []

    for fold in range(5):
        cb_parts = cb_features.skb.apply(
            InnerFold(fold), y=y, no_wrap=True
        )

        # The original early-stopped on its own OOF rows. Preserve patience=50
        # and iterations=600, but keep those OOF rows unseen: an additional 20%
        # eval set is carved from this inner fold's training rows.
        cb_eval = cb_parts["X"].skb.apply(
            GetXY(),
            y=cb_parts["target"],
            no_wrap=True,
        )
        cb_prediction = cb_eval["X"].skb.apply(
            CatBoostProbability(
                iterations=600,
                learning_rate=0.03,
                depth=5,
                cat_features=CAT_COLS,
                eval_metric="Accuracy",
                random_seed=42 + fold,
                verbose=0,
            ),
            y=cb_eval["target"],
            fit_kwargs={
                "eval_set": (cb_eval["X_val"], cb_eval["target_val"]),
                "early_stopping_rounds": 50,
            },
        )
        cb_frames.append(
            cb_prediction.skb.apply_func(
                fold_probability_frame,
                cb_parts["X_oof"],
                base_features,
                "CB",
                skrub.eval_mode(),
            )
        )

        xgb_parts = xgb_features.skb.apply(
            InnerFold(fold), y=y, no_wrap=True
        )
        xgb_prediction = xgb_parts["X"].skb.apply(
            XGBProbability(
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
            y=xgb_parts["target"],
        )
        xgb_frames.append(
            xgb_prediction.skb.apply_func(
                fold_probability_frame,
                xgb_parts["X_oof"],
                base_features,
                "XGB",
                skrub.eval_mode(),
            )
        )

        rf_categories = string_categories.skb.apply(
            InnerFoldEncoder(fold, "onehot"), y=y, no_wrap=True
        )
        rf_features = rf_categories.skb.concat([numeric_features], axis=1)
        rf_parts = rf_features.skb.apply(
            InnerFold(fold), y=y, no_wrap=True
        )
        rf_prediction = rf_parts["X"].skb.apply(
            RFProbability(
                n_estimators=300,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42 + fold,
            ),
            y=rf_parts["target"],
        )
        rf_frames.append(
            rf_prediction.skb.apply_func(
                fold_probability_frame,
                rf_parts["X_oof"],
                base_features,
                "RF",
                skrub.eval_mode(),
            )
        )

        nn_categories = string_categories.skb.apply(
            InnerFoldEncoder(fold, "ordinal"), y=y, no_wrap=True
        )
        nn_numbers = numeric_features.astype(np.float32).skb.apply(
            InnerFoldScaler(fold), y=y, no_wrap=True
        )
        nn_features = nn_categories.skb.concat([nn_numbers], axis=1)[FEATURE_COLS]
        nn_parts = nn_features.skb.apply(
            InnerFold(fold), y=y, no_wrap=True
        )
        nn_prediction = nn_parts["X"].skb.apply(
            TorchProbability(fold), y=nn_parts["target"]
        )
        nn_frames.append(
            nn_prediction.skb.apply_func(
                fold_probability_frame,
                nn_parts["X_oof"],
                base_features,
                "NN",
                skrub.eval_mode(),
            )
        )

    # Each training row has exactly one nonzero OOF contribution. At scoring,
    # each fold_probability_frame already contributes one fifth of its model's
    # probability, exactly matching the original fold-wise accumulation.
    cb_oof = cb_frames[0] + cb_frames[1] + cb_frames[2] + cb_frames[3] + cb_frames[4]
    xgb_oof = (
        xgb_frames[0] + xgb_frames[1] + xgb_frames[2]
        + xgb_frames[3] + xgb_frames[4]
    )
    rf_oof = rf_frames[0] + rf_frames[1] + rf_frames[2] + rf_frames[3] + rf_frames[4]
    nn_oof = nn_frames[0] + nn_frames[1] + nn_frames[2] + nn_frames[3] + nn_frames[4]

    # 5. Four independent calibrators, each fitted to the training OOF column.
    cb_calibrated = cb_oof.skb.apply(
        ProbabilityCalibrator(), y=y, no_wrap=True
    )
    xgb_calibrated = xgb_oof.skb.apply(
        ProbabilityCalibrator(), y=y, no_wrap=True
    )
    rf_calibrated = rf_oof.skb.apply(
        ProbabilityCalibrator(), y=y, no_wrap=True
    )
    nn_calibrated = nn_oof.skb.apply(
        ProbabilityCalibrator(), y=y, no_wrap=True
    )
    calibrated = cb_calibrated.skb.concat(
        [xgb_calibrated, rf_calibrated, nn_calibrated], axis=1
    )[PROB_COLS]

    # 6. Recorded meta-features. NumPy's variance and std use population ddof=0.
    meta_probabilities = calibrated.assign(
        prob_mean=calibrated.mean(axis=1),
        prob_variance=calibrated.var(axis=1, ddof=0),
        prob_std=calibrated.std(axis=1, ddof=0),
    )

    # get_dummies leaves numeric Pclass/IsAlone first, then Sex/Title dummies.
    # Learn and reuse the training dummy schema rather than rediscovering it
    # independently on the hold-out table.
    anchor_numbers = features[["Pclass", "IsAlone"]]
    anchor_categories = features[["Sex", "Title"]].skb.apply(
        OneHotEncoder(
            drop="first", handle_unknown="ignore", sparse_output=False
        )
    )
    meta_features = meta_probabilities.skb.concat(
        [anchor_numbers, anchor_categories], axis=1
    )

    # 7. Same meta-model and same 161-point training threshold search, including
    #    >= comparison and selection of the first threshold on a score tie.
    pred = meta_features.skb.apply(OptimizedThresholdClassifier(), y=y)

    # 8. Score the original single outer hold-out. No cv= here.
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
