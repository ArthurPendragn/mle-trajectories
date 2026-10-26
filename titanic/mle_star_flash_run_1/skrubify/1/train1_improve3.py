import copy

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    """Learn the grouped medians and subsequent global fallback."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        self.fallback_ = X["Age"].fillna(grouped).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X.assign(Age=X["Age"].fillna(grouped).fillna(self.fallback_))


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"TicketFreq": X["Ticket"].map(self.counts_).fillna(1)},
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Expose an inner split through pandas-compatible output columns.

    This skrub version does not support apply(how="no_wrap"). Returning a
    same-index DataFrame instead of a dict works with its transformer wrapper.
    Actual train/eval slicing and ordering remain recorded operations.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        train_idx, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        is_eval = np.zeros(len(X), dtype=bool)
        is_eval[val_idx] = True
        split_order = np.empty(len(X), dtype=np.int64)
        split_order[train_idx] = np.arange(len(train_idx))
        split_order[val_idx] = np.arange(len(val_idx))
        return X.assign(
            _inner_eval=is_eval,
            _inner_order=split_order,
            _inner_y=np.asarray(y),
        )

    def transform(self, X):
        # All prediction rows pass through, in their original order.
        return X.assign(
            _inner_eval=False,
            _inner_order=np.arange(len(X)),
            _inner_y=None,
        )


class CatBoostPositiveProbability(CatBoostClassifier):
    """Expose the probability consumed by the original ensemble."""

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def predict(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)[:, 1]


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


class SnapshotNeuralClassifier(ClassifierMixin, BaseEstimator):
    """Original five-seed torch training and snapshot averaging."""

    def __init__(
        self,
        seeds=(42, 123, 456, 789, 2024),
        num_epochs=100,
        batch_size=32,
        warmup_epochs=25,
    ):
        self.seeds = seeds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs

    def _tensors(self, X):
        categorical = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        numeric = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        return categorical, numeric

    def fit(self, X, y, *, X_val, y_val, cat_dims):
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.classes_ = np.unique(y)
        dimensions = np.asarray(cat_dims, dtype=np.int64).tolist()
        embedding_dimensions = [
            max(2, int(round(dimension**0.5 * 2)))
            for dimension in dimensions
        ]
        train_cat, train_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_y = torch.tensor(
            np.asarray(y_val), dtype=torch.long, device=self.device_
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.seed_models_ = []

        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = TabularNeuralNet(
                dimensions,
                embedding_dimensions,
                num_features=len(NUM_COLS),
            ).to(self.device_)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-3, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=25, T_mult=1, eta_min=1e-5
            )
            snapshots = []
            best_val_loss = float("inf")
            best_state = None

            for epoch in range(self.num_epochs):
                model.train()
                permutation = torch.randperm(train_cat.size(0))

                for start in range(0, train_cat.size(0), self.batch_size):
                    batch = permutation[start : start + self.batch_size]
                    optimizer.zero_grad()
                    outputs = model(train_cat[batch], train_num[batch])
                    loss = criterion(outputs, train_y[batch])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=2.0
                    )
                    optimizer.step()

                scheduler.step()
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(
                        model(val_cat, val_num), val_y
                    ).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())

                # Preserve the original update order and checkpoint condition.
                if epoch >= self.warmup_epochs and (
                    (epoch + 1) % 25 == 0 or val_loss <= best_val_loss
                ):
                    snapshots.append(copy.deepcopy(model.state_dict()))

            if not snapshots:
                snapshots = [best_state]
            self.seed_models_.append((model, snapshots[-4:]))

        return self

    def predict_proba(self, X):
        categorical, numeric = self._tensors(X)
        all_seed_probs = []

        for model, snapshots in self.seed_models_:
            snapshot_probs = []
            for state in snapshots:
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    probabilities = torch.softmax(
                        model(categorical, numeric), dim=1
                    )[:, 1].cpu().numpy()
                snapshot_probs.append(probabilities)
            all_seed_probs.append(np.mean(snapshot_probs, axis=0))

        positive = np.mean(all_seed_probs, axis=0)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def blend_predictions(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    return (0.5 * cb_probability + 0.5 * nn_probability >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read; importing does not access the file.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark raw target and features. This splitter reproduces the original
    #    stratified 80/20 train_test_split's outer row indices.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Feature engineering. Fold-fitted statistics fix the original's
    #    full-table age/fare imputation and ticket-frequency leakage.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = raw_title.map(
        {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        }
    ).fillna("Rare")
    title = title.where(raw_title.notna(), "Missing")

    age = X[["Pclass", "Sex", "Age"]].skb.apply(GroupedAgeImputer())
    tickets = X[["Ticket"]].skb.apply(TicketFrequency())
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))

    features = X.assign(Title=title)
    features = features.assign(Age=age["Age"])
    features = features.assign(
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
    )
    features = features.assign(FamilySize=X["SibSp"] + X["Parch"] + 1)
    family_tier = (features["FamilySize"] <= 4).map(
        {True: "SmallFamily", False: "LargeFamily"}
    )
    family_tier = family_tier.where(
        features["FamilySize"] != 1, "IsAlone"
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=X["Cabin"].notna().astype(int))
    features = features.assign(TicketFreq=tickets["TicketFreq"])
    features = features.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.assign(Fare=fare["Fare"])

    categorical = features[CAT_COLS].astype(str)
    numeric = features[NUM_COLS]
    cb_features = categorical.skb.concat([numeric], axis=1)

    # 4. Keep CatBoost's exact parameters and early-stopping patience.
    #    Its original eval set was the scored hold-out. Use an inner 20%
    #    stratified split instead: reproducing that leaky score is not possible.
    #    The split is exposed as recorded filtering and fit_kwargs, not hidden
    #    inside a predictor wrapper. No unsupported how= argument is used.
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_fit = cb_parts[~cb_parts["_inner_eval"]].sort_values("_inner_order")
    cb_eval = cb_parts[cb_parts["_inner_eval"]].sort_values("_inner_order")

    cb_probability = cb_fit[FEATURE_COLS].skb.apply(
        CatBoostPositiveProbability(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_fit["_inner_y"],
        fit_kwargs={
            "eval_set": (cb_eval[FEATURE_COLS], cb_eval["_inner_y"]),
            "early_stopping_rounds": 50,
        },
    )

    # 5. Separate neural preprocessing. Sorted OrdinalEncoder codes match the
    #    individual LabelEncoders, including code 0 for unseen categories.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=0,
            dtype=np.int64,
        )
    )
    nn_numeric = numeric.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_categorical.skb.concat([nn_numeric], axis=1)

    # Fit-only embedding dimensions come from the outer training rows.
    cat_dimensions = nn_categorical.max() + 1

    # 6. Keep the torch loop and five-seed snapshot ensemble. The original
    #    selected snapshots using scored rows; expose an inner validation split
    #    instead, retaining the same snapshot-selection arithmetic.
    nn_parts = nn_features.skb.apply(GetXY(), y=y)
    nn_fit = nn_parts[~nn_parts["_inner_eval"]].sort_values("_inner_order")
    nn_eval = nn_parts[nn_parts["_inner_eval"]].sort_values("_inner_order")

    nn_probability = nn_fit[FEATURE_COLS].skb.apply(
        SnapshotNeuralClassifier(),
        y=nn_fit["_inner_y"],
        fit_kwargs={
            "X_val": nn_eval[FEATURE_COLS],
            "y_val": nn_eval["_inner_y"],
            "cat_dims": cat_dimensions,
        },
    )

    # 7. Preserve the original probability average and >= 0.5 threshold.
    #    Prediction arithmetic must not run on fit-mode estimator objects.
    pred = cb_probability.skb.apply_func(
        blend_predictions, nn_probability, skrub.eval_mode()
    )

    # 8. Score the single original ensemble using outer hold-out accuracy.
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
