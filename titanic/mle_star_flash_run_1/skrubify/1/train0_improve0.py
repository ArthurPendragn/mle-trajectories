import numpy as np
import pandas as pd
import stratum as skrub  # Drop-in for skrub: same .skb API, faster evaluator.
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title"]
NUM_COLS = [
    "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin"
]


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Learn one grouped imputation and its post-group-fill fallback median."""

    def __init__(self, column, groups):
        self.column = column
        self.groups = groups

    def _group_values(self, X):
        lookup = self.medians_.drop_duplicates(subset=self.groups, keep="last")
        return X[self.groups].merge(
            lookup,
            on=self.groups,
            how="left",
            sort=False,
            validate="many_to_one",
        )["_group_median"].to_numpy()

    def fit(self, X, y=None):
        self.medians_ = (
            X.groupby(self.groups)[self.column]
            .median()
            .rename("_group_median")
            .reset_index()
        )
        grouped_values = pd.Series(self._group_values(X), index=X.index)
        # The reference computes the fallback AFTER filling grouped medians.
        self.fallback_ = X[self.column].fillna(grouped_values).median()
        return self

    def transform(self, X):
        grouped_values = pd.Series(self._group_values(X), index=X.index)
        return X.assign(
            **{
                self.column: X[self.column]
                .fillna(grouped_values)
                .fillna(self.fallback_)
            }
        )


class FoldLabelEncoder(TransformerMixin, BaseEstimator):
    """Per-column LabelEncoder with the reference's unknown-category code 0."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.encoders_ = {
            column: LabelEncoder().fit(X[column]) for column in self.columns_
        }
        return self

    def transform(self, X):
        result = {}
        for column in self.columns_:
            mapping = {
                value: i
                for i, value in enumerate(self.encoders_[column].classes_)
            }
            result[column] = X[column].map(mapping).fillna(0).astype(np.int64)
        return pd.DataFrame(result, index=X.index)


class GetXY(TransformerMixin, BaseEstimator):
    """Expose an inner early-stopping split through a pandas container."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        # This is an INNER split of this outer fold's training rows only.
        fit_indices, val_indices = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        # This API version lacks how="no_wrap", so return a pandas frame rather
        # than a dict. Recorded selections below expose all four split pieces.
        # Preserve train_test_split's row order using an explicit position key.
        positions = np.empty(len(X), dtype=np.int64)
        positions[fit_indices] = np.arange(len(fit_indices))
        positions[val_indices] = np.arange(len(val_indices))
        return X.assign(
            _eval_row=np.isin(np.arange(len(X)), val_indices),
            _split_position=positions,
            _target=np.asarray(y),
        )

    def transform(self, X):
        # Stable metadata columns also exist during prediction. Every prediction
        # row goes to X_fit; fit-only eval inputs are then empty and unused.
        return X.assign(
            _eval_row=False,
            _split_position=np.arange(len(X)),
            _target=np.nan,
        )


class CatBoostPositiveProbability(CatBoostClassifier):
    """Emit exactly predict_proba(X)[:, 1] from the predictor node."""

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def predict(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)[:, 1]


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n, d in zip(cat_dims, emb_dims)]
        )
        layers = []
        previous_dim = sum(emb_dims) + num_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            previous_dim = hidden_dim
        # Two classes are explicit in the reference architecture.
        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Keep the torch loop and its validation-driven learning-rate scheduler."""

    def __init__(
        self,
        batch_size=32,
        num_epochs=150,
        learning_rate=1e-2,
        weight_decay=1e-4,
        validation_size=0.2,
        random_state=42,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_size = validation_size
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        cat_dims = [int(X[column].max()) + 1 for column in CAT_COLS]
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]
        self.model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        # Legitimate INNER validation for the opaque torch training loop.
        # The reference scheduler used the scored hold-out; using only outer
        # training rows removes that leakage while keeping its scheduler.
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=y,
        )

        cat_train, num_train = self._tensors(X_train)
        cat_val, num_val = self._tensors(X_val)
        target_train = torch.tensor(
            np.asarray(y_train), dtype=torch.long, device=self.device_
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        # Keep all 150 epochs: the reference never restores a best checkpoint.
        for _ in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(cat_train.size(0))
            for start in range(0, cat_train.size(0), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(cat_train[batch], num_train[batch])
                loss = criterion(outputs, target_train[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_predictions = (
                    self.model_(cat_val, num_val).argmax(dim=1).cpu().numpy()
                )
            scheduler.step(accuracy_score(y_val, val_predictions))

        self.model_.eval()
        return self

    def _tensors(self, X):
        return (
            torch.tensor(
                X[CAT_COLS].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=self.device_,
            ),
            torch.tensor(
                X[NUM_COLS].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=self.device_,
            ),
        )

    def predict(self, X):
        x_cat, x_num = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            return (
                torch.softmax(self.model_(x_cat, x_num), dim=1)[:, 1]
                .cpu()
                .numpy()
            )


def ensemble_predictions(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    probabilities = 0.5 * cb_probability + 0.5 * nn_probability
    return (probabilities >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read. String extension dtypes avoid object
    #    columns at scheduler boundaries without changing category values.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(
        pd.read_csv,
        dtype={
            "Name": "string",
            "Sex": "string",
            "Ticket": "string",
            "Cabin": "string",
            "Embarked": "string",
        },
    )

    # 2. Prepare data -- mark the RAW target. This splitter reproduces the
    #    reference's single stratified train_test_split, including its rows.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, including the exact title buckets.
    title_raw = (
        X["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .fillna("Missing")
    )
    title = title_raw.replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )
    title = title.where(
        title.isin(["Mr", "Miss", "Mrs", "Master", "Missing"]), "Rare"
    )
    family_size = X["SibSp"] + X["Parch"] + 1
    features = X.assign(
        Title=title,
        FamilySize=family_size,
        IsAlone=(family_size == 1).astype(int),
        HasCabin=X["Cabin"].notna().astype(int),
        Embarked=X["Embarked"].fillna("Missing"),
    )[CAT_COLS + NUM_COLS]

    # Independent learned imputations are independent nodes. The reference
    # learned these on the full table; per-fold fitting fixes that leakage.
    features = features.skb.apply(
        GroupMedianImputer(column="Age", groups=["Title", "Pclass"])
    )
    features = features.skb.apply(
        GroupMedianImputer(column="Fare", groups=["Pclass"])
    )

    categorical = features[CAT_COLS].astype("str")
    cb_features = categorical.skb.concat([features[NUM_COLS]], axis=1)

    # 4. CatBoost -- retain all parameters and early-stopping patience.
    #    The scored hold-out cannot also be an honest early-stopping set.
    #    GetXY instead exposes an inner 20% split of outer training rows.
    #    Frame output supports this installed API, which lacks how="no_wrap".
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_fit = cb_parts[~cb_parts["_eval_row"]].sort_values("_split_position")
    cb_eval = cb_parts[cb_parts["_eval_row"]].sort_values("_split_position")

    cb_probability = cb_fit[CAT_COLS + NUM_COLS].skb.apply(
        CatBoostPositiveProbability(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_fit["_target"],
        fit_kwargs={
            "eval_set": (
                cb_eval[CAT_COLS + NUM_COLS],
                cb_eval["_target"],
            ),
            "early_stopping_rounds": 50,
        },
    )

    # 5. Neural network -- keep encoding and scaling as visible fitted nodes.
    #    Preserve float32 input to StandardScaler, not a cast of the raw target.
    nn_categorical = categorical.skb.apply(FoldLabelEncoder())
    nn_numeric = (
        features[NUM_COLS]
        .astype(np.float32)
        .skb.apply(StandardScaler())
    )
    nn_features = nn_categorical.skb.concat([nn_numeric], axis=1)
    nn_probability = nn_features.skb.apply(
        NeuralPositiveProbability(), y=y
    )

    # 6. Preserve the manual probability average and >= 0.5 threshold.
    #    Gate arithmetic because fit-mode predictor values are estimators.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 7. Score the single ensemble. The splitter on mark_as_X drives.
    if __name__ == "__main__":
        with skrub.config(scheduler=True):
            search = pred.skb.make_grid_search(
                n_jobs=1, fitted=True, refit=False, scoring="accuracy"
            )
        results = search.results_
        print(results)
        for variant_score in results["scores"]:
            print(f"Variant score: {variant_score}")
        print(f"Final Validation Performance: {results['scores'][0]}")
