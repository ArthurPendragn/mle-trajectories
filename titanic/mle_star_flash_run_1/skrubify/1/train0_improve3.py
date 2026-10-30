import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import stratum as skrub  # Drop-in DataOps API with the faster scheduler.

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck", "Sex_Pclass"]
NUM_COLS = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "GroupSize",
    "FarePerPerson",
    "LogFarePerPerson",
    "IsChild",
    "IsMother",
]
FEATURE_COLS = CAT_COLS + NUM_COLS

TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Ms": "Miss",
    "Capt": "Rare",
    "Lady": "Rare",
    "Jonkheer": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
    "Countess": "Rare",
    "Sir": "Rare",
}


class MedianFill(TransformerMixin, BaseEstimator):
    """Learn column medians without dropping an entirely missing column."""

    def fit(self, X, y=None):
        self.medians_ = X.median()
        return self

    def transform(self, X):
        return X.fillna(self.medians_)


class GetXY(TransformerMixin, BaseEstimator):
    """Expose an inner split as a pandas frame for the installed .skb API."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        # This is an INNER, per-fit validation split, not the outer scored
        # hold-out. The installed API requires a frame rather than a dict.
        train_indices, validation_indices = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        labeled = X.assign(_inner_target=np.asarray(y))
        training = labeled.iloc[train_indices].assign(_inner_validation=False)
        validation = labeled.iloc[validation_indices].assign(
            _inner_validation=True
        )
        # Preserve train_test_split's ordering within each piece.
        return pd.concat([training, validation], axis=0)

    def transform(self, X):
        # Prediction keeps every incoming row in its original order.
        # Keep helper columns present if fit-only nodes are visited.
        return X.assign(_inner_target=np.nan, _inner_validation=False)


class CatBoostPositiveProbability(CatBoostClassifier):
    """Expose exactly the positive-class probability used by the ensemble."""

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def predict(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)[:, 1]


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, e) for n, e in zip(cat_dims, emb_dims)]
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


class TorchPositiveProbability(ClassifierMixin, BaseEstimator):
    """Only the original torch training loop and probability prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dimensions = np.asarray(cat_dims, dtype=int).tolist()
        embedding_dimensions = [
            max(2, int(round(dim**0.5 * 2))) for dim in dimensions
        ]
        self.model_ = TabularNeuralNet(
            dimensions, embedding_dimensions, num_features=len(NUM_COLS)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        train_cat = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        train_num = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        validation_cat = torch.tensor(
            X_val[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        validation_num = torch.tensor(
            X_val[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )

        # Keep all 150 epochs: the reference did not restore a best checkpoint.
        for epoch in range(150):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for start in range(0, train_cat.size(0), 32):
                batch = permutation[start : start + 32]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[batch], train_num[batch])
                loss = criterion(outputs, train_y[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(validation_cat, validation_num)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                validation_accuracy = accuracy_score(y_val, predictions)
            scheduler.step(validation_accuracy)

        self.model_.eval()
        return self

    def predict_proba(self, X):
        categorical = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        numerical = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(categorical, numerical)
            return torch.softmax(outputs, dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def ensemble_predictions(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    return (
        0.5 * np.asarray(cb_probability) + 0.5 * np.asarray(nn_probability) >= 0.5
    ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records, but does not execute, the CSV read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and features. This reproduces the original
    #    stratified 80/20 outer hold-out, including its random seed.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Median imputation now learns from each
    #    outer fold's training rows, fixing the original full-table leak.
    ages = X[["Age"]].skb.apply(MedianFill())
    fares = X[["Fare"]].skb.apply(MedianFill())

    raw_title = (
        X["Name"]
        .astype("string")
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .fillna("Missing")
    )
    title = raw_title.replace(TITLE_MAP).where(
        raw_title.isin(list(TITLE_MAP)), "Rare"
    )

    features = X.assign(Title=title)
    features = features.assign(Age=ages["Age"])
    features = features.assign(Fare=fares["Fare"])
    features = features.assign(
        Embarked=features["Embarked"].astype("string").fillna("Missing")
    )
    features = features.assign(
        CabinDeck=features["Cabin"].astype("string").str[0].fillna("Missing")
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )

    # Recompute the original frequency arithmetic on each available batch
    # instead of leaking the entire table into the outer training features.
    ticket_counts = features["Ticket"].value_counts()
    features = features.assign(
        TicketFreq=features["Ticket"].map(ticket_counts)
    )
    features = features.assign(
        GroupSize=features["FamilySize"].skb.apply_func(
            np.maximum, features["TicketFreq"]
        )
    )
    features = features.assign(
        FarePerPerson=features["Fare"] / features["GroupSize"]
    )
    features = features.assign(
        LogFarePerPerson=features["FarePerPerson"].skb.apply_func(np.log1p)
    )
    features = features.assign(IsChild=(features["Age"] < 12).astype(int))
    features = features.assign(
        IsMother=(
            (features["Sex"] == "female")
            & (features["Parch"] > 0)
            & (features["Title"] == "Mrs")
        ).astype(int)
    )
    features = features.assign(
        Sex_Pclass=features["Sex"].astype("string")
        + "_"
        + features["Pclass"].astype("string")
    )

    categorical = features[CAT_COLS].astype("string")
    cb_features = categorical.skb.concat([features[NUM_COLS]], axis=1)
    cb_features = cb_features[FEATURE_COLS]

    # 4. CatBoost. The reference early-stopped on its scored hold-out. Expose
    #    an inner eval split from the outer training rows instead, keeping all
    #    model parameters and patience. No scored rows are supplied to fit().
    #    Frame selections keep the split and eval_set visible in the graph
    #    without requiring the unavailable how="no_wrap" API.
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_training = cb_parts[~cb_parts["_inner_validation"]]
    cb_validation = cb_parts[cb_parts["_inner_validation"]]

    cb_probability = cb_training[FEATURE_COLS].skb.apply(
        CatBoostPositiveProbability(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_training["_inner_target"],
        fit_kwargs={
            "eval_set": (
                cb_validation[FEATURE_COLS],
                cb_validation["_inner_target"],
            ),
            "early_stopping_rounds": 50,
        },
    )

    # 5. Neural preprocessing stays separate from the torch training node.
    #    OrdinalEncoder requires an unknown sentinel outside its learned codes.
    #    Use -1, then record clipping it to zero to reproduce mapping.get(v, 0)
    #    exactly, without shifting any known category's LabelEncoder position.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).set_output(transform="pandas")
    )
    nn_categorical = nn_categorical.clip(lower=0)

    nn_numerical = (
        features[NUM_COLS]
        .astype(np.float32)
        .skb.apply(StandardScaler().set_output(transform="pandas"))
    )
    nn_features = nn_categorical.skb.concat([nn_numerical], axis=1)
    nn_features = nn_features[FEATURE_COLS]

    # Embedding dimensions are fit-only statistics exposed in the graph.
    cat_dimensions = nn_categorical.max().astype(int) + 1

    # The reference used the scored hold-out to schedule the learning rate.
    # Retain that scheduler but supply inner training-only validation rows.
    # As with CatBoost, removing this leakage can change the reported score.
    nn_parts = nn_features.skb.apply(GetXY(), y=y)
    nn_training = nn_parts[~nn_parts["_inner_validation"]]
    nn_validation = nn_parts[nn_parts["_inner_validation"]]

    nn_probability = nn_training[FEATURE_COLS].skb.apply(
        TorchPositiveProbability(random_state=42),
        y=nn_training["_inner_target"],
        fit_kwargs={
            "X_val": nn_validation[FEATURE_COLS],
            "y_val": nn_validation["_inner_target"],
            "cat_dims": cat_dimensions,
        },
    )

    # 6. Preserve the exact equal-weight probability average and >= threshold.
    #    These are components of one ensemble, not separately scored variants.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 7. Score; the outer splitter is declared only on mark_as_X.
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
