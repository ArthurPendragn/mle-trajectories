import numpy as np
import pandas as pd
import stratum as skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]
NUM_COLS = [
    "Age", "SibSp", "Parch", "Fare", "FamilySize", "TicketCount",
    "FarePerPerson", "LogFare", "LogFarePerPerson", "CabinCount",
    "Age_Pclass", "IsChild",
]
FEATURE_COLS = CAT_COLS + NUM_COLS

TITLE_MAPPING = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Royalty",
    "Countess": "Royalty", "Capt": "Officer", "Col": "Officer",
    "Don": "Royalty", "Dr": "Officer", "Major": "Officer",
    "Rev": "Officer", "Sir": "Royalty", "Jonkheer": "Royalty",
    "Dona": "Royalty",
}


class TicketFrequency(TransformerMixin, BaseEstimator):
    """Learn the ticket-frequency lookup from this fold's training rows."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"TicketCount": X["Ticket"].map(self.counts_).fillna(1).astype(int)},
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Expose an internal split through a pandas-compatible transport table.

    This installed API has no how="no_wrap". Instead of returning a dict,
    return the unchanged rows with split membership, ordering, and target
    metadata. Recorded operations below extract the four split pieces.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        indices = np.arange(len(X))
        train, val = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        # Reindexing restores original row positions; the explicit ordering
        # metadata subsequently restores train_test_split's shuffled row order.
        membership = pd.Series(
            np.concatenate([
                np.zeros(len(train), dtype=np.int8),
                np.ones(len(val), dtype=np.int8),
            ]),
            index=np.concatenate([train, val]),
        ).reindex(indices).to_numpy()
        order = pd.Series(
            np.concatenate([np.arange(len(train)), np.arange(len(val))]),
            index=np.concatenate([train, val]),
        ).reindex(indices).to_numpy()
        return X.assign(
            _inner_part=membership,
            _inner_order=order,
            _inner_y=np.asarray(y),
        )

    def transform(self, X):
        # Every scored row follows the prediction path, with no eval rows.
        return X.assign(
            _inner_part=np.int8(0),
            _inner_order=np.arange(len(X)),
            _inner_y=np.int64(0),
        )


class CatBoostPositiveProbability(CatBoostClassifier):
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
            layers.extend([
                nn.Linear(previous, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            previous = width
        layers.append(nn.Linear(previous, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """The torch model and training loop; preprocessing is external."""

    def fit(self, X, y, eval_set=None, cat_dims=None):
        np.random.seed(42)
        torch.manual_seed(42)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dimensions = [int(d) for d in cat_dims]
        embedding_dimensions = [
            max(2, int(round(d ** 0.5 * 2))) for d in dimensions
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

        X_val, y_val = eval_set
        train_cat = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long, device=self.device_,
        )
        train_num = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32, device=self.device_,
        )
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_cat = torch.tensor(
            X_val[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long, device=self.device_,
        )
        val_num = torch.tensor(
            X_val[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32, device=self.device_,
        )

        for epoch in range(150):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for start in range(0, train_cat.size(0), 32):
                batch = permutation[start:start + 32]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[batch], train_num[batch])
                loss = criterion(outputs, train_y[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(val_cat, val_num)
                labels = torch.argmax(outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_val, labels))

        # The original uses the last epoch, not a best-epoch checkpoint.
        self.model_.eval()
        return self

    def predict(self, X):
        cat = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long, device=self.device_,
        )
        num = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32, device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(
                self.model_(cat, num), dim=1
            )[:, 1].cpu().numpy()


def ensemble_predictions(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    probability = 0.5 * cb_probability + 0.5 * nn_probability
    return (probability >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records the read without accessing the file.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — raw target and the exact stratified 80/20 holdout.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = raw_title.map(TITLE_MAPPING).fillna("Other").astype("string")

    # Fix full-table median and frequency leakage by learning each statistic
    # from the outer fold's training rows only.
    imputed = X[["Age", "Fare"]].skb.apply(SimpleImputer(strategy="median"))
    ticket_count = X[["Ticket"]].skb.apply(TicketFrequency())

    deck = X["Cabin"].str[0].fillna("Missing")
    deck = deck.where(
        deck.isin(["A", "B", "C", "D", "E", "F", "G", "T"]), "Missing"
    ).astype("string")
    # Count whitespace-delimited tokens without producing Python-list columns.
    cabin_count = X["Cabin"].fillna("").str.count(r"\S+")
    family_size = X["SibSp"] + X["Parch"] + 1
    group_size = family_size.skb.apply_func(
        np.maximum, ticket_count["TicketCount"]
    )
    fare_per_person = imputed["Fare"] / group_size

    features = X.assign(
        Title=title,
        Age=imputed["Age"],
        Fare=imputed["Fare"],
        Embarked=X["Embarked"].fillna("Missing"),
        CabinDeck=deck,
        CabinCount=cabin_count,
        FamilySize=family_size,
        TicketCount=ticket_count["TicketCount"],
        FarePerPerson=fare_per_person,
        LogFare=imputed["Fare"].skb.apply_func(np.log1p),
        LogFarePerPerson=fare_per_person.skb.apply_func(np.log1p),
        Age_Pclass=imputed["Age"] * X["Pclass"],
        IsChild=(imputed["Age"] < 12).astype(int),
    )[FEATURE_COLS]

    categories = features[CAT_COLS].astype("string").fillna("nan")
    cb_features = categories.skb.concat([features[NUM_COLS]], axis=1)

    # The original early-stopped and scheduled learning rates on the scored
    # holdout. Carve internal eval sets from outer TRAINING rows instead.
    # This removes that leakage and cannot reproduce the original leaky score.
    # GetXY uses pandas transport metadata because this API does not support
    # how="no_wrap"; all four pieces remain visible as recorded operations.
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_train = cb_parts[cb_parts["_inner_part"] == 0].sort_values("_inner_order")
    cb_val = cb_parts[cb_parts["_inner_part"] == 1].sort_values("_inner_order")

    cb_probability = cb_train[FEATURE_COLS].skb.apply(
        CatBoostPositiveProbability(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_train["_inner_y"],
        fit_kwargs={
            "eval_set": (cb_val[FEATURE_COLS], cb_val["_inner_y"]),
            "early_stopping_rounds": 50,
        },
    )

    # Sorted vocabularies match LabelEncoder; unseen categories map to zero.
    encoded_categories = categories.skb.apply(
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    ).replace(-1, 0).astype(np.int64)
    scaled_numbers = (
        features[NUM_COLS].astype(np.float32).skb.apply(StandardScaler())
    )
    nn_features = encoded_categories.skb.concat([scaled_numbers], axis=1)
    cat_dims = (encoded_categories.max() + 1).astype(int).tolist()

    nn_parts = nn_features.skb.apply(GetXY(), y=y)
    nn_train = nn_parts[nn_parts["_inner_part"] == 0].sort_values("_inner_order")
    nn_val = nn_parts[nn_parts["_inner_part"] == 1].sort_values("_inner_order")

    nn_probability = nn_train[FEATURE_COLS].skb.apply(
        NeuralPositiveProbability(),
        y=nn_train["_inner_y"],
        fit_kwargs={
            "eval_set": (nn_val[FEATURE_COLS], nn_val["_inner_y"]),
            "cat_dims": cat_dims,
        },
    )

    # Preserve the original 50/50 average and >= 0.5 threshold.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 4. Score — mark_as_X owns the sole outer holdout.
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
