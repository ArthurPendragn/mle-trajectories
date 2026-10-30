import numpy as np
import pandas as pd
import stratum as skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = [
    "Pclass", "Sex", "Embarked", "Title", "CabinDeck",
    "TicketPrefix", "Sex_Pclass", "Child_Pclass",
]
NUM_COLS = [
    "Age", "SibSp", "Parch", "Fare", "FamilySize", "GroupSize",
    "FarePerPerson", "LogFare", "LogFarePerPerson", "IsChild",
]
FEATURE_COLS = CAT_COLS + NUM_COLS

TITLE_MAPPING = {
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Dona": "Royalty",
    "Sir": "Royalty",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Dr": "Officer",
    "Rev": "Officer",
}
DECK_MAPPING = {
    "A": "ABC", "B": "ABC", "C": "ABC",
    "D": "DE", "E": "DE", "F": "FG", "G": "FG",
    "T": "ABC", "Missing": "Missing",
}


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Learn one conditional imputation and its post-imputation fallback."""

    def __init__(self, column, groups):
        self.column = column
        self.groups = groups

    def _conditional_fill(self, X):
        keys = pd.MultiIndex.from_frame(X[list(self.groups)])
        medians = self.group_medians_.reindex(keys).to_numpy()
        return X[self.column].fillna(pd.Series(medians, index=X.index))

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(list(self.groups))[self.column].median()
        if not isinstance(self.group_medians_.index, pd.MultiIndex):
            self.group_medians_.index = pd.MultiIndex.from_arrays(
                [self.group_medians_.index], names=list(self.groups)
            )
        self.fallback_ = self._conditional_fill(X).median()
        return self

    def transform(self, X):
        return X.assign(
            **{self.column: self._conditional_fill(X).fillna(self.fallback_)}
        )


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        # FamilySize remains the lower bound for previously unseen tickets.
        return X.assign(TicketCount=X["Ticket"].map(self.counts_).fillna(0))


class GetXY(TransformerMixin, BaseEstimator):
    """Expose a fold-local early-stopping split as a pandas container.

    This installed API has no apply(how="no_wrap"). Return a frame rather than
    a dict, and expose its four pieces with recorded selections downstream.
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
        target = np.asarray(y)
        training = X.iloc[train_idx].assign(
            _early_stop_eval=False, _early_stop_target=target[train_idx]
        )
        validation = X.iloc[val_idx].assign(
            _early_stop_eval=True, _early_stop_target=target[val_idx]
        )
        # Preserve each split's train_test_split ordering.
        return pd.concat([training, validation], axis=0)

    def transform(self, X):
        # All prediction rows flow through the training-feature selection.
        # Keep the same schema even when fit-only branches are evaluated.
        return X.assign(_early_stop_eval=False, _early_stop_target=np.nan)


class CatBoostPositiveProbability(CatBoostClassifier):
    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def predict(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)[:, 1]


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, embedding_dim)
                for num_categories, embedding_dim in zip(cat_dims, emb_dims)
            ]
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
        # The original explicitly uses two outputs for the binary target.
        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """The original torch optimization loop, without feature preprocessing."""

    def __init__(
        self,
        batch_size=32,
        num_epochs=150,
        learning_rate=1e-2,
        weight_decay=1e-4,
        random_state=42,
        validation_fraction=0.2,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.validation_fraction = validation_fraction

    def _tensors(self, X):
        categorical = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64, copy=True),
            dtype=torch.long,
            device=self.device_,
        )
        numeric = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=self.device_,
        )
        return categorical, numeric

    def fit(self, X, y, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # This is an INTERNAL neural-training validation split, not outer CV.
        # The original scheduled learning rates using the scored hold-out.
        # Keep that schedule, but use only this fold's training rows.
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y,
        )
        dimensions = np.asarray(cat_dims, dtype=int).tolist()
        embedding_dimensions = [
            max(2, int(round(dimension**0.5 * 2)))
            for dimension in dimensions
        ]
        self.model_ = TabularNeuralNet(
            dimensions, embedding_dimensions, num_features=len(NUM_COLS)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        train_cat, train_num = self._tensors(X_train)
        val_cat, val_num = self._tensors(X_val)
        train_y = torch.tensor(
            np.asarray(y_train), dtype=torch.long, device=self.device_
        )

        # Keep all 150 epochs and use the final network, with no checkpoint
        # selection or additional early stopping.
        for epoch in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for start in range(0, train_cat.size(0), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[batch], train_num[batch])
                loss = criterion(outputs, train_y[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(val_cat, val_num)
                predictions = outputs.argmax(dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_val, predictions))
        return self

    def predict(self, X):
        categorical, numeric = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(categorical, numeric)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    return (
        0.5 * np.asarray(cb_probability) + 0.5 * np.asarray(nn_probability)
        >= 0.5
    ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records the read without executing it.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — mark the raw target and reproduce the stratified
    #    80/20 outer hold-out, including its random seed.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Learn medians and ticket frequencies
    #    from training rows only, fixing the original full-table leakage.
    title = (
        X["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .fillna("Missing")
        .replace(TITLE_MAPPING)
        .astype("string")
    )
    features = X.assign(Title=title)
    features = features.skb.apply(
        GroupMedianImputer(column="Age", groups=("Pclass", "Title"))
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )

    # Extract fields without producing object-dtype lists.
    ticket_text = (
        features["Ticket"]
        .astype("string")
        .fillna("nan")
        .str.replace(".", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.strip()
    )
    first_field = ticket_text.str.extract(r"^(\S+)", expand=False)
    prefix = first_field.str.upper().where(
        ticket_text.str.contains(r"\s+", regex=True)
        | ~first_field.str.isdigit(),
        "NUMERIC",
    )
    features = features.assign(TicketPrefix=prefix)
    features = features.skb.apply(TicketFrequency())
    features = features.assign(
        GroupSize=features["FamilySize"].skb.apply_func(
            np.maximum, features["TicketCount"]
        )
    )
    features = features.skb.apply(
        GroupMedianImputer(column="Fare", groups=("Pclass",))
    )
    features = features.assign(
        FarePerPerson=features["Fare"] / features["GroupSize"]
    )
    features = features.assign(
        LogFare=features["Fare"].skb.apply_func(np.log1p)
    )
    features = features.assign(
        LogFarePerPerson=features["FarePerPerson"].skb.apply_func(np.log1p)
    )
    raw_deck = features["Cabin"].str[0].fillna("Missing")
    features = features.assign(
        CabinDeck=raw_deck.map(DECK_MAPPING).fillna("Missing").astype("string")
    )
    features = features.assign(
        Sex_Pclass=(
            features["Sex"].astype("string").fillna("nan")
            + "_"
            + features["Pclass"].astype("string").fillna("nan")
        )
    )
    features = features.assign(IsChild=(features["Age"] < 12).astype(int))
    features = features.assign(
        Child_Pclass=(
            features["IsChild"].astype("string")
            + "_"
            + features["Pclass"].astype("string").fillna("nan")
        )
    )
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )
    features = features[FEATURE_COLS]

    categorical_strings = features[CAT_COLS].astype("string").fillna("nan")
    cb_features = categorical_strings.skb.concat(
        [features[NUM_COLS]], axis=1
    )[FEATURE_COLS]

    # 4. CatBoost — retain every parameter and the 50-round patience.
    #    The original early-stopped on the scored rows. Instead carve an eval
    #    set from this fold's training rows; its optimistic score cannot be
    #    reproduced without leakage.
    #    A frame-returning GetXY supports the installed API, which lacks how=.
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_train = cb_parts[~cb_parts["_early_stop_eval"]]
    cb_validation = cb_parts[cb_parts["_early_stop_eval"]]
    X_cb_fit = cb_train[FEATURE_COLS]
    y_cb_fit = cb_train["_early_stop_target"]
    X_cb_val = cb_validation[FEATURE_COLS]
    y_cb_val = cb_validation["_early_stop_target"]

    cb_probability = X_cb_fit.skb.apply(
        CatBoostPositiveProbability(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=y_cb_fit,
        fit_kwargs={
            "eval_set": (X_cb_val, y_cb_val),
            "early_stopping_rounds": 50,
        },
    )

    # 5. Neural preprocessing — separate fitted encoding and scaling nodes.
    #    Sorted ordinal categories match LabelEncoder, with unseen values
    #    redirected to index zero.
    nn_categorical = categorical_strings.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_categorical = nn_categorical.replace(-1, 0).astype(np.int64)
    nn_numeric = features[NUM_COLS].astype(np.float32).skb.apply(
        StandardScaler()
    )
    nn_features = nn_categorical.skb.concat(
        [nn_numeric], axis=1
    )[FEATURE_COLS]

    # Vocabulary sizes are fit-only statistics evaluated on training rows.
    cat_dimensions = nn_categorical.max() + 1
    nn_probability = nn_features.skb.apply(
        NeuralPositiveProbability(),
        y=y,
        fit_kwargs={"cat_dims": cat_dimensions},
    )

    # 6. Preserve the sole scored variant and its exact prediction arithmetic.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 7. Score — outer validation is owned exclusively by mark_as_X.
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
