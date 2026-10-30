import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import stratum as skrub  # Drop-in .skb API with the scheduler execution engine.

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped_fill = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        self.fallback_ = X["Age"].fillna(grouped_fill).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped_fill = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X.assign(
            Age=X["Age"].fillna(grouped_fill).fillna(self.fallback_)
        )


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
    """Expose an inner validation split in a pandas-compatible container.

    The installed API lacks how="no_wrap", so recorded selections extract
    the four pieces from this frame instead of a dictionary.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        # This is an INNER monitoring split of the outer fold's training rows,
        # not the outer validation split (which lives on mark_as_X).
        train_indices, val_indices = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        target = np.asarray(y)
        train = X.iloc[train_indices].assign(
            __inner_target=target[train_indices],
            __inner_validation=False,
        )
        validation = X.iloc[val_indices].assign(
            __inner_target=target[val_indices],
            __inner_validation=True,
        )
        return pd.concat([train, validation], axis=0)

    def transform(self, X):
        # All scored rows enter the prediction feature selection. Fit-only
        # validation selections remain well-defined empty frames.
        return X.assign(__inner_target=0, __inner_validation=False)


class CatBoostPositiveProbability(CatBoostClassifier):
    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def predict(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)[:, 1]


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_c, e_dim)
                for num_c, e_dim in zip(cat_dims, emb_dims)
            ]
        )
        layers = []
        prev_dim = sum(emb_dims) + num_features
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.network(torch.cat(embs + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Only the torch training loop lives here, not feature preprocessing."""

    def __init__(self, random_state=42, batch_size=32, num_epochs=150):
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        dimensions = [int(dim) for dim in cat_dims]
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in dimensions]
        self.model_ = TabularNeuralNet(
            dimensions, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        t_X_cat_train = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        t_X_num_train = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        t_y_train = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        t_X_cat_val = torch.tensor(
            X_val[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        t_X_num_val = torch.tensor(
            X_val[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )

        # Keep all 150 epochs and the final weights, as in the original.
        # Validation accuracy changes the learning rate, not epoch count.
        for epoch in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(t_X_cat_train.size(0))
            for i in range(0, t_X_cat_train.size(0), self.batch_size):
                indices_batch = permutation[i : i + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(
                    t_X_cat_train[indices_batch],
                    t_X_num_train[indices_batch],
                )
                loss = criterion(outputs, t_y_train[indices_batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(t_X_cat_val, t_X_num_val)
                val_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)
            scheduler.step(val_acc)

        self.model_.eval()
        return self

    def predict(self, X):
        t_cat = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        t_num = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(t_cat, t_num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probs, nn_probs, mode):
    if mode == "fit":
        return None
    ensemble_probs = 0.5 * cb_probs + 0.5 * nn_probs
    return (ensemble_probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this file records the read without performing I/O.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target. The splitter reproduces the
    # original stratified train_test_split's single 80/20 hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Feature engineering — learned medians and ticket frequencies now use
    # outer training rows only, fixing the original full-table leakage.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = (
        raw_title.astype("string")
        .where(raw_title.isin(["Mr", "Miss", "Mrs", "Master"]), "Rare")
        .mask(raw_title.isin(["Mlle", "Ms"]), "Miss")
        .mask(raw_title == "Mme", "Mrs")
        .mask(raw_title.isna(), "Missing")
    )
    features = X.assign(Title=title)
    features = features.skb.apply(
        GroupAgeImputer(), cols=["Pclass", "Sex", "Age"]
    )
    features = features.assign(
        Sex_Pclass=(
            features["Sex"].astype("string")
            + "_"
            + features["Pclass"].astype("string")
        )
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    family_size = features["FamilySize"]
    family_tier = (
        family_size.astype("string")
        .where(family_size <= 4, "LargeFamily")
        .mask(family_size <= 4, "SmallFamily")
        .mask(family_size == 1, "IsAlone")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=features["Cabin"].notna().astype(int))
    features = features.skb.apply(TicketFrequency(), cols=["Ticket"])
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing").astype("string")
    )
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Fare"]
    )
    features = features[FEATURE_COLS]

    # StringDtype preserves string categories without object-valued columns.
    cats = features[CAT_COLS].astype("string")
    numeric = features[NUM_COLS]
    cb_features = cats.skb.concat([numeric], axis=1)[FEATURE_COLS]

    # 4. CatBoost — keep every parameter and early-stopping patience.
    # The original monitored the scored hold-out; instead carve an inner
    # stratified 20% eval set from outer training rows. This fixes leakage,
    # so the original leaky score cannot be reproduced exactly.
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_train = cb_parts[~cb_parts["__inner_validation"]]
    cb_validation = cb_parts[cb_parts["__inner_validation"]]
    cb_probs = cb_train[FEATURE_COLS].skb.apply(
        CatBoostPositiveProbability(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_train["__inner_target"],
        fit_kwargs={
            "eval_set": (
                cb_validation[FEATURE_COLS],
                cb_validation["__inner_target"],
            ),
            "early_stopping_rounds": 50,
        },
    )

    # 5. Neural preprocessing — sorted string vocabularies match LabelEncoder.
    # OrdinalEncoder forbids unknown_value=0 because zero encodes a known
    # category. Encode unknowns as -1, then replace only that sentinel with
    # zero through recorded clipping, exactly matching mapping.get(value, 0).
    nn_cats = cats.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_cats = nn_cats.clip(lower=0).astype(np.int64)
    nn_numeric = numeric.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_cats.skb.concat([nn_numeric], axis=1)[FEATURE_COLS]
    cat_dims = nn_cats.max().add(1).to_numpy()

    # The original NN scheduler also monitored scored rows. Expose an inner
    # validation split instead, retaining architecture, optimizer, epoch count
    # and scheduler. Encoding/scaling precede this inner split.
    nn_parts = nn_features.skb.apply(GetXY(), y=y)
    nn_train = nn_parts[~nn_parts["__inner_validation"]]
    nn_validation = nn_parts[nn_parts["__inner_validation"]]
    nn_probs = nn_train[FEATURE_COLS].skb.apply(
        NeuralPositiveProbability(
            random_state=42, batch_size=32, num_epochs=150
        ),
        y=nn_train["__inner_target"],
        fit_kwargs={
            "X_val": nn_validation[FEATURE_COLS],
            "y_val": nn_validation["__inner_target"],
            "cat_dims": cat_dims,
        },
    )

    # 6. Preserve the equal-weight probability blend and >= 0.5 threshold.
    # Only this ensemble was scored; there are no model-comparison choices.
    pred = cb_probs.skb.apply_func(
        ensemble_predictions, nn_probs, skrub.eval_mode()
    )

    # 7. Score — the outer splitter lives only on mark_as_X.
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
