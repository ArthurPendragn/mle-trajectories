import numpy as np
import pandas as pd
import stratum as skrub  # Drop-in for skrub with the scheduler execution engine.
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GetXY(TransformerMixin, BaseEstimator):
    """Split the current outer fold's training rows for training feedback.

    This API version does not support apply(how="no_wrap"), so return a
    pandas frame. Recorded selections extract the four pieces, preserving
    train_test_split's shuffled row order.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        # This is an INTERNAL split, not the outer scored hold-out.
        train_idx, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        order = np.concatenate([train_idx, val_idx])
        is_fit = np.arange(len(X)) < len(train_idx)
        return X.iloc[order].assign(
            __is_fit=is_fit,
            __target=np.asarray(y)[order],
        )

    def transform(self, X):
        # Keep fit-only columns available at prediction time. All prediction
        # rows belong in the model input; the internal eval frame is empty.
        return X.assign(__is_fit=True, __target=np.nan)


class CatBoostPositiveProbability(CatBoostClassifier):
    """Expose the original positive-class probability as this branch's output."""

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
    """Original torch training loop; preprocessing remains outside this node."""

    def __init__(self, random_state=42, batch_size=32, num_epochs=150):
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def _inputs(self, X):
        # These are the embedding network's two inputs, not feature engineering.
        cat = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64, copy=True),
            dtype=torch.long,
            device=self.device_,
        )
        num = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=self.device_,
        )
        return cat, num

    def fit(self, X, y, *, eval_set, cat_dims):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.classes_ = np.unique(y)

        dims = np.asarray(cat_dims, dtype=np.int64).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in dims]
        self.model_ = TabularNeuralNet(
            dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        X_val, y_val = eval_set
        train_cat, train_num = self._inputs(X)
        val_cat, val_num = self._inputs(X_val)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

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
                labels = torch.argmax(outputs, dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, labels)
            scheduler.step(val_acc)

        # The original retains the final epoch, not a best-epoch checkpoint.
        return self

    def predict_proba(self, X):
        cat, num = self._inputs(X)
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(
                self.model_(cat, num), dim=1
            ).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def ensemble_predictions(cb_probs, nn_probs, mode):
    if mode == "fit":
        return None
    return (0.5 * cb_probs + 0.5 * nn_probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this file only records the CSV read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and preserve the exact stratified
    #    outer hold-out. Internal splits below are only for training feedback.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Fit median imputation on each outer
    #    training fold, fixing the original full-table imputation leakage.
    title = X["Name"].str.extract(
        r" ([A-Za-z]+)\.", expand=False
    ).fillna("Missing")
    features = X.assign(Title=title)
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    features = features.assign(
        CabinDeck=features["Cabin"].str[0].fillna("Missing")
    )
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )
    features = features[FEATURE_COLS]
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Age", "Fare"]
    )

    # StringDtype avoids object-dtype intermediates under the scheduler while
    # retaining the original categorical string values.
    categories = features[CAT_COLS].astype("string")
    cb_features = categories.skb.concat([features[NUM_COLS]], axis=1)
    cb_features = cb_features[FEATURE_COLS]

    # The original early-stopped on its scored hold-out. Keep the 600-iteration
    # limit and 50-round patience, but carve a stratified 20% evaluation set out
    # of the outer TRAINING rows. The original optimistic score is not exactly
    # reproducible without that leakage.
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_train = cb_parts[cb_parts["__is_fit"]]
    cb_val = cb_parts[~cb_parts["__is_fit"]]
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
        y=cb_train["__target"],
        fit_kwargs={
            "eval_set": (cb_val[FEATURE_COLS], cb_val["__target"]),
            "early_stopping_rounds": 50,
        },
    )

    # Sorted per-column ordinal codes reproduce LabelEncoder on strings.
    # OrdinalEncoder forbids an unknown sentinel that overlaps known codes:
    # encode unknowns as -1, then map them to 0 with a recorded clip.
    nn_categories = categories.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_categories = nn_categories.clip(lower=0)

    nn_numbers = features[NUM_COLS].astype(np.float32).skb.apply(
        StandardScaler()
    )
    nn_features = nn_categories.skb.concat([nn_numbers], axis=1)
    nn_features = nn_features[FEATURE_COLS]

    # This fit-only statistic remains a recorded node, computed on the outer
    # training fold, including its internal scheduler-validation categories.
    cat_dims = nn_categories.max() + 1

    # Move neural scheduler feedback off the scored rows to an internal
    # training-only split. Preserve all epochs and optimizer/scheduler settings.
    nn_parts = nn_features.skb.apply(GetXY(), y=y)
    nn_train = nn_parts[nn_parts["__is_fit"]]
    nn_val = nn_parts[~nn_parts["__is_fit"]]
    nn_probs = nn_train[FEATURE_COLS].skb.apply(
        NeuralPositiveProbability(
            random_state=42, batch_size=32, num_epochs=150
        ),
        y=nn_train["__target"],
        fit_kwargs={
            "eval_set": (nn_val[FEATURE_COLS], nn_val["__target"]),
            "cat_dims": cat_dims,
        },
    )

    # Preserve the original equal-weight averaging and inclusive threshold.
    pred = cb_probs.skb.apply_func(
        ensemble_predictions, nn_probs, skrub.eval_mode()
    )

    # 4. Score — one ensemble, one stratified hold-out, original accuracy metric.
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
