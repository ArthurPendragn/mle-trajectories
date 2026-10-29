import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_c, e_dim) for num_c, e_dim in zip(cat_dims, emb_dims)]
        )
        in_dim = sum(emb_dims) + num_features

        layers = []
        prev_dim = in_dim
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


class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
    """Keep the original torch training loop inside a per-fold estimator."""

    def __init__(
        self,
        hidden_dims=(64, 32),
        batch_size=32,
        num_epochs=150,
        learning_rate=1e-2,
        weight_decay=1e-4,
        validation_size=0.2,
        random_state=42,
    ):
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_size = validation_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Dimensions are determined from this fold's encoded training frame,
        # not from the full dataset or the outer scored rows.
        cat_dims = (X[CAT_COLS].max().to_numpy(dtype=np.int64) + 1).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

        # The original used the scored hold-out to update its LR scheduler.
        # Keep that scheduler, but use an inner split of the fold's training
        # rows. This removes validation leakage and can change the score.
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.model_ = TabularNeuralNet(
            cat_dims,
            emb_dims,
            num_features=len(NUM_COLS),
            hidden_dims=self.hidden_dims,
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

        t_cat_train = torch.tensor(
            X_train[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        t_num_train = torch.tensor(
            X_train[NUM_COLS].to_numpy(),
            dtype=torch.float32,
            device=self.device_,
        )
        t_y_train = torch.tensor(
            np.asarray(y_train), dtype=torch.long, device=self.device_
        )
        t_cat_val = torch.tensor(
            X_val[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        t_num_val = torch.tensor(
            X_val[NUM_COLS].to_numpy(),
            dtype=torch.float32,
            device=self.device_,
        )

        for epoch in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(t_cat_train.size(0))

            for i in range(0, t_cat_train.size(0), self.batch_size):
                batch_indices = permutation[i : i + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(
                    t_cat_train[batch_indices], t_num_train[batch_indices]
                )
                loss = criterion(outputs, t_y_train[batch_indices])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(t_cat_val, t_num_val)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

            scheduler.step(accuracy_score(y_val, val_preds))

        # As in the original, use the final epoch, not a best-epoch checkpoint.
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
            return torch.argmax(outputs, dim=1).cpu().numpy()


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this module records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target. StratifiedShuffleSplit reproduces
    #    the original stratified train_test_split's exact outer hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, followed by separate learned steps.
    features = X.assign(
        Title=X["Name"].str.extract(
            r" ([A-Za-z]+)\.", expand=False
        ).fillna("Unknown"),
        FamilySize=X["SibSp"] + X["Parch"] + 1,
        Embarked=X["Embarked"].fillna("Missing"),
    )

    # Sorted ordinal codes match the original per-column LabelEncoder codes.
    # Unlike the original full-table encoding, vocabularies are learned only
    # on the fold's training rows. Previously unseen validation categories use
    # code zero (the first known category), keeping embedding indices valid.
    categorical = (
        features[CAT_COLS]
        .astype(str)
        .skb.apply(
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        )
        .clip(lower=0)
        .astype(np.int64)
    )

    # Fix the original full-table median/scaling leakage by fitting each
    # estimator inside CV. Preserve median fill before the float32 conversion.
    numeric = features[NUM_COLS].skb.apply(
        SimpleImputer(strategy="median"),
        cols=["Age", "Fare"],
    )
    numeric = numeric.astype(np.float32).skb.apply(StandardScaler())

    design = categorical.skb.concat([numeric], axis=1)[FEATURE_COLS]
    pred = design.skb.apply(
        TorchTabularClassifier(
            hidden_dims=(64, 32),
            batch_size=32,
            num_epochs=150,
            learning_rate=1e-2,
            weight_decay=1e-4,
            validation_size=0.2,
            random_state=42,
        ),
        y=y,
    )

    # 4. Score — the marked splitter supplies the single outer hold-out.
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
