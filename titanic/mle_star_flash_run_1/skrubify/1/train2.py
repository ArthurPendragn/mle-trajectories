import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        group_values = pd.Series(
            X.set_index(["Pclass", "Sex"]).index.map(self.group_medians_),
            index=X.index,
        )
        self.fallback_ = X["Age"].fillna(group_values).median()
        return self

    def transform(self, X):
        group_values = pd.Series(
            X.set_index(["Pclass", "Sex"]).index.map(self.group_medians_),
            index=X.index,
        )
        return X.assign(
            Age=X["Age"].fillna(group_values).fillna(self.fallback_)
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
    """Expose an inner early-stopping split through a pandas container.

    This skrub version has no apply(how="no_wrap"), so the split is returned
    as a frame with partition metadata instead of a dict. Recorded row
    selections below expose the same four pieces to the predictor.
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
        # Rank metadata preserves train_test_split's shuffled row order.
        train_order = pd.Series(
            np.arange(len(train_idx)), index=X.index[train_idx]
        ).reindex(X.index)
        val_order = pd.Series(
            np.arange(len(val_idx)), index=X.index[val_idx]
        ).reindex(X.index)
        return X.assign(
            _inner_train_order=train_order,
            _inner_val_order=val_order,
            _inner_target=np.asarray(y),
        )

    def transform(self, X):
        # Prediction uses all incoming rows in their original order. Keep the
        # eval columns present even if an executor visits fit-only nodes.
        return X.assign(
            _inner_train_order=np.arange(len(X)),
            _inner_val_order=np.nan,
            _inner_target=np.nan,
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
            [
                nn.Embedding(num_c, emb_dim)
                for num_c, emb_dim in zip(cat_dims, emb_dims)
            ]
        )
        previous_dim = sum(emb_dims) + num_features
        layers = []
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
        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Original torch loop, with a legitimate per-fit scheduler hold-out."""

    def __init__(
        self,
        batch_size=32,
        num_epochs=150,
        learning_rate=1e-2,
        weight_decay=1e-4,
        validation_fraction=0.2,
        random_state=42,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Encoding and scaling are independent recorded nodes.
        x_cat = X[CAT_COLS].to_numpy(dtype=np.int64)
        x_num = X[NUM_COLS].to_numpy(dtype=np.float32)
        cat_dims = (x_cat.max(axis=0) + 1).tolist()
        emb_dims = [
            max(2, int(round(dim**0.5 * 2))) for dim in cat_dims
        ]
        self.model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        # This is an INNER split for the torch training loop, not outer CV.
        # The original scheduler monitored the scored outer validation rows;
        # retain its mechanism using only the current fold's training rows.
        train_idx, val_idx = train_test_split(
            np.arange(len(y)),
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y,
        )
        labels = np.asarray(y)
        t_cat_train = torch.tensor(
            x_cat[train_idx], dtype=torch.long, device=self.device_
        )
        t_num_train = torch.tensor(
            x_num[train_idx], dtype=torch.float32, device=self.device_
        )
        t_y_train = torch.tensor(
            labels[train_idx], dtype=torch.long, device=self.device_
        )
        t_cat_val = torch.tensor(
            x_cat[val_idx], dtype=torch.long, device=self.device_
        )
        t_num_val = torch.tensor(
            x_num[val_idx], dtype=torch.float32, device=self.device_
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

        # Preserve all 150 epochs and final-epoch probabilities.
        for epoch in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(t_cat_train.size(0))
            for start in range(0, t_cat_train.size(0), self.batch_size):
                batch_idx = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(
                    t_cat_train[batch_idx], t_num_train[batch_idx]
                )
                loss = criterion(outputs, t_y_train[batch_idx])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(t_cat_val, t_num_val)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(labels[val_idx], val_preds))

        return self

    def predict_proba(self, X):
        x_cat = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        x_num = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(
                self.model_(x_cat, x_num), dim=1
            ).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def ensemble_predictions(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    return (0.5 * cb_probability + 0.5 * nn_probability >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records the read without accessing the file.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target. This splitter reproduces the
    #    original stratified train_test_split's single 80/20 hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and models. Whole-table age, fare and ticket
    #    statistics originally leaked; learn each independently per fold.
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
    ).fillna("Rare").where(raw_title.notna(), "Missing")

    age = X[["Pclass", "Sex", "Age"]].skb.apply(GroupedAgeImputer())
    ticket_freq = X[["Ticket"]].skb.apply(TicketFrequency())
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))
    family_size = X["SibSp"] + X["Parch"] + 1
    family_tier = (
        skrub.as_data_op("LargeFamily")
        .skb.apply_func(pd.Series, index=X.index)
        .where(~(family_size <= 4), "SmallFamily")
        .where(~(family_size == 1), "IsAlone")
    )

    features = X.assign(
        Title=title,
        Age=age["Age"],
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str),
        FamilySize=family_size,
        FamilyTier=family_tier,
        HasCabin=X["Cabin"].notna().astype(int),
        TicketFreq=ticket_freq["TicketFreq"],
        Embarked=X["Embarked"].fillna("Missing"),
        Fare=fare["Fare"],
    )[FEATURE_COLS]

    categorical = features[CAT_COLS].astype(str)
    cb_features = categorical.skb.concat([features[NUM_COLS]], axis=1)

    # Keep early stopping visible in the plan. The installed skrub requires
    # pandas transformer outputs, so GetXY exposes partition metadata and
    # recorded selections unpack it rather than using how="no_wrap".
    # The original used the scored hold-out as eval_set; instead carve a
    # stratified 20% eval set from outer training, keeping 600 rounds/patience 50.
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_train = cb_parts[
        cb_parts["_inner_train_order"].notna()
    ].sort_values("_inner_train_order")
    cb_eval = cb_parts[
        cb_parts["_inner_val_order"].notna()
    ].sort_values("_inner_val_order")

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
        y=cb_train["_inner_target"],
        fit_kwargs={
            "eval_set": (cb_eval[FEATURE_COLS], cb_eval["_inner_target"]),
            "early_stopping_rounds": 50,
        },
    )

    # Sorted string vocabularies match LabelEncoder. Unseen categories map
    # to zero, exactly as in the original validation preparation.
    encoded_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).replace(-1, 0)
    scaled_numeric = features[NUM_COLS].astype(np.float32).skb.apply(
        StandardScaler()
    )
    nn_features = encoded_categorical.skb.concat(
        [scaled_numeric], axis=1
    )[FEATURE_COLS]
    nn_probability = nn_features.skb.apply(
        NeuralPositiveProbability(), y=y
    )

    # Only the ensemble was scored. Preserve probability-column averaging and
    # the inclusive threshold, with prediction arithmetic gated in fit mode.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 4. Score — outer validation is driven solely by mark_as_X.
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
