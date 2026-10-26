import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
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


class FareMedianImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.median_ = X["Fare"].median()
        return self

    def transform(self, X):
        return X.assign(Fare=X["Fare"].fillna(self.median_))


class GetXY(TransformerMixin, BaseEstimator):
    """Represent an inner split as a pandas table for this skrub version."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        # Preserve train_test_split's row order within each partition.
        return pd.concat(
            [
                X_fit.assign(
                    _inner_fit=True, _inner_target=np.asarray(y_fit)
                ),
                X_val.assign(
                    _inner_fit=False, _inner_target=np.asarray(y_val)
                ),
            ],
            axis=0,
        )

    def transform(self, X):
        # At prediction time all input rows go to the predictor.
        return X.assign(_inner_fit=True, _inner_target=np.nan)


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
        embs = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embs + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Original torch loop; preprocessing and eval splitting stay in the plan."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y, *, eval_set, cat_dims):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        dims = [int(dim) for dim in np.asarray(cat_dims)]
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in dims]
        self.model_ = TabularNeuralNet(
            dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        num_epochs = 150
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-5
        )

        X_val, y_val = eval_set
        train_cat = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        train_num = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_cat = torch.tensor(
            X_val[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        val_num = torch.tensor(
            X_val[NUM_COLS].to_numpy(),
            dtype=torch.float32,
            device=self.device_,
        )
        val_y = torch.tensor(
            np.asarray(y_val), dtype=torch.long, device=self.device_
        )

        best_val_loss = float("inf")
        best_model_state = None
        batch_size = 32

        for epoch in range(num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for i in range(0, train_cat.size(0), batch_size):
                indices_batch = permutation[i : i + batch_size]
                optimizer.zero_grad()
                outputs = self.model_(
                    train_cat[indices_batch], train_num[indices_batch]
                )
                loss = criterion(outputs, train_y[indices_batch])
                loss.backward()
                optimizer.step()

            scheduler.step()
            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(val_cat, val_num)
                val_loss = criterion(val_outputs, val_y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu().clone()
                    for k, v in self.model_.state_dict().items()
                }

        if best_model_state is not None:
            self.model_.load_state_dict(
                {
                    k: v.to(self.device_)
                    for k, v in best_model_state.items()
                }
            )
        self.model_.eval()
        return self

    def predict(self, X):
        x_cat = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        x_num = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(x_cat, x_num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probs, nn_probs, mode):
    if mode == "fit":
        return None
    return (0.5 * cb_probs + 0.5 * nn_probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing only records the CSV read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and design matrix. The splitter reproduces the
    #    original stratified 80/20 train_test_split exactly.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. Learned medians and
    #    ticket counts use only outer training rows, fixing full-table leakage.
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

    age = X[["Pclass", "Sex", "Age"]].skb.apply(GroupedAgeImputer())
    ticket = X[["Ticket"]].skb.apply(TicketFrequency())
    fare = X[["Fare"]].skb.apply(FareMedianImputer())

    features = X.assign(Title=title)
    features = features.assign(Age=age["Age"])
    features = features.assign(
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
    )
    features = features.assign(FamilySize=X["SibSp"] + X["Parch"] + 1)
    family_size = features["FamilySize"]
    family_tier = (
        family_size.le(4)
        .map({True: "SmallFamily", False: "LargeFamily"})
        .where(family_size.ne(1), "IsAlone")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=X["Cabin"].notna().astype(int))
    features = features.assign(TicketFreq=ticket["TicketFreq"])
    features = features.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.assign(Fare=fare["Fare"])
    features = features[FEATURE_COLS]

    cat_strings = features[CAT_COLS].astype(str)
    cb_features = cat_strings.skb.concat([features[NUM_COLS]], axis=1)

    # Sorted string categories match the original per-column LabelEncoders;
    # unseen validation categories retain the original fallback index zero.
    nn_cat = cat_strings.skb.apply(
        OrdinalEncoder(
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    )
    nn_cat = nn_cat.replace(-1, 0)
    nn_num = features[NUM_COLS].astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_cat.skb.concat([nn_num], axis=1)

    # This fit-only statistic is evaluated from the fold's training rows.
    cat_dims = nn_cat.max(axis=0) + 1

    # The original selected both models using the scored hold-out. Preserve
    # CatBoost's patience and neural checkpoint selection, but use matching
    # inner stratified eval splits of the outer training rows instead.
    # The original leaky validation score is therefore not reproducible.
    #
    # This skrub version has no apply(how="no_wrap"). GetXY consequently
    # returns a pandas table, not a dict; ordinary recorded selections expose
    # all four split pieces without hiding the split inside either predictor.
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_fit = cb_parts[cb_parts["_inner_fit"]]
    cb_val = cb_parts[~cb_parts["_inner_fit"]]

    nn_parts = nn_features.skb.apply(GetXY(), y=y)
    nn_fit = nn_parts[nn_parts["_inner_fit"]]
    nn_val = nn_parts[~nn_parts["_inner_fit"]]

    cb_probs = cb_fit[FEATURE_COLS].skb.apply(
        CatBoostPositiveProbability(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_fit["_inner_target"],
        fit_kwargs={
            "eval_set": (
                cb_val[FEATURE_COLS],
                cb_val["_inner_target"],
            ),
            "early_stopping_rounds": 50,
        },
    )
    nn_probs = nn_fit[FEATURE_COLS].skb.apply(
        NeuralPositiveProbability(random_state=42),
        y=nn_fit["_inner_target"],
        fit_kwargs={
            "eval_set": (
                nn_val[FEATURE_COLS],
                nn_val["_inner_target"],
            ),
            "cat_dims": cat_dims,
        },
    )

    # Preserve the single ensemble's 50/50 average and >= 0.5 threshold.
    # Prediction arithmetic is gated because fit mode returns estimators.
    pred = cb_probs.skb.apply_func(
        ensemble_predictions, nn_probs, skrub.eval_mode()
    )

    # 4. Score — only the splitter on mark_as_X drives validation.
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
