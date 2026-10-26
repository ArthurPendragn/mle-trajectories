import copy

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
NUM_COLS = [
    "Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"
]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn group medians and the post-group-imputation fallback median."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        group_values = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        self.fallback_ = X["Age"].fillna(group_values).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        group_values = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X["Age"].fillna(group_values).fillna(self.fallback_).to_frame("Age")


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class GetXY(TransformerMixin, BaseEstimator):
    """Expose an internal split as a pandas table for this skrub version.

    This version of skb.apply does not support how="no_wrap". Returning a
    same-length pandas table, rather than a dict, avoids that unsupported API.
    Ordinary recorded selections downstream extract the four split pieces.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        train_indices, _ = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        is_fit = np.isin(np.arange(len(X)), train_indices)
        return X.assign(
            _internal_fit=is_fit,
            _internal_y=np.asarray(y),
            _internal_order=np.arange(len(X)),
        )

    def transform(self, X):
        return X.assign(
            _internal_fit=True,
            _internal_y=np.nan,
            _internal_order=np.arange(len(X)),
        )


class CatBoostProbabilities(CatBoostClassifier):
    """Expose probabilities through predict for the recorded blend."""

    def __sklearn_clone__(self):
        return CatBoostProbabilities(**self.get_params(deep=False))

    def predict(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n, d in zip(cat_dims, emb_dims)]
        )
        layers = []
        prev_dim = sum(emb_dims) + num_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class CheckpointNeuralEnsemble(ClassifierMixin, BaseEstimator):
    """The original torch loop and checkpoint ensemble, rerun per outer fold."""

    def __init__(
        self,
        seeds=(42, 1337, 2024, 777, 999),
        num_epochs=150,
        batch_size=32,
        top_k_checkpoints=3,
    ):
        self.seeds = seeds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.top_k_checkpoints = top_k_checkpoints

    def fit(self, X, y, X_val=None, y_val=None):
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Categories were learned by a separate encoder on outer-training rows.
        category_values = np.concatenate(
            [X[CAT_COLS].to_numpy(), X_val[CAT_COLS].to_numpy()], axis=0
        )
        cat_dims = (category_values.max(axis=0) + 1).astype(int).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

        t_cat = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        t_num = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        t_y = torch.tensor(np.asarray(y), dtype=torch.long, device=self.device_)
        v_cat = torch.tensor(
            X_val[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        v_num = torch.tensor(
            X_val[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        v_y = torch.tensor(
            np.asarray(y_val), dtype=torch.long, device=self.device_
        )

        self.models_ = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = TabularNeuralNet(
                cat_dims, emb_dims, num_features=len(NUM_COLS)
            ).to(self.device_)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-2, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10
            )
            checkpoints = []

            for _ in range(self.num_epochs):
                model.train()
                permutation = torch.randperm(t_cat.size(0))
                for start in range(0, t_cat.size(0), self.batch_size):
                    batch_idx = permutation[start : start + self.batch_size]
                    batch_cat = t_cat[batch_idx]
                    batch_num = t_num[batch_idx]
                    batch_y = t_y[batch_idx]

                    if batch_num.size(0) > 1 and np.random.rand() < 0.5:
                        noisy = batch_num + torch.randn_like(batch_num) * 0.015
                        lam = np.random.beta(0.4, 0.4)
                        perm_mix = torch.randperm(batch_num.size(0))
                        mixed = lam * noisy + (1.0 - lam) * noisy[perm_mix]
                        optimizer.zero_grad()
                        outputs = model(batch_cat, mixed)
                        loss = (
                            lam * criterion(outputs, batch_y)
                            + (1.0 - lam)
                            * criterion(outputs, batch_y[perm_mix])
                        )
                    else:
                        noise = torch.randn_like(batch_num) * 0.01
                        optimizer.zero_grad()
                        outputs = model(batch_cat, batch_num + noise)
                        loss = criterion(outputs, batch_y)

                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    outputs = model(v_cat, v_num)
                    val_loss = criterion(outputs, v_y).item()
                    val_preds = outputs.argmax(dim=1).cpu().numpy()
                    val_acc = accuracy_score(y_val, val_preds)

                scheduler.step(val_acc)
                checkpoints.append(
                    (val_acc, -val_loss, copy.deepcopy(model.state_dict()))
                )
                checkpoints.sort(
                    key=lambda checkpoint: (checkpoint[0], checkpoint[1]),
                    reverse=True,
                )
                if len(checkpoints) > self.top_k_checkpoints:
                    checkpoints.pop()

            averaged = copy.deepcopy(checkpoints[0][2])
            for key in averaged:
                averaged[key] = (
                    torch.stack(
                        [checkpoint[2][key].float() for checkpoint in checkpoints],
                        dim=0,
                    )
                    .mean(dim=0)
                    .to(averaged[key].dtype)
                )
            model.load_state_dict(averaged)
            model.eval()
            self.models_.append(model)

        return self

    def predict_proba(self, X):
        x_cat = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        x_num = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        probabilities = []
        with torch.no_grad():
            for model in self.models_:
                probabilities.append(
                    torch.softmax(model(x_cat, x_num), dim=1)[:, 1]
                    .cpu()
                    .numpy()
                )
        positive = np.mean(probabilities, axis=0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        # The final recorded node performs thresholding after the 50/50 blend.
        return self.predict_proba(X)


def blend_predictions(cb_probabilities, nn_probabilities, mode):
    if mode == "fit":
        return None
    probabilities = (
        0.5 * cb_probabilities[:, 1] + 0.5 * nn_probabilities[:, 1]
    )
    return (probabilities >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records the read without executing it.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and preserve the exact stratified
    #    outer 80/20 holdout from the original train_test_split.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing. Medians and ticket frequencies are learned
    #    from outer-training rows, fixing the original full-table leakage.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = raw_title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    title = title.where(
        title.isin(["Mr", "Miss", "Mrs", "Master"]), "Rare"
    )
    title = title.where(raw_title.notna(), "Missing")

    age = X[["Pclass", "Sex", "Age"]].skb.apply(GroupAgeImputer())
    ticket = X[["Ticket"]].skb.apply(TicketFrequency())
    fare = X[["Fare"]].skb.apply(
        SimpleImputer(strategy="median").set_output(transform="pandas")
    )

    family_size = X["SibSp"] + X["Parch"] + 1
    family_tier = family_size.astype(str).where(
        family_size == 1, "SmallFamily"
    )
    family_tier = family_tier.where(family_size != 1, "IsAlone")
    family_tier = family_tier.where(
        (family_size == 1) | (family_size <= 4), "LargeFamily"
    )

    features = X.assign(
        Title=title,
        Age=age["Age"],
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str),
        FamilySize=family_size,
        FamilyTier=family_tier,
        HasCabin=X["Cabin"].notna().astype(int),
        TicketFreq=ticket["TicketFreq"],
        Embarked=X["Embarked"].fillna("Missing"),
        Fare=fare["Fare"],
    )[FEATURE_COLS]

    # 4. CatBoost early stopping remains explicit in the graph.
    #    The original used the scored holdout for early stopping; instead carve
    #    a stratified 20% eval set from outer-training rows. Its original leaky
    #    score cannot be reproduced honestly. Patience and iterations are kept.
    cb_features = features[CAT_COLS].astype(str).skb.concat(
        [features[NUM_COLS]], axis=1
    )
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_fit = cb_parts[cb_parts["_internal_fit"]]
    cb_eval = cb_parts[~cb_parts["_internal_fit"]]

    cb_probs = cb_fit[FEATURE_COLS].skb.apply(
        CatBoostProbabilities(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_fit["_internal_y"],
        fit_kwargs={
            "eval_set": (
                cb_eval[FEATURE_COLS],
                cb_eval["_internal_y"],
            ),
            "early_stopping_rounds": 50,
        },
    )

    # 5. Separate visible neural encoders and scaler. OrdinalEncoder uses the
    #    same sorted categories as LabelEncoder; unseen values map to zero.
    nn_categories = (
        features[CAT_COLS]
        .astype(str)
        .skb.apply(
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int64,
            ).set_output(transform="pandas")
        )
        .replace(-1, 0)
        .astype(np.int64)
    )
    nn_numeric = features[NUM_COLS].astype(np.float32).skb.apply(
        StandardScaler().set_output(transform="pandas")
    )
    nn_features = nn_categories.skb.concat([nn_numeric], axis=1)

    # The scheduler and checkpoint selection originally also saw scored rows.
    # Keep their behavior on an internal holdout, not the outer scored holdout.
    nn_parts = nn_features.skb.apply(GetXY(), y=y)
    nn_fit = nn_parts[nn_parts["_internal_fit"]]
    nn_eval = nn_parts[~nn_parts["_internal_fit"]]
    nn_probs = nn_fit[FEATURE_COLS].skb.apply(
        CheckpointNeuralEnsemble(),
        y=nn_fit["_internal_y"],
        fit_kwargs={
            "X_val": nn_eval[FEATURE_COLS],
            "y_val": nn_eval["_internal_y"],
        },
    )

    # 6. Preserve probability averaging and >= 0.5 decoding, gated during fit.
    pred = cb_probs.skb.apply_func(
        blend_predictions, nn_probs, skrub.eval_mode()
    )

    # 7. Score — only mark_as_X supplies the outer splitter.
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
