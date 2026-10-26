import copy
import math

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        self.fallback_ = self._fill_groups(X).median()
        return self

    def _fill_groups(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X["Age"].fillna(medians)

    def transform(self, X):
        return self._fill_groups(X).fillna(self.fallback_).to_frame("Age")


class TicketFrequencyEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class GetXY(TransformerMixin, BaseEstimator):
    """Expose the early-stopping split through a pandas container."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        train_indices, val_indices = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        # Keep the input rows and their index unchanged at this transformer.
        # Recorded selections below recover each split in its shuffled order.
        order = np.empty(len(X), dtype=np.int64)
        order[train_indices] = np.arange(len(train_indices))
        order[val_indices] = np.arange(len(val_indices))
        is_eval = np.zeros(len(X), dtype=bool)
        is_eval[val_indices] = True
        return X.assign(
            __es_target=np.asarray(y),
            __es_eval=is_eval,
            __es_order=order,
        )

    def transform(self, X):
        # All prediction rows follow the training-feature branch. Keep the same
        # columns available even if the executor visits fit-only dependencies.
        return X.assign(
            __es_target=np.nan,
            __es_eval=False,
            __es_order=np.arange(len(X)),
        )


class CatBoostProbabilityClassifier(CatBoostClassifier):
    def __sklearn_clone__(self):
        return CatBoostProbabilityClassifier(**self.get_params(deep=False))

    def predict(self, X, **kwargs):
        # Expose probabilities through the default prediction method, avoiding
        # version-dependent method-selection keywords on DataOps.apply.
        return self.predict_proba(X, **kwargs)


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n, d in zip(cat_dims, emb_dims)]
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


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class EMATabularClassifier(ClassifierMixin, BaseEstimator):
    """The torch training loop; all fitted feature preprocessing is external."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cat_values = X[CAT_COLS].to_numpy(dtype=np.int64)
        num_values = X[NUM_COLS].to_numpy(dtype=np.float32)
        cat_dims = (cat_values.max(axis=0) + 1).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]
        model = TabularNeuralNet(
            cat_dims, emb_dims, num_features=num_values.shape[1]
        ).to(self.device_)
        ema_model = copy.deepcopy(model).to(self.device_)
        for param in ema_model.parameters():
            param.requires_grad = False

        criterion = FocalCrossEntropyLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-3, weight_decay=1e-4
        )
        t_cat = torch.tensor(cat_values, dtype=torch.long, device=self.device_)
        t_num = torch.tensor(num_values, dtype=torch.float32, device=self.device_)
        t_y = torch.tensor(np.asarray(y), dtype=torch.long, device=self.device_)

        batch_size = 32
        num_epochs = 150
        num_samples = t_cat.size(0)
        total_steps = num_epochs * math.ceil(num_samples / batch_size)
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda
        )
        mixup_alpha = 0.2
        ema_decay = 0.995

        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_indices = permutation[i : i + batch_size]
                batch_cat = t_cat[batch_indices]
                batch_num = t_num[batch_indices]
                batch_y = t_y[batch_indices]
                cur_bs = batch_cat.size(0)

                if mixup_alpha > 0 and cur_bs > 1:
                    lam = float(
                        torch.distributions.Beta(
                            mixup_alpha, mixup_alpha
                        ).sample().item()
                    )
                    mix_perm = torch.randperm(cur_bs)
                    batch_num_mixed = (
                        lam * batch_num + (1.0 - lam) * batch_num[mix_perm]
                    )
                    cat_mask = (
                        torch.rand(batch_cat.shape, device=self.device_) < lam
                    )
                    batch_cat_mixed = torch.where(
                        cat_mask, batch_cat, batch_cat[mix_perm]
                    )
                    optimizer.zero_grad()
                    outputs = model(batch_cat_mixed, batch_num_mixed)
                    loss = (
                        lam * criterion(outputs, batch_y)
                        + (1.0 - lam) * criterion(outputs, batch_y[mix_perm])
                    )
                else:
                    optimizer.zero_grad()
                    outputs = model(batch_cat, batch_num)
                    loss = criterion(outputs, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Preserve parameter-only EMA: the original did not update the
                # shadow model's batch-normalization buffers.
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(
                            p.data, alpha=1.0 - ema_decay
                        )

            # Original validation monitoring updated only an unused scalar:
            # no checkpoint selection, early stopping, or training changes.

        self.model_ = ema_model.eval()
        return self

    def predict_proba(self, X):
        t_cat = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        t_num = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(t_cat, t_num), dim=1).cpu().numpy()

    def predict(self, X):
        # This branch feeds the recorded ensemble, not a standalone scorer.
        return self.predict_proba(X)


def ensemble_predictions(cb_probs, nn_probs, mode):
    if mode == "fit":
        return None
    probs = 0.5 * np.asarray(cb_probs)[:, 1] + 0.5 * np.asarray(nn_probs)[:, 1]
    return (probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read without accessing data during import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — raw target and the exact original stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and separate fitted preprocessing nodes.
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
    ).fillna("Rare")
    title = title.where(raw_title.notna(), "Missing")

    # The original learned these statistics before splitting. Fit them only on
    # outer training rows to eliminate that leakage.
    age = X[["Pclass", "Sex", "Age"]].skb.apply(GroupedAgeImputer())
    ticket_freq = X[["Ticket"]].skb.apply(TicketFrequencyEncoder())
    fare = X[["Fare"]].skb.apply(
        SimpleImputer(strategy="median").set_output(transform="pandas")
    )

    family_size = X["SibSp"] + X["Parch"] + 1
    family_tier = family_size.astype(str).where(family_size <= 4, "LargeFamily")
    family_tier = family_tier.where(
        ~((family_size != 1) & (family_size <= 4)), "SmallFamily"
    )
    family_tier = family_tier.where(family_size != 1, "IsAlone")

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
    )[CAT_COLS + NUM_COLS]
    cat_strings = features[CAT_COLS].astype(str)
    cb_features = cat_strings.skb.concat([features[NUM_COLS]], axis=1)

    # 3a. Keep CatBoost early stopping visible in the graph. This skrub version
    #     lacks how="no_wrap", so GetXY exposes a frame rather than a dict.
    #     Recorded row selections extract its four pieces without opaque fitting.
    #     The original early-stopped on scored rows; use an inner stratified 20%
    #     split instead, preserving patience and iterations without that leakage.
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_train = cb_parts[~cb_parts["__es_eval"]].sort_values("__es_order")
    cb_eval = cb_parts[cb_parts["__es_eval"]].sort_values("__es_order")

    cb_pred = cb_train[CAT_COLS + NUM_COLS].skb.apply(
        CatBoostProbabilityClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_train["__es_target"],
        fit_kwargs={
            "eval_set": (
                cb_eval[CAT_COLS + NUM_COLS],
                cb_eval["__es_target"],
            ),
            "early_stopping_rounds": 50,
        },
    )

    # 3b. OrdinalEncoder reproduces the sorted LabelEncoder vocabularies and the
    #     original fallback of zero for unseen validation categories.
    nn_cat = cat_strings.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=0,
            dtype=np.int64,
        ).set_output(transform="pandas")
    )
    nn_num = features[NUM_COLS].astype(np.float32).skb.apply(
        StandardScaler().set_output(transform="pandas")
    )
    nn_features = nn_cat.skb.concat([nn_num], axis=1)[CAT_COLS + NUM_COLS]
    nn_pred = nn_features.skb.apply(EMATabularClassifier(random_state=42), y=y)

    # 3c. The original scored only this ensemble. Preserve its probability
    #     arithmetic and threshold, gated to avoid arithmetic on fitted models.
    pred = cb_pred.skb.apply_func(
        ensemble_predictions, nn_pred, skrub.eval_mode()
    )

    # 4. Score — use the splitter declared on mark_as_X, not a second cv argument.
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
