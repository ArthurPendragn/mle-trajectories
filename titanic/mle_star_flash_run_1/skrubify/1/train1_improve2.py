import copy

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.optim.swa_utils import AveragedModel


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        group_values = self.group_medians_.reindex(
            pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        )
        group_values = pd.Series(group_values.to_numpy(), index=X.index)
        self.fallback_ = X["Age"].fillna(group_values).median()
        return self

    def transform(self, X):
        group_values = self.group_medians_.reindex(
            pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        )
        group_values = pd.Series(group_values.to_numpy(), index=X.index)
        return X.assign(Age=X["Age"].fillna(group_values).fillna(self.fallback_))


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
    """Expose an inner split in a pandas container for older skrub versions.

    The installed API does not accept how="no_wrap". Returning a frame rather
    than a dict lets the ordinary transformer application preserve the split.
    Recorded selections downstream recover the four pieces.
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
        train = X.iloc[train_idx].assign(
            _inner_fit=True, _inner_target=target[train_idx]
        )
        val = X.iloc[val_idx].assign(
            _inner_fit=False, _inner_target=target[val_idx]
        )
        return pd.concat([train, val], axis=0)

    def transform(self, X):
        # All prediction rows follow the fit-feature branch. Fit-only target
        # and eval selections remain well-defined but are not used to predict.
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
            [nn.Embedding(n, d) for n, d in zip(cat_dims, emb_dims)]
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
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Original torch loop, SWA, and temperature calibration, run per fit."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cat_dims = np.asarray(cat_dims, dtype=int).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]
        model = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        x_cat = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        x_num = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        target = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_cat = torch.tensor(
            X_val[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        val_num = torch.tensor(
            X_val[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        val_target = torch.tensor(
            np.asarray(y_val), dtype=torch.long, device=self.device_
        )

        criterion = nn.CrossEntropyLoss()
        batch_size = 32
        num_epochs = 120
        swa_start_epoch = 80
        feature_dropout_rate = 0.05
        steps_per_epoch = (x_cat.size(0) + batch_size - 1) // batch_size

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=8e-3,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.15,
            div_factor=10.0,
            final_div_factor=1e3,
        )
        swa_model = AveragedModel(model)
        best_val_loss = float("inf")
        best_model_weights = None
        patience = 25
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(x_cat.size(0), device=self.device_)
            for i in range(0, x_cat.size(0), batch_size):
                batch_indices = permutation[i : i + batch_size]
                batch_cat = x_cat[batch_indices]
                batch_num = x_num[batch_indices].clone()
                batch_y = target[batch_indices]

                if feature_dropout_rate > 0 and batch_num.size(1) > 0:
                    mask = (
                        torch.rand(batch_num.shape, device=self.device_)
                        >= feature_dropout_rate
                    ).float()
                    batch_num = batch_num * mask

                optimizer.zero_grad()
                outputs = model(batch_cat, batch_num)
                loss = criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()

            if epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(val_cat, val_num), val_target).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience and epoch >= swa_start_epoch + 15:
                break

        if epoch >= swa_start_epoch:
            self.model_ = swa_model
        else:
            self.model_ = model
            if best_model_weights is not None:
                self.model_.load_state_dict(best_model_weights)

        # Preserve the original SWA behavior: it did not call update_bn.
        self.model_.eval()
        with torch.no_grad():
            val_logits = self.model_(val_cat, val_num)

        temperature = nn.Parameter(torch.ones(1, device=self.device_))
        temp_optimizer = torch.optim.LBFGS(
            [temperature], lr=0.01, max_iter=50
        )

        def temp_eval():
            temp_optimizer.zero_grad()
            scaled_logits = val_logits / torch.clamp(temperature, min=1e-3)
            loss = criterion(scaled_logits, val_target)
            loss.backward()
            return loss

        temp_optimizer.step(temp_eval)
        self.temperature_ = temperature.detach()
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
            logits = self.model_(x_cat, x_num)
            calibrated = logits / torch.clamp(self.temperature_, min=1e-3)
            return torch.softmax(calibrated, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    return (0.5 * cb_probability + 0.5 * nn_probability >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records, but does not execute, the CSV read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and design matrix. This splitter reproduces the
    #    original stratified 80/20 train_test_split, including its exact rows.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Age medians, ticket frequencies, and
    #    fare medians are learned on outer training rows rather than the whole
    #    table, fixing the original preprocessing leakage.
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

    ages = X[["Pclass", "Sex", "Age"]].skb.apply(GroupedAgeImputer())
    ticket_frequency = X[["Ticket"]].skb.apply(TicketFrequency())
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))

    features = X.assign(Title=title)
    features = features.assign(Age=ages["Age"])
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str)
        + "_"
        + features["Pclass"].astype(str)
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    family_size = features["FamilySize"]
    family_tier = (
        family_size.astype(str)
        .where(family_size <= 4, "LargeFamily")
        .where(~(family_size <= 4), "SmallFamily")
        .where(family_size != 1, "IsAlone")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=features["Cabin"].notna().astype(int))
    features = features.assign(TicketFreq=ticket_frequency["TicketFreq"])
    features = features.assign(Embarked=features["Embarked"].fillna("Missing"))
    features = features.assign(Fare=fare["Fare"])

    categorical = features[CAT_COLS].astype(str)
    numeric = features[NUM_COLS]
    cb_features = categorical.skb.concat([numeric], axis=1)

    # Sorted ordinal vocabularies match LabelEncoder; unknown categories map
    # to position zero, exactly as in the original validation preparation.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_categorical = nn_categorical.replace(-1, 0)
    nn_numeric = numeric.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_categorical.skb.concat([nn_numeric], axis=1)
    cat_dims = (nn_categorical.max(axis=0) + 1).to_numpy()

    # 4. Expose inner validation splits as recorded selections and fit_kwargs.
    #    The original scored the same hold-out used for early stopping and
    #    temperature calibration. Keep every training/patience setting, but use
    #    a stratified 20% subset of outer training rows for those purposes.
    #    GetXY returns a frame because this skrub API lacks how="no_wrap".
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_train = cb_parts[cb_parts["_inner_fit"]]
    cb_val = cb_parts[~cb_parts["_inner_fit"]]
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
            "eval_set": (cb_val[FEATURE_COLS], cb_val["_inner_target"]),
            "early_stopping_rounds": 50,
        },
    )

    nn_parts = nn_features.skb.apply(GetXY(), y=y)
    nn_train = nn_parts[nn_parts["_inner_fit"]]
    nn_val = nn_parts[~nn_parts["_inner_fit"]]
    nn_probability = nn_train[FEATURE_COLS].skb.apply(
        NeuralPositiveProbability(random_state=42),
        y=nn_train["_inner_target"],
        fit_kwargs={
            "X_val": nn_val[FEATURE_COLS],
            "y_val": nn_val["_inner_target"],
            "cat_dims": cat_dims,
        },
    )

    # Preserve positive-probability averaging and the >= 0.5 threshold.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 5. Score with the original accuracy metric. The outer splitter lives
    #    exclusively on mark_as_X.
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
