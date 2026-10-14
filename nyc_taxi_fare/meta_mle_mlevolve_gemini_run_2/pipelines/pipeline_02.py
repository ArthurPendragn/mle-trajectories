import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TabularResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(self, in_features, hidden_dim=256, num_blocks=3, dropout=0.05):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            TabularResBlock(hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)


class PyTorchTabularRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        hidden_dim=256,
        num_blocks=3,
        dropout=0.05,
        lr=2e-3,
        weight_decay=1e-4,
        batch_size=4096,
        epochs=12,
        random_state=42,
    ):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

        if hasattr(X, "values"):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        if hasattr(y, "values"):
            y_arr = y.values.astype(np.float32)
        else:
            y_arr = np.asarray(y, dtype=np.float32)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_features = X_arr.shape[1]

        self.net_ = TabularResNet(
            in_features=in_features,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
        ).to(device)

        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            self.net_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-5,
        )
        criterion = nn.MSELoss()

        self.net_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                pred = self.net_(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=5.0)
                optimizer.step()
            scheduler.step()

        return self

    def predict(self, X):
        self.net_.eval()
        if hasattr(X, "values"):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        device = next(self.net_.parameters()).device
        dataset = TensorDataset(torch.from_numpy(X_arr))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
        )

        preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                out = self.net_(batch_x)
                preds.append(out.cpu().numpy())

        all_preds = np.concatenate(preds, axis=0)
        return np.clip(all_preds, 2.5, 500.0)

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).predict(X)


def build():
    setup = build_evaluation()
    X = setup["X"]
    y = setup["y"]

    dlon = X["dropoff_longitude"] - X["pickup_longitude"]
    dlat = X["dropoff_latitude"] - X["pickup_latitude"]
    abs_dlon = dlon.abs()
    abs_dlat = dlat.abs()

    euclidean = ((dlon * 82.0) ** 2 + (dlat * 111.0) ** 2) ** 0.5
    manhattan = (abs_dlon * 82.0) + (abs_dlat * 111.0)

    jfk_dist_pick = (
        ((X["pickup_longitude"] - (-73.7781)) * 82.0) ** 2
        + ((X["pickup_latitude"] - 40.6413) * 111.0) ** 2
    ) ** 0.5
    jfk_dist_drop = (
        ((X["dropoff_longitude"] - (-73.7781)) * 82.0) ** 2
        + ((X["dropoff_latitude"] - 40.6413) * 111.0) ** 2
    ) ** 0.5
    lga_dist_pick = (
        ((X["pickup_longitude"] - (-73.8740)) * 82.0) ** 2
        + ((X["pickup_latitude"] - 40.7769) * 111.0) ** 2
    ) ** 0.5
    lga_dist_drop = (
        ((X["dropoff_longitude"] - (-73.8740)) * 82.0) ** 2
        + ((X["dropoff_latitude"] - 40.7769) * 111.0) ** 2
    ) ** 0.5
    ewr_dist_pick = (
        ((X["pickup_longitude"] - (-74.1745)) * 82.0) ** 2
        + ((X["pickup_latitude"] - 40.6895) * 111.0) ** 2
    ) ** 0.5
    ewr_dist_drop = (
        ((X["dropoff_longitude"] - (-74.1745)) * 82.0) ** 2
        + ((X["dropoff_latitude"] - 40.6895) * 111.0) ** 2
    ) ** 0.5

    dt = X["pickup_datetime"].skb.apply_func(
        pd.to_datetime, format="%Y-%m-%d %H:%M:%S UTC", errors="coerce"
    )
    hour = dt.dt.hour.astype("float64").fillna(12.0)
    weekday = dt.dt.dayofweek.astype("float64").fillna(3.0)
    month = dt.dt.month.astype("float64").fillna(6.0)
    year = dt.dt.year.astype("float64").fillna(2012.0)
    year_diff = year - 2009.0

    geo_temporal_features = X.assign(
        dlon=dlon,
        dlat=dlat,
        abs_dlon=abs_dlon,
        abs_dlat=abs_dlat,
        euclidean=euclidean,
        manhattan=manhattan,
        jfk_dist_pick=jfk_dist_pick,
        jfk_dist_drop=jfk_dist_drop,
        lga_dist_pick=lga_dist_pick,
        lga_dist_drop=lga_dist_drop,
        ewr_dist_pick=ewr_dist_pick,
        ewr_dist_drop=ewr_dist_drop,
        hour=hour,
        weekday=weekday,
        month=month,
        year_diff=year_diff,
    )[
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "dlon",
            "dlat",
            "abs_dlon",
            "abs_dlat",
            "euclidean",
            "manhattan",
            "jfk_dist_pick",
            "jfk_dist_drop",
            "lga_dist_pick",
            "lga_dist_drop",
            "ewr_dist_pick",
            "ewr_dist_drop",
            "hour",
            "weekday",
            "month",
            "year_diff",
        ]
    ].skb.set_name("geo_temporal_features")

    scaled_features = geo_temporal_features.skb.apply(
        StandardScaler()
    ).skb.set_name("scaled_features")

    model = PyTorchTabularRegressor(
        hidden_dim=256,
        num_blocks=3,
        dropout=0.05,
        lr=2e-3,
        weight_decay=1e-4,
        batch_size=4096,
        epochs=12,
        random_state=42,
    )

    pred = scaled_features.skb.apply(model, y=y).skb.set_name("model")

    return {"pred": pred, "scoring": setup["scoring"]}