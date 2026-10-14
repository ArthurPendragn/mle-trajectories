import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class MLPNet(nn.Module):
    def __init__(self, in_features, hidden_dims=(256, 256, 128)):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchTabularRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_dims=(256, 256, 128),
        lr=2e-3,
        weight_decay=1e-5,
        batch_size=4096,
        epochs=15,
        seed=42,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler_ = StandardScaler()
        X_mat = self.scaler_.fit_transform(X)
        X_mat = np.nan_to_num(X_mat, nan=0.0).astype(np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        X_t = torch.from_numpy(X_mat)
        y_t = torch.from_numpy(y_arr)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        model = MLPNet(
            in_features=X_mat.shape[1], hidden_dims=self.hidden_dims
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        self.model_ = model
        self.device_ = device
        return self

    def predict(self, X):
        X_mat = self.scaler_.transform(X)
        X_mat = np.nan_to_num(X_mat, nan=0.0).astype(np.float32)
        X_t = torch.from_numpy(X_mat)
        loader = DataLoader(X_t, batch_size=self.batch_size * 2, shuffle=False)

        self.model_.eval()
        preds = []
        with torch.no_grad():
            for bx in loader:
                bx = bx.to(self.device_)
                pred = self.model_(bx)
                preds.append(pred.cpu().numpy())

        res = np.concatenate(preds, axis=0)
        return np.clip(res, 2.5, 500.0)


def build():
    setup = build_evaluation()
    X = setup["X"]
    y = setup["y"]

    dt = X["pickup_datetime"].skb.apply_func(pd.to_datetime)
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    dayofweek = dt.dt.dayofweek
    hour = dt.dt.hour

    d_lon = (X["dropoff_longitude"] - X["pickup_longitude"]).abs()
    d_lat = (X["dropoff_latitude"] - X["pickup_latitude"]).abs()

    dx = (X["dropoff_longitude"] - X["pickup_longitude"]) * 0.7575
    dy = X["dropoff_latitude"] - X["pickup_latitude"]
    dist_km = ((dx**2 + dy**2) ** 0.5) * 111.0

    jfk_pu = (
        ((X["pickup_longitude"] - (-73.7781)) * 0.7575) ** 2
        + (X["pickup_latitude"] - 40.6413) ** 2
    ) ** 0.5 * 111.0
    lga_pu = (
        ((X["pickup_longitude"] - (-73.8740)) * 0.7575) ** 2
        + (X["pickup_latitude"] - 40.7769) ** 2
    ) ** 0.5 * 111.0
    ewr_pu = (
        ((X["pickup_longitude"] - (-74.1745)) * 0.7575) ** 2
        + (X["pickup_latitude"] - 40.6895) ** 2
    ) ** 0.5 * 111.0
    midtown_pu = (
        ((X["pickup_longitude"] - (-73.9855)) * 0.7575) ** 2
        + (X["pickup_latitude"] - 40.7580) ** 2
    ) ** 0.5 * 111.0

    jfk_do = (
        ((X["dropoff_longitude"] - (-73.7781)) * 0.7575) ** 2
        + (X["dropoff_latitude"] - 40.6413) ** 2
    ) ** 0.5 * 111.0
    lga_do = (
        ((X["dropoff_longitude"] - (-73.8740)) * 0.7575) ** 2
        + (X["dropoff_latitude"] - 40.7769) ** 2
    ) ** 0.5 * 111.0
    ewr_do = (
        ((X["dropoff_longitude"] - (-74.1745)) * 0.7575) ** 2
        + (X["dropoff_latitude"] - 40.6895) ** 2
    ) ** 0.5 * 111.0
    midtown_do = (
        ((X["dropoff_longitude"] - (-73.9855)) * 0.7575) ** 2
        + (X["dropoff_latitude"] - 40.7580) ** 2
    ) ** 0.5 * 111.0

    features = (
        X[
            [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
                "passenger_count",
            ]
        ]
        .assign(
            year=year,
            month=month,
            day=day,
            dayofweek=dayofweek,
            hour=hour,
            d_lon=d_lon,
            d_lat=d_lat,
            dist_km=dist_km,
            jfk_pu=jfk_pu,
            lga_pu=lga_pu,
            ewr_pu=ewr_pu,
            midtown_pu=midtown_pu,
            jfk_do=jfk_do,
            lga_do=lga_do,
            ewr_do=ewr_do,
            midtown_do=midtown_do,
        )
        .skb.set_name("features")
    )

    model = TorchTabularRegressor()
    pred = features.skb.apply(model, y=y).skb.set_name("model")

    return {"pred": pred, "scoring": setup["scoring"]}