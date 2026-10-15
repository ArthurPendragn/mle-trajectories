import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class MLPModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.res1 = ResNetBlock(256)
        self.res2 = ResNetBlock(256)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)


class TorchTaxiRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        arch="mlp",
        lr=1e-3,
        batch_size=2048,
        epochs=12,
        weight_decay=1e-4,
        seed=42,
    ):
        self.arch = arch
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        y_arr = y_arr.astype(np.float32).reshape(-1, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device

        in_features = X_scaled.shape[1]
        if self.arch == "resnet":
            self.net_ = ResNetModel(in_features)
        else:
            self.net_ = MLPModel(in_features)
        self.net_.to(device)

        dataset = TensorDataset(
            torch.from_numpy(X_scaled.astype(np.float32)),
            torch.from_numpy(y_arr),
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        self.net_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                pred = self.net_(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def predict(self, X):
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        X_scaled = self.scaler_.transform(X_arr)

        dataset = TensorDataset(torch.from_numpy(X_scaled.astype(np.float32)))
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False
        )

        self.net_.eval()
        preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device_)
                out = self.net_(batch_x)
                preds.append(out.cpu().numpy().ravel())

        preds = np.concatenate(preds)
        preds = np.clip(preds, 2.5, 500.0)
        return preds

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).predict(X)


def build():
    setup = build_evaluation()
    X = setup["X"]
    y = setup["y"]

    p_lon = X["pickup_longitude"]
    p_lat = X["pickup_latitude"]
    d_lon = X["dropoff_longitude"]
    d_lat = X["dropoff_latitude"]

    dlon = d_lon - p_lon
    dlat = d_lat - p_lat
    abs_dlon = dlon.abs()
    abs_dlat = dlat.abs()

    euclidean = ((dlon * 84.0) ** 2 + (dlat * 111.0) ** 2).pow(0.5)
    manhattan = abs_dlon * 84.0 + abs_dlat * 111.0

    jfk_dist_p = (
        ((p_lon - (-73.7781)) * 84.0) ** 2 + ((p_lat - 40.6413) * 111.0) ** 2
    ).pow(0.5)
    jfk_dist_d = (
        ((d_lon - (-73.7781)) * 84.0) ** 2 + ((d_lat - 40.6413) * 111.0) ** 2
    ).pow(0.5)
    lga_dist_p = (
        ((p_lon - (-73.8740)) * 84.0) ** 2 + ((p_lat - 40.7769) * 111.0) ** 2
    ).pow(0.5)
    lga_dist_d = (
        ((d_lon - (-73.8740)) * 84.0) ** 2 + ((d_lat - 40.7769) * 111.0) ** 2
    ).pow(0.5)
    ewr_dist_p = (
        ((p_lon - (-74.1745)) * 84.0) ** 2 + ((p_lat - 40.6895) * 111.0) ** 2
    ).pow(0.5)
    ewr_dist_d = (
        ((d_lon - (-74.1745)) * 84.0) ** 2 + ((d_lat - 40.6895) * 111.0) ** 2
    ).pow(0.5)

    dt = X["pickup_datetime"].skb.apply_func(pd.to_datetime)
    year = (dt.dt.year.astype("float64") - 2009.0) / 6.0
    hour = dt.dt.hour.astype("float64")
    dayofweek = dt.dt.dayofweek.astype("float64")

    hour_rad = hour * (2.0 * np.pi / 24.0)
    hour_sin = hour_rad.skb.apply_func(np.sin)
    hour_cos = hour_rad.skb.apply_func(np.cos)

    dow_rad = dayofweek * (2.0 * np.pi / 7.0)
    dow_sin = dow_rad.skb.apply_func(np.sin)
    dow_cos = dow_rad.skb.apply_func(np.cos)

    features = (
        X[["passenger_count"]]
        .astype("float64")
        .assign(
            pickup_longitude=p_lon,
            pickup_latitude=p_lat,
            dropoff_longitude=d_lon,
            dropoff_latitude=d_lat,
            dlon=dlon,
            dlat=dlat,
            abs_dlon=abs_dlon,
            abs_dlat=abs_dlat,
            euclidean=euclidean,
            manhattan=manhattan,
            jfk_dist_p=jfk_dist_p,
            jfk_dist_d=jfk_dist_d,
            lga_dist_p=lga_dist_p,
            lga_dist_d=lga_dist_d,
            ewr_dist_p=ewr_dist_p,
            ewr_dist_d=ewr_dist_d,
            year=year,
            hour_sin=hour_sin,
            hour_cos=hour_cos,
            dow_sin=dow_sin,
            dow_cos=dow_cos,
        )
        .skb.set_name("features")
    )

    arch_choice = skrub.choose_from(
        {"mlp_3layer": "mlp", "resnet_4layer": "resnet"}, name="arch"
    )
    model = TorchTaxiRegressor(
        arch=arch_choice, lr=1e-3, batch_size=2048, epochs=12
    )

    pred = features.skb.apply(model, y=y).skb.set_name("model")

    return {
        "pred": pred,
        "scoring": setup["scoring"],
    }