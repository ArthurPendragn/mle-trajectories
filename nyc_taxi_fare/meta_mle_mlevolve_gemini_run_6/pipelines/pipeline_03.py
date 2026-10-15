import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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

    def __init__(self, lr=0.001, batch_size=2048, epochs=12, weight_decay=0.0001, seed=42):
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
        self.net_ = ResNetModel(in_features).to(device)

        dataset = TensorDataset(
            torch.from_numpy(X_scaled.astype(np.float32)),
            torch.from_numpy(y_arr),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

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
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        self.net_.eval()
        preds = []
        with torch.no_grad():
            for batch_x, in loader:
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

    # Spatial coordinate differences in kilometers
    dx_km = (d_lon - p_lon) * 84.0
    dy_km = (d_lat - p_lat) * 111.0
    abs_dx = dx_km.abs()
    abs_dy = dy_km.abs()

    euclidean = (dx_km**2 + dy_km**2).pow(0.5)
    cardinal_manhattan = abs_dx + abs_dy

    # Trajectory unit direction vector (bearing)
    safe_dist = euclidean + 1e-5
    bearing_x = dx_km / safe_dist
    bearing_y = dy_km / safe_dist

    # Manhattan street grid rotation (~29 degrees clockwise from North)
    theta = 29.0 * np.pi / 180.0
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    rot_p_x = (p_lon * 84.0) * cos_t - (p_lat * 111.0) * sin_t
    rot_p_y = (p_lon * 84.0) * sin_t + (p_lat * 111.0) * cos_t
    rot_d_x = (d_lon * 84.0) * cos_t - (d_lat * 111.0) * sin_t
    rot_d_y = (d_lon * 84.0) * sin_t + (d_lat * 111.0) * cos_t

    rot_dx = (rot_d_x - rot_p_x).abs()
    rot_dy = (rot_d_y - rot_p_y).abs()
    rot_manhattan = rot_dx + rot_dy

    # Airport distances
    jfk_dist_p = (((p_lon - -73.7781) * 84.0) ** 2 + ((p_lat - 40.6413) * 111.0) ** 2).pow(0.5)
    jfk_dist_d = (((d_lon - -73.7781) * 84.0) ** 2 + ((d_lat - 40.6413) * 111.0) ** 2).pow(0.5)
    lga_dist_p = (((p_lon - -73.8740) * 84.0) ** 2 + ((p_lat - 40.7769) * 111.0) ** 2).pow(0.5)
    lga_dist_d = (((d_lon - -73.8740) * 84.0) ** 2 + ((d_lat - 40.7769) * 111.0) ** 2).pow(0.5)
    ewr_dist_p = (((p_lon - -74.1745) * 84.0) ** 2 + ((p_lat - 40.6895) * 111.0) ** 2).pow(0.5)
    ewr_dist_d = (((d_lon - -74.1745) * 84.0) ** 2 + ((d_lat - 40.6895) * 111.0) ** 2).pow(0.5)

    # Airport proximity indicators
    jfk_p_near = (jfk_dist_p <= 2.5).astype("float64")
    jfk_d_near = (jfk_dist_d <= 2.5).astype("float64")
    lga_p_near = (lga_dist_p <= 2.0).astype("float64")
    lga_d_near = (lga_dist_d <= 2.0).astype("float64")
    ewr_p_near = (ewr_dist_p <= 2.5).astype("float64")
    ewr_d_near = (ewr_dist_d <= 2.5).astype("float64")

    # Temporal features
    dt = X["pickup_datetime"].skb.apply_func(pd.to_datetime)
    year_raw = dt.dt.year.astype("float64")
    month_raw = dt.dt.month.astype("float64")
    day_raw = dt.dt.day.astype("float64")
    hour = dt.dt.hour.astype("float64")
    dayofweek = dt.dt.dayofweek.astype("float64")

    # Continuous fractional year capturing mid-year fare increases
    frac_year = (year_raw + (month_raw - 1.0) / 12.0 + (day_raw - 1.0) / 365.0 - 2009.0) / 6.0

    # Cyclical encodings
    hour_rad = hour * (2.0 * np.pi / 24.0)
    hour_sin = hour_rad.skb.apply_func(np.sin)
    hour_cos = hour_rad.skb.apply_func(np.cos)

    dow_rad = dayofweek * (2.0 * np.pi / 7.0)
    dow_sin = dow_rad.skb.apply_func(np.sin)
    dow_cos = dow_rad.skb.apply_func(np.cos)

    month_rad = (month_raw - 1.0) * (2.0 * np.pi / 12.0)
    month_sin = month_rad.skb.apply_func(np.sin)
    month_cos = month_rad.skb.apply_func(np.cos)

    features = (
        X[["passenger_count"]]
        .astype("float64")
        .assign(
            pickup_longitude=p_lon,
            pickup_latitude=p_lat,
            dropoff_longitude=d_lon,
            dropoff_latitude=d_lat,
            dx_km=dx_km,
            dy_km=dy_km,
            abs_dx=abs_dx,
            abs_dy=abs_dy,
            euclidean=euclidean,
            cardinal_manhattan=cardinal_manhattan,
            bearing_x=bearing_x,
            bearing_y=bearing_y,
            rot_dx=rot_dx,
            rot_dy=rot_dy,
            rot_manhattan=rot_manhattan,
            jfk_dist_p=jfk_dist_p,
            jfk_dist_d=jfk_dist_d,
            jfk_p_near=jfk_p_near,
            jfk_d_near=jfk_d_near,
            lga_dist_p=lga_dist_p,
            lga_dist_d=lga_dist_d,
            lga_p_near=lga_p_near,
            lga_d_near=lga_d_near,
            ewr_dist_p=ewr_dist_p,
            ewr_dist_d=ewr_dist_d,
            ewr_p_near=ewr_p_near,
            ewr_d_near=ewr_d_near,
            frac_year=frac_year,
            hour_sin=hour_sin,
            hour_cos=hour_cos,
            dow_sin=dow_sin,
            dow_cos=dow_cos,
            month_sin=month_sin,
            month_cos=month_cos,
        )
        .skb.set_name("features")
    )

    model = TorchTaxiRegressor(lr=0.001, batch_size=2048, epochs=12)
    pred = features.skb.apply(model, y=y).skb.set_name("model")
    return {"pred": pred, "scoring": setup["scoring"]}