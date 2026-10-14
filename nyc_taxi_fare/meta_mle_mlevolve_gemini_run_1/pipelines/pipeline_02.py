import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import skrub


class TabularPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def _extract_raw(self, X):
        df = pd.DataFrame(index=X.index)

        p_lon = X["pickup_longitude"].astype("float64")
        p_lat = X["pickup_latitude"].astype("float64")
        d_lon = X["dropoff_longitude"].astype("float64")
        d_lat = X["dropoff_latitude"].astype("float64")

        df["p_lon"] = p_lon
        df["p_lat"] = p_lat
        df["d_lon"] = d_lon
        df["d_lat"] = d_lat

        df["d_lat_abs"] = (d_lat - p_lat).abs()
        df["d_lon_abs"] = (d_lon - p_lon).abs()
        df["euclidean"] = np.sqrt(df["d_lat_abs"] ** 2 + df["d_lon_abs"] ** 2)

        lat1 = np.radians(p_lat)
        lat2 = np.radians(d_lat)
        dlat = np.radians(d_lat - p_lat)
        dlon = np.radians(d_lon - p_lon)
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        df["haversine_km"] = 6371.0 * c

        df["manhattan_km"] = (
            111.0 * df["d_lat_abs"] + 111.0 * np.cos(lat1) * df["d_lon_abs"]
        )

        y_bear = np.sin(dlon) * np.cos(lat2)
        x_bear = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(
            lat2
        ) * np.cos(dlon)
        df["bearing"] = np.arctan2(y_bear, x_bear)

        landmarks = {
            "jfk": (-73.7781, 40.6413),
            "lga": (-73.8740, 40.7769),
            "ewr": (-74.1745, 40.6895),
            "midtown": (-73.9851, 40.7580),
        }
        for name, (l_lon, l_lat) in landmarks.items():
            l_lat_rad = np.radians(l_lat)
            dlat_p = np.radians(l_lat - p_lat)
            dlon_p = np.radians(l_lon - p_lon)
            ap = (
                np.sin(dlat_p / 2.0) ** 2
                + np.cos(lat1) * np.cos(l_lat_rad) * np.sin(dlon_p / 2.0) ** 2
            )
            df[f"p_dist_{name}"] = (
                6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(ap, 0.0, 1.0)))
            )

            dlat_d = np.radians(l_lat - d_lat)
            dlon_d = np.radians(l_lon - d_lon)
            ad = (
                np.sin(dlat_d / 2.0) ** 2
                + np.cos(lat2) * np.cos(l_lat_rad) * np.sin(dlon_d / 2.0) ** 2
            )
            df[f"d_dist_{name}"] = (
                6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(ad, 0.0, 1.0)))
            )

        df["passenger_count"] = X["passenger_count"].astype("float64")

        dt = pd.to_datetime(
            X["pickup_datetime"].astype(str).str.slice(0, 19),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        hour = dt.dt.hour.fillna(12).astype("float64")
        dow = dt.dt.dayofweek.fillna(2).astype("float64")
        month = dt.dt.month.fillna(6).astype("float64")
        year = dt.dt.year.fillna(2012).astype("float64")

        df["year"] = year
        df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
        df["sin_dow"] = np.sin(2.0 * np.pi * dow / 7.0)
        df["cos_dow"] = np.cos(2.0 * np.pi * dow / 7.0)
        df["sin_month"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        df["cos_month"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
        df["is_weekend"] = (dow >= 5.0).astype("float64")

        return df.fillna(0.0)

    def fit(self, X, y=None):
        raw = self._extract_raw(X)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(raw)
        self.columns_ = list(raw.columns)
        return self

    def transform(self, X):
        raw = self._extract_raw(X)
        scaled = self.scaler_.transform(raw)
        return pd.DataFrame(scaled, columns=self.columns_, index=X.index)


class MLPModule(nn.Module):

    def __init__(self, in_features, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.05))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        hidden_dims=(128, 64, 32),
        lr=1e-3,
        batch_size=2048,
        epochs=15,
        random_state=42,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device

        X_vals = (
            X.values.astype(np.float32)
            if hasattr(X, "values")
            else np.asarray(X, dtype=np.float32)
        )
        y_vals = (
            y.values.astype(np.float32)
            if hasattr(y, "values")
            else np.asarray(y, dtype=np.float32)
        )

        dataset = TensorDataset(
            torch.from_numpy(X_vals), torch.from_numpy(y_vals)
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        self.net_ = MLPModule(
            in_features=X_vals.shape[1], hidden_dims=self.hidden_dims
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        self.net_.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx = bx.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                out = self.net_(bx).squeeze(-1)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def predict(self, X):
        self.net_.eval()
        X_vals = (
            X.values.astype(np.float32)
            if hasattr(X, "values")
            else np.asarray(X, dtype=np.float32)
        )
        dataset = TensorDataset(torch.from_numpy(X_vals))
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False
        )

        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device_)
                out = self.net_(bx).squeeze(-1)
                preds.append(out.cpu().numpy())

        pred_arr = np.concatenate(preds, axis=0)
        return np.clip(pred_arr, 2.5, 500.0)


def build():
    setup = build_evaluation()
    X = setup["X"]
    y = setup["y"]
    scoring = setup["scoring"]

    geo_features = X.skb.apply(TabularPreprocessor()).skb.set_name(
        "geo_time_features"
    )

    model = skrub.choose_from(
        {
            "mlp_3layer_128": TorchMLPRegressor(
                hidden_dims=(128, 64, 32),
                lr=1e-3,
                batch_size=2048,
                epochs=15,
                random_state=42,
            ),
            "mlp_4layer_256": TorchMLPRegressor(
                hidden_dims=(256, 128, 64, 32),
                lr=1e-3,
                batch_size=2048,
                epochs=15,
                random_state=42,
            ),
        },
        name="mlp_capacity",
    )

    pred = geo_features.skb.apply(model, y=y).skb.set_name("model")

    return {"pred": pred, "scoring": scoring}