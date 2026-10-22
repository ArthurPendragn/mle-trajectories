import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor


class TaxiDomainFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)

        p_lat = df["pickup_latitude"].to_numpy(dtype=np.float64)
        p_lon = df["pickup_longitude"].to_numpy(dtype=np.float64)
        d_lat = df["dropoff_latitude"].to_numpy(dtype=np.float64)
        d_lon = df["dropoff_longitude"].to_numpy(dtype=np.float64)

        dlat = np.radians(d_lat - p_lat)
        dlon = np.radians(d_lon - p_lon)
        lat1 = np.radians(p_lat)
        lat2 = np.radians(d_lat)

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arcsin(np.sqrt(a))
        haversine_km = 6371.0 * c

        dlat_km = np.abs(d_lat - p_lat) * 111.0
        dlon_km = np.abs(d_lon - p_lon) * (111.0 * np.cos(np.radians(40.75)))
        manhattan_km = dlat_km + dlon_km

        features = pd.DataFrame(
            {
                "pickup_longitude": p_lon,
                "pickup_latitude": p_lat,
                "dropoff_longitude": d_lon,
                "dropoff_latitude": d_lat,
                "passenger_count": df["passenger_count"].to_numpy(dtype=np.float64),
                "haversine_km": haversine_km,
                "manhattan_km": manhattan_km,
                "abs_lat_diff": np.abs(d_lat - p_lat),
                "abs_lon_diff": np.abs(d_lon - p_lon),
                "center_latitude": (p_lat + d_lat) / 2.0,
                "center_longitude": (p_lon + d_lon) / 2.0,
                "hour": dt.dt.hour.fillna(12).to_numpy(dtype=np.float64),
                "dayofweek": dt.dt.dayofweek.fillna(2).to_numpy(dtype=np.float64),
                "month": dt.dt.month.fillna(6).to_numpy(dtype=np.float64),
                "year": dt.dt.year.fillna(2012).to_numpy(dtype=np.float64),
            },
            index=df.index,
        )

        return features


def build():
    setup = build_evaluation()
    X = setup["X"]
    y = setup["y"]

    features = X.skb.apply(TaxiDomainFeatureExtractor()).skb.set_name("features")

    rf = RandomForestRegressor(
        n_estimators=60,
        max_depth=16,
        max_samples=0.3,
        n_jobs=16,
        random_state=42,
    )
    pred = features.skb.apply(rf, y=y).skb.set_name("model")

    return {
        "pred": pred,
        "scoring": setup["scoring"],
    }