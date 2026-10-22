import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin, clone
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

BASIC = [
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
    'dropoff_latitude', 'dx', 'dy', 'distance', 'manhattan',
    'passenger_count', 'year', 'month', 'day', 'weekday', 'hour', 'minute'
]


class CalibratedConvexEnsemble(RegressorMixin, BaseEstimator):
    def __init__(self, xgb, lgb, seed=1971):
        self.xgb = xgb
        self.lgb = lgb
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.RandomState(self.seed)
        order = rng.permutation(len(X))
        calibration = order[:30000]
        candidates = order[30000:]
        held_times = X.iloc[calibration]['pickup_datetime']
        eligible = ~X.iloc[candidates]['pickup_datetime'].isin(held_times).to_numpy()
        fitting = candidates[eligible][:300000]
        self.lgb_columns_ = [c for c in X.columns if c != 'pickup_datetime']
        columns = [BASIC, self.lgb_columns_]
        prototypes = [self.xgb, self.lgb]
        predictions = []
        for model, cols in zip(prototypes, columns):
            fitted = clone(model).fit(X.iloc[fitting][cols], y[fitting])
            predictions.append(fitted.predict(X.iloc[calibration][cols]))
        a, b = predictions
        difference = a - b
        denominator = np.dot(difference, difference)
        weight = (
            np.clip(np.dot(y[calibration] - b, difference) / denominator, 0, 1)
            if denominator > 0 else 0.5
        )
        self.weights_ = np.array([weight, 1 - weight])
        residuals = np.column_stack([a - y[calibration], b - y[calibration]])
        print("Calibration weights:", self.weights_)
        print("Calibration residual correlation:", np.corrcoef(residuals.T))
        combined = weight * a + (1 - weight) * b
        print("Calibration component and ensemble RMSE:",
              np.sqrt(np.mean(residuals ** 2, axis=0)),
              np.sqrt(np.mean((combined - y[calibration]) ** 2)))
        print("Calibration/fitting rows:", len(calibration), len(fitting))
        numeric_cal = X.iloc[calibration][self.lgb_columns_]
        self.calibration_min_ = numeric_cal.min()
        self.calibration_max_ = numeric_cal.max()
        print("Calibration missing fractions:", numeric_cal.isna().mean().to_dict())
        self.models_ = []
        for model, cols in zip(prototypes, columns):
            self.models_.append(clone(model).fit(X[cols], y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        numeric = X[self.lgb_columns_]
        outside = (
            numeric.lt(self.calibration_min_, axis='columns')
            | numeric.gt(self.calibration_max_, axis='columns')
        )
        print("Prediction missing fractions:", numeric.isna().mean().to_dict())
        print("Prediction outside calibration range:", outside.mean().to_dict())
        return (
            self.weights_[0] * self.models_[0].predict(X[BASIC])
            + self.weights_[1] * self.models_[1].predict(numeric)
        )


def build():
    setup = build_evaluation()
    X, y = setup['X'], setup['y']
    dt = X['pickup_datetime'].skb.apply_func(
        pd.to_datetime, utc=True, errors='coerce'
    )
    calendar = X[['passenger_count']].assign(
        year=dt.dt.year, month=dt.dt.month, day=dt.dt.day,
        weekday=dt.dt.dayofweek, hour=dt.dt.hour, minute=dt.dt.minute
    ).astype('float32').skb.set_name('calendar')
    dx = (X['dropoff_longitude'] - X['pickup_longitude']) * 84.4
    dy = (X['dropoff_latitude'] - X['pickup_latitude']) * 111.2
    geography = X[[
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
    ]].assign(
        dx=dx, dy=dy, distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan=dx.abs() + dy.abs()
    ).astype('float32').skb.set_name('geography')
    angle = np.deg2rad(29.0)
    c, s = np.cos(angle), np.sin(angle)
    px = (X['pickup_longitude'] + 74.0) * 84.4
    py = (X['pickup_latitude'] - 40.75) * 111.2
    qx = (X['dropoff_longitude'] + 74.0) * 84.4
    qy = (X['dropoff_latitude'] - 40.75) * 111.2
    route = X[['passenger_count']].assign(
        pickup_grid_x=px * c - py * s,
        pickup_grid_y=px * s + py * c,
        dropoff_grid_x=qx * c - qy * s,
        dropoff_grid_y=qx * s + qy * c,
        grid_manhattan=(dx * c - dy * s).abs() + (dx * s + dy * c).abs(),
        bearing=dy.skb.apply_func(np.arctan2, dx),
        midpoint_x=(px + qx) * 0.5,
        midpoint_y=(py + qy) * 0.5,
        post_tariff=(dt >= pd.Timestamp('2012-09-04', tz='UTC')).fillna(False),
        night=((dt.dt.hour >= 20) | (dt.dt.hour < 6)).fillna(False),
        weekday_rush=(
            (dt.dt.dayofweek < 5) & (dt.dt.hour >= 16) & (dt.dt.hour < 20)
        ).fillna(False)
    ).drop(columns=['passenger_count'])
    for airport, lon, lat in [
        ('jfk', -73.7781, 40.6413),
        ('lga', -73.874, 40.7769),
        ('newark', -74.1745, 40.6895)
    ]:
        for endpoint in ['pickup', 'dropoff']:
            ax = (X[endpoint + '_longitude'] - lon) * 84.4
            ay = (X[endpoint + '_latitude'] - lat) * 111.2
            route = route.assign(**{
                endpoint + '_' + airport + '_distance':
                (ax ** 2 + ay ** 2) ** 0.5
            })
    route = route.astype('float32').skb.set_name('route_regime')
    numeric = geography.skb.concat([calendar, route], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).skb.set_name('finite_features')
    features = numeric.skb.concat(
        [X[['pickup_datetime']]], axis=1
    ).skb.set_name('calibration_timestamp')
    xgb = XGBRegressor(
        objective='reg:squarederror', tree_method='hist', device='cuda',
        n_estimators=600, max_depth=8, learning_rate=0.06,
        min_child_weight=20, subsample=0.85, colsample_bytree=0.9,
        reg_lambda=10, n_jobs=8, random_state=1971
    )
    lgb = LGBMRegressor(
        objective='regression', n_estimators=400, num_leaves=31,
        learning_rate=0.07, min_child_samples=100, reg_lambda=10,
        n_jobs=8, random_state=1971, verbosity=-1, device_type='cpu'
    )
    pred = features.skb.apply(
        CalibratedConvexEnsemble(xgb=xgb, lgb=lgb), y=y
    ).skb.set_name('model')
    result = {'pred': pred, 'scoring': setup['scoring']}
    if 'row_keys' in setup:
        result['row_keys'] = setup['row_keys']
    return result