import math
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class ResBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class TabularResNet(nn.Module):

    def __init__(self, in_features, hidden_dim=256, n_blocks=2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).view(-1)

class BlendedEnsembleRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        lgb_weight=0.75,
        n_estimators=400,
        lr_lgb=0.07,
        num_leaves=63,
        hidden_dim=256,
        n_blocks=2,
        epochs=14,
        batch_size=4096,
        lr_nn=0.003,
        weight_decay=0.0001,
        random_state=42,
    ):
        self.lgb_weight = lgb_weight
        self.n_estimators = n_estimators
        self.lr_lgb = lr_lgb
        self.num_leaves = num_leaves
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_nn = lr_nn
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        self.lgb_ = LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=self.n_estimators,
            learning_rate=self.lr_lgb,
            num_leaves=self.num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=8,
            verbosity=-1,
        )
        self.lgb_.fit(X, y)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(X, pd.DataFrame):
            x_arr = X.to_numpy(dtype=np.float32)
        else:
            x_arr = np.asarray(X, dtype=np.float32)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.to_numpy(dtype=np.float32).ravel()
        else:
            y_arr = np.asarray(y, dtype=np.float32).ravel()

        self.mean_ = np.nanmean(x_arr, axis=0, keepdims=True)
        self.std_ = np.nanstd(x_arr, axis=0, keepdims=True)
        self.std_[self.std_ == 0.0] = 1.0

        x_norm = np.nan_to_num((x_arr - self.mean_) / self.std_, nan=0.0, posinf=0.0, neginf=0.0)
        tensor_x = torch.from_numpy(x_norm)
        tensor_y = torch.from_numpy(y_arr)

        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        model = TabularResNet(
            in_features=x_norm.shape[1],
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr_nn, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        self.net_ = model.cpu()
        return self

    def predict(self, X):
        pred_lgb = self.lgb_.predict(X)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(X, pd.DataFrame):
            x_arr = X.to_numpy(dtype=np.float32)
        else:
            x_arr = np.asarray(X, dtype=np.float32)

        x_norm = np.nan_to_num((x_arr - self.mean_) / self.std_, nan=0.0, posinf=0.0, neginf=0.0)
        tensor_x = torch.from_numpy(x_norm)
        loader = DataLoader(tensor_x, batch_size=self.batch_size * 2, shuffle=False)

        self.net_.to(device)
        self.net_.eval()
        preds_list = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(device)
                preds = self.net_(batch_x)
                preds_list.append(preds.cpu().numpy())

        pred_nn = np.concatenate(preds_list, axis=0)
        return self.lgb_weight * pred_lgb + (1.0 - self.lgb_weight) * pred_nn

def build():
    eval_dict = build_evaluation()
    X = eval_dict['X']
    y = eval_dict['y']
    scoring = eval_dict['scoring']
    row_keys = eval_dict.get('row_keys')

    p_lon = X['pickup_longitude'].astype('float64')
    p_lat = X['pickup_latitude'].astype('float64')
    d_lon = X['dropoff_longitude'].astype('float64')
    d_lat = X['dropoff_latitude'].astype('float64')

    dx_km = (d_lon - p_lon) * 84.3
    dy_km = (d_lat - p_lat) * 111.0
    abs_dx_km = dx_km.abs()
    abs_dy_km = dy_km.abs()
    euclidean_km = (dx_km ** 2 + dy_km ** 2).skb.apply_func(np.sqrt)
    manhattan_km = abs_dx_km + abs_dy_km

    theta = math.radians(-29.0)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rot_x = dx_km * cos_theta - dy_km * sin_theta
    rot_y = dx_km * sin_theta + dy_km * cos_theta
    rot_manhattan_km = rot_x.abs() + rot_y.abs()

    grid_dir_x = rot_x / (euclidean_km + 1e-05)
    grid_dir_y = rot_y / (euclidean_km + 1e-05)

    jfk_p_dist = (((p_lat - 40.6413) * 111.0) ** 2 + ((p_lon - -73.7781) * 84.3) ** 2).skb.apply_func(np.sqrt)
    jfk_d_dist = (((d_lat - 40.6413) * 111.0) ** 2 + ((d_lon - -73.7781) * 84.3) ** 2).skb.apply_func(np.sqrt)
    jfk_min_dist = (jfk_p_dist + jfk_d_dist - (jfk_p_dist - jfk_d_dist).abs()) * 0.5

    lga_p_dist = (((p_lat - 40.7769) * 111.0) ** 2 + ((p_lon - -73.874) * 84.3) ** 2).skb.apply_func(np.sqrt)
    lga_d_dist = (((d_lat - 40.7769) * 111.0) ** 2 + ((d_lon - -73.874) * 84.3) ** 2).skb.apply_func(np.sqrt)
    lga_min_dist = (lga_p_dist + lga_d_dist - (lga_p_dist - lga_d_dist).abs()) * 0.5

    ewr_p_dist = (((p_lat - 40.6895) * 111.0) ** 2 + ((p_lon - -74.1745) * 84.3) ** 2).skb.apply_func(np.sqrt)
    ewr_d_dist = (((d_lat - 40.6895) * 111.0) ** 2 + ((d_lon - -74.1745) * 84.3) ** 2).skb.apply_func(np.sqrt)
    ewr_min_dist = (ewr_p_dist + ewr_d_dist - (ewr_p_dist - ewr_d_dist).abs()) * 0.5

    midtown_p_dist = (((p_lat - 40.758) * 111.0) ** 2 + ((p_lon - -73.9855) * 84.3) ** 2).skb.apply_func(np.sqrt)
    midtown_d_dist = (((d_lat - 40.758) * 111.0) ** 2 + ((d_lon - -73.9855) * 84.3) ** 2).skb.apply_func(np.sqrt)

    spatial_features = p_lon.to_frame('pickup_longitude').skb.concat([
        p_lat.to_frame('pickup_latitude'),
        d_lon.to_frame('dropoff_longitude'),
        d_lat.to_frame('dropoff_latitude'),
        abs_dx_km.to_frame('abs_dx_km'),
        abs_dy_km.to_frame('abs_dy_km'),
        euclidean_km.to_frame('euclidean_km'),
        manhattan_km.to_frame('manhattan_km'),
        rot_manhattan_km.to_frame('rot_manhattan_km'),
        grid_dir_x.to_frame('grid_dir_x'),
        grid_dir_y.to_frame('grid_dir_y'),
        jfk_p_dist.to_frame('jfk_p_dist'),
        jfk_d_dist.to_frame('jfk_d_dist'),
        jfk_min_dist.to_frame('jfk_min_dist'),
        lga_p_dist.to_frame('lga_p_dist'),
        lga_d_dist.to_frame('lga_d_dist'),
        lga_min_dist.to_frame('lga_min_dist'),
        ewr_p_dist.to_frame('ewr_p_dist'),
        ewr_d_dist.to_frame('ewr_d_dist'),
        ewr_min_dist.to_frame('ewr_min_dist'),
        midtown_p_dist.to_frame('midtown_p_dist'),
        midtown_d_dist.to_frame('midtown_d_dist'),
    ], axis=1).skb.set_name('spatial_features')

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime)
    year = dt.dt.year.astype('float64')
    month = dt.dt.month.astype('float64')
    day = dt.dt.day.astype('float64')
    hour = dt.dt.hour.astype('float64')
    dayofweek = dt.dt.dayofweek.astype('float64')
    dayofyear = dt.dt.dayofyear.astype('float64')

    is_weekday = (dayofweek < 5).astype('float64')
    is_rush_hour = ((is_weekday == 1.0) & (hour >= 16.0) & (hour < 20.0)).fillna(False).astype('float64')
    is_overnight = ((hour >= 20.0) | (hour < 6.0)).fillna(False).astype('float64')
    is_post_2012_hike = ((year > 2012.0) | ((year == 2012.0) & (month >= 9.0))).fillna(False).astype('float64')
    time_elapsed = year + (dayofyear - 1.0) / 365.25
    passengers = X['passenger_count'].astype('float64')

    hour_sin = (2.0 * math.pi * hour / 24.0).skb.apply_func(np.sin)
    hour_cos = (2.0 * math.pi * hour / 24.0).skb.apply_func(np.cos)
    month_sin = (2.0 * math.pi * month / 12.0).skb.apply_func(np.sin)
    month_cos = (2.0 * math.pi * month / 12.0).skb.apply_func(np.cos)
    dow_sin = (2.0 * math.pi * dayofweek / 7.0).skb.apply_func(np.sin)
    dow_cos = (2.0 * math.pi * dayofweek / 7.0).skb.apply_func(np.cos)

    temporal_features = year.to_frame('year').skb.concat([
        month.to_frame('month'),
        day.to_frame('day'),
        hour.to_frame('hour'),
        dayofweek.to_frame('dayofweek'),
        is_weekday.to_frame('is_weekday'),
        is_rush_hour.to_frame('is_rush_hour'),
        is_overnight.to_frame('is_overnight'),
        is_post_2012_hike.to_frame('is_post_2012_hike'),
        time_elapsed.to_frame('time_elapsed'),
        hour_sin.to_frame('hour_sin'),
        hour_cos.to_frame('hour_cos'),
        month_sin.to_frame('month_sin'),
        month_cos.to_frame('month_cos'),
        dow_sin.to_frame('dow_sin'),
        dow_cos.to_frame('dow_cos'),
        passengers.to_frame('passenger_count'),
    ], axis=1).skb.set_name('temporal_features')

    features = spatial_features.skb.concat([temporal_features], axis=1).skb.set_name('combined_features')

    model = BlendedEnsembleRegressor(
        lgb_weight=0.75,
        n_estimators=400,
        lr_lgb=0.07,
        num_leaves=63,
        hidden_dim=256,
        n_blocks=2,
        epochs=14,
        batch_size=4096,
        lr_nn=0.003,
        weight_decay=0.0001,
        random_state=42,
    )
    pred = features.skb.apply(model, y=y).skb.set_name('blended_ensemble')

    result = {'pred': pred, 'scoring': scoring}
    if row_keys is not None:
        result['row_keys'] = row_keys
    return result