import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor

class ResBlock(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class TabularResNet(nn.Module):

    def __init__(self, in_features, hidden_dim=384, n_blocks=3, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).view(-1)

class BlendedEnsembleRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        n_estimators=1000,
        num_leaves=255,
        lr_lgb=0.035,
        max_depth=10,
        lr_xgb=0.035,
        subsample=0.8,
        colsample_bytree=0.8,
        hidden_dim=384,
        n_blocks=3,
        epochs=18,
        batch_size=4096,
        lr_nn=0.0015,
        weight_decay=0.0001,
        lgb_weight=0.45,
        xgb_weight=0.45,
        nn_weight=0.1,
        min_fare=2.5,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.lr_lgb = lr_lgb
        self.max_depth = max_depth
        self.lr_xgb = lr_xgb
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_nn = lr_nn
        self.weight_decay = weight_decay
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight
        self.nn_weight = nn_weight
        self.min_fare = min_fare
        self.random_state = random_state

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            x_arr = X.to_numpy(dtype=np.float32)
        else:
            x_arr = np.asarray(X, dtype=np.float32)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.to_numpy(dtype=np.float32).ravel()
        else:
            y_arr = np.asarray(y, dtype=np.float32).ravel()

        self.lgb_ = LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=self.n_estimators,
            learning_rate=self.lr_lgb,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            subsample_freq=1,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=8,
            verbosity=-1,
        )
        self.lgb_.fit(X, y_arr)

        xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xgb_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.lr_xgb,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method='hist',
            device=xgb_device,
            random_state=self.random_state,
            n_jobs=8,
        )
        self.xgb_.fit(X, y_arr)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mean_ = np.nanmean(x_arr, axis=0, keepdims=True)
        self.std_ = np.nanstd(x_arr, axis=0, keepdims=True)
        self.std_[self.std_ == 0.0] = 1.0
        x_norm = np.nan_to_num((x_arr - self.mean_) / self.std_, nan=0.0, posinf=0.0, neginf=0.0)

        self.y_mean_ = float(np.nanmean(y_arr))
        self.y_std_ = float(np.nanstd(y_arr))
        if self.y_std_ == 0.0:
            self.y_std_ = 1.0
        y_norm = (y_arr - self.y_mean_) / self.y_std_

        tensor_x = torch.from_numpy(x_norm)
        tensor_y = torch.from_numpy(y_norm.astype(np.float32))
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        model = TabularResNet(
            in_features=x_norm.shape[1],
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            dropout=0.1,
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
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        self.net_ = model.cpu()
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            x_arr = X.to_numpy(dtype=np.float32)
        else:
            x_arr = np.asarray(X, dtype=np.float32)

        pred_lgb = self.lgb_.predict(X)
        pred_xgb = self.xgb_.predict(X)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        pred_nn_norm = np.concatenate(preds_list, axis=0)
        pred_nn = pred_nn_norm * self.y_std_ + self.y_mean_

        blended = self.lgb_weight * pred_lgb + self.xgb_weight * pred_xgb + self.nn_weight * pred_nn
        return np.clip(blended, self.min_fare, None)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']
    row_keys = setup.get('row_keys')

    p_lon = X['pickup_longitude']
    p_lat = X['pickup_latitude']
    d_lon = X['dropoff_longitude']
    d_lat = X['dropoff_latitude']

    dx = (d_lon - p_lon) * 84.14
    dy = (d_lat - p_lat) * 111.0
    abs_dx = dx.abs()
    abs_dy = dy.abs()
    euclidean_dist = (dx ** 2 + dy ** 2) ** 0.5

    rot_x = 0.8746 * dx - 0.4848 * dy
    rot_y = 0.4848 * dx + 0.8746 * dy
    abs_rot_x = rot_x.abs()
    abs_rot_y = rot_y.abs()
    rot_manhattan = abs_rot_x + abs_rot_y
    tortuosity = rot_manhattan / (euclidean_dist + 0.0001)

    dir_rot_x = rot_x / (rot_manhattan + 0.0001)
    dir_rot_y = rot_y / (rot_manhattan + 0.0001)

    p_dx_mid = (p_lon - -73.9855) * 84.14
    p_dy_mid = (p_lat - 40.758) * 111.0
    p_rot_x = 0.8746 * p_dx_mid - 0.4848 * p_dy_mid
    p_rot_y = 0.4848 * p_dx_mid + 0.8746 * p_dy_mid

    d_dx_mid = (d_lon - -73.9855) * 84.14
    d_dy_mid = (d_lat - 40.758) * 111.0
    d_rot_x = 0.8746 * d_dx_mid - 0.4848 * d_dy_mid
    d_rot_y = 0.4848 * d_dx_mid + 0.8746 * d_dy_mid

    mid_rot_x = (p_rot_x + d_rot_x) * 0.5
    mid_rot_y = (p_rot_y + d_rot_y) * 0.5
    midpoint_to_midtown = (mid_rot_x ** 2 + mid_rot_y ** 2) ** 0.5

    in_manhattan_p = ((p_rot_x >= -3.5) & (p_rot_x <= 3.5) & (p_rot_y >= -12.0) & (p_rot_y <= 12.0)).fillna(False).astype('float64')
    in_manhattan_d = ((d_rot_x >= -3.5) & (d_rot_x <= 3.5) & (d_rot_y >= -12.0) & (d_rot_y <= 12.0)).fillna(False).astype('float64')
    is_intra_manhattan = ((in_manhattan_p == 1.0) & (in_manhattan_d == 1.0)).fillna(False).astype('float64')
    is_borough_crossing = (in_manhattan_p - in_manhattan_d).abs()

    jfk_p = (((p_lon - -73.7781) * 84.14) ** 2 + ((p_lat - 40.6413) * 111.0) ** 2) ** 0.5
    jfk_d = (((d_lon - -73.7781) * 84.14) ** 2 + ((d_lat - 40.6413) * 111.0) ** 2) ** 0.5
    jfk_min_dist = (jfk_p + jfk_d - (jfk_p - jfk_d).abs()) * 0.5
    is_jfk = ((jfk_p < 2.5).fillna(False) | (jfk_d < 2.5).fillna(False)).astype('float64')

    lga_p = (((p_lon - -73.874) * 84.14) ** 2 + ((p_lat - 40.7769) * 111.0) ** 2) ** 0.5
    lga_d = (((d_lon - -73.874) * 84.14) ** 2 + ((d_lat - 40.7769) * 111.0) ** 2) ** 0.5
    lga_min_dist = (lga_p + lga_d - (lga_p - lga_d).abs()) * 0.5
    is_lga = ((lga_p < 2.0).fillna(False) | (lga_d < 2.0).fillna(False)).astype('float64')

    ewr_p = (((p_lon - -74.1745) * 84.14) ** 2 + ((p_lat - 40.6895) * 111.0) ** 2) ** 0.5
    ewr_d = (((d_lon - -74.1745) * 84.14) ** 2 + ((d_lat - 40.6895) * 111.0) ** 2) ** 0.5
    ewr_min_dist = (ewr_p + ewr_d - (ewr_p - ewr_d).abs()) * 0.5
    is_ewr = ((ewr_p < 3.0).fillna(False) | (ewr_d < 3.0).fillna(False)).astype('float64')

    uptown_p = (((p_lon - -73.945) * 84.14) ** 2 + ((p_lat - 40.81) * 111.0) ** 2) ** 0.5
    uptown_d = (((d_lon - -73.945) * 84.14) ** 2 + ((d_lat - 40.81) * 111.0) ** 2) ** 0.5
    uptown_min_dist = (uptown_p + uptown_d - (uptown_p - uptown_d).abs()) * 0.5

    bk_p = (((p_lon - -73.9442) * 84.14) ** 2 + ((p_lat - 40.6782) * 111.0) ** 2) ** 0.5
    bk_d = (((d_lon - -73.9442) * 84.14) ** 2 + ((d_lat - 40.6782) * 111.0) ** 2) ** 0.5
    bk_min_dist = (bk_p + bk_d - (bk_p - bk_d).abs()) * 0.5

    qn_p = (((p_lon - -73.7949) * 84.14) ** 2 + ((p_lat - 40.7282) * 111.0) ** 2) ** 0.5
    qn_d = (((d_lon - -73.7949) * 84.14) ** 2 + ((d_lat - 40.7282) * 111.0) ** 2) ** 0.5
    qn_min_dist = (qn_p + qn_d - (qn_p - qn_d).abs()) * 0.5

    bx_p = (((p_lon - -73.8648) * 84.14) ** 2 + ((p_lat - 40.8448) * 111.0) ** 2) ** 0.5
    bx_d = (((d_lon - -73.8648) * 84.14) ** 2 + ((d_lat - 40.8448) * 111.0) ** 2) ** 0.5
    bx_min_dist = (bx_p + bx_d - (bx_p - bx_d).abs()) * 0.5

    is_hudson_crossing = (
        (p_lon < -74.025).fillna(False) & (d_lon >= -74.01).fillna(False) |
        (p_lon >= -74.01).fillna(False) & (d_lon < -74.025).fillna(False)
    ).astype('float64')

    lincoln_p = (((p_lon - -74.0044) * 84.14) ** 2 + ((p_lat - 40.7615) * 111.0) ** 2) ** 0.5
    lincoln_d = (((d_lon - -74.0044) * 84.14) ** 2 + ((d_lat - 40.7615) * 111.0) ** 2) ** 0.5
    lincoln_min_dist = (lincoln_p + lincoln_d - (lincoln_p - lincoln_d).abs()) * 0.5
    lincoln_detour = lincoln_p + lincoln_d - euclidean_dist

    holland_p = (((p_lon - -74.0216) * 84.14) ** 2 + ((p_lat - 40.7277) * 111.0) ** 2) ** 0.5
    holland_d = (((d_lon - -74.0216) * 84.14) ** 2 + ((d_lat - 40.7277) * 111.0) ** 2) ** 0.5
    holland_min_dist = (holland_p + holland_d - (holland_p - holland_d).abs()) * 0.5
    holland_detour = holland_p + holland_d - euclidean_dist

    gwb_p = (((p_lon - -73.9527) * 84.14) ** 2 + ((p_lat - 40.8517) * 111.0) ** 2) ** 0.5
    gwb_d = (((d_lon - -73.9527) * 84.14) ** 2 + ((d_lat - 40.8517) * 111.0) ** 2) ** 0.5
    gwb_min_dist = (gwb_p + gwb_d - (gwb_p - gwb_d).abs()) * 0.5
    gwb_detour = gwb_p + gwb_d - euclidean_dist

    qmt_p = (((p_lon - -73.9645) * 84.14) ** 2 + ((p_lat - 40.7445) * 111.0) ** 2) ** 0.5
    qmt_d = (((d_lon - -73.9645) * 84.14) ** 2 + ((d_lat - 40.7445) * 111.0) ** 2) ** 0.5
    qmt_min_dist = (qmt_p + qmt_d - (qmt_p - qmt_d).abs()) * 0.5
    qmt_detour = qmt_p + qmt_d - euclidean_dist

    rfk_p = (((p_lon - -73.9282) * 84.14) ** 2 + ((p_lat - 40.7798) * 111.0) ** 2) ** 0.5
    rfk_d = (((d_lon - -73.9282) * 84.14) ** 2 + ((d_lat - 40.7798) * 111.0) ** 2) ** 0.5
    rfk_min_dist = (rfk_p + rfk_d - (rfk_p - rfk_d).abs()) * 0.5
    rfk_detour = rfk_p + rfk_d - euclidean_dist

    qbb_p = (((p_lon - -73.9542) * 84.14) ** 2 + ((p_lat - 40.757) * 111.0) ** 2) ** 0.5
    qbb_d = (((d_lon - -73.9542) * 84.14) ** 2 + ((d_lat - 40.757) * 111.0) ** 2) ** 0.5
    qbb_min_dist = (qbb_p + qbb_d - (qbb_p - qbb_d).abs()) * 0.5
    qbb_detour = qbb_p + qbb_d - euclidean_dist

    bbt_p = (((p_lon - -74.0131) * 84.14) ** 2 + ((p_lat - 40.6958) * 111.0) ** 2) ** 0.5
    bbt_d = (((d_lon - -74.0131) * 84.14) ** 2 + ((d_lat - 40.6958) * 111.0) ** 2) ** 0.5
    bbt_min_dist = (bbt_p + bbt_d - (bbt_p - bbt_d).abs()) * 0.5
    bbt_detour = bbt_p + bbt_d - euclidean_dist

    verrazzano_p = (((p_lon - -74.0447) * 84.14) ** 2 + ((p_lat - 40.6066) * 111.0) ** 2) ** 0.5
    verrazzano_d = (((d_lon - -74.0447) * 84.14) ** 2 + ((d_lat - 40.6066) * 111.0) ** 2) ** 0.5
    verrazzano_min_dist = (verrazzano_p + verrazzano_d - (verrazzano_p - verrazzano_d).abs()) * 0.5
    verrazzano_detour = verrazzano_p + verrazzano_d - euclidean_dist

    bk_bridge_p = (((p_lon - -73.995) * 84.14) ** 2 + ((p_lat - 40.706) * 111.0) ** 2) ** 0.5
    bk_bridge_d = (((d_lon - -73.995) * 84.14) ** 2 + ((d_lat - 40.706) * 111.0) ** 2) ** 0.5
    bk_bridge_min_dist = (bk_bridge_p + bk_bridge_d - (bk_bridge_p - bk_bridge_d).abs()) * 0.5
    bk_bridge_detour = bk_bridge_p + bk_bridge_d - euclidean_dist
    toll_diff_brooklyn = bbt_detour - bk_bridge_detour

    is_hudson_tolled = is_hudson_crossing * (
        (lincoln_detour < 1.5).fillna(False) | (holland_detour < 1.5).fillna(False)
    ).astype('float64')

    spatial_features = X[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].assign(
        dx=dx,
        dy=dy,
        abs_dx=abs_dx,
        abs_dy=abs_dy,
        euclidean_dist=euclidean_dist,
        rot_x=rot_x,
        rot_y=rot_y,
        dir_rot_x=dir_rot_x,
        dir_rot_y=dir_rot_y,
        rot_manhattan=rot_manhattan,
        abs_rot_x=abs_rot_x,
        abs_rot_y=abs_rot_y,
        p_rot_x=p_rot_x,
        p_rot_y=p_rot_y,
        d_rot_x=d_rot_x,
        d_rot_y=d_rot_y,
        mid_rot_x=mid_rot_x,
        mid_rot_y=mid_rot_y,
        midpoint_to_midtown=midpoint_to_midtown,
        tortuosity=tortuosity,
        in_manhattan_p=in_manhattan_p,
        in_manhattan_d=in_manhattan_d,
        is_intra_manhattan=is_intra_manhattan,
        is_borough_crossing=is_borough_crossing,
        is_hudson_crossing=is_hudson_crossing,
        jfk_p=jfk_p,
        jfk_d=jfk_d,
        jfk_min_dist=jfk_min_dist,
        is_jfk=is_jfk,
        lga_p=lga_p,
        lga_d=lga_d,
        lga_min_dist=lga_min_dist,
        is_lga=is_lga,
        ewr_p=ewr_p,
        ewr_d=ewr_d,
        ewr_min_dist=ewr_min_dist,
        is_ewr=is_ewr,
        uptown_p=uptown_p,
        uptown_d=uptown_d,
        uptown_min_dist=uptown_min_dist,
        bk_p=bk_p,
        bk_d=bk_d,
        bk_min_dist=bk_min_dist,
        qn_p=qn_p,
        qn_d=qn_d,
        qn_min_dist=qn_min_dist,
        bx_p=bx_p,
        bx_d=bx_d,
        bx_min_dist=bx_min_dist,
        lincoln_min_dist=lincoln_min_dist,
        lincoln_detour=lincoln_detour,
        holland_min_dist=holland_min_dist,
        holland_detour=holland_detour,
        gwb_min_dist=gwb_min_dist,
        gwb_detour=gwb_detour,
        qmt_min_dist=qmt_min_dist,
        qmt_detour=qmt_detour,
        rfk_min_dist=rfk_min_dist,
        rfk_detour=rfk_detour,
        qbb_min_dist=qbb_min_dist,
        qbb_detour=qbb_detour,
        bbt_min_dist=bbt_min_dist,
        bbt_detour=bbt_detour,
        verrazzano_min_dist=verrazzano_min_dist,
        verrazzano_detour=verrazzano_detour,
        bk_bridge_min_dist=bk_bridge_min_dist,
        bk_bridge_detour=bk_bridge_detour,
        toll_diff_brooklyn=toll_diff_brooklyn,
        is_hudson_tolled=is_hudson_tolled,
    ).skb.set_name('spatial_features')

    pickup_dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime)
    hour = pickup_dt.dt.hour.astype('float64')
    minute = pickup_dt.dt.minute.astype('float64')
    month = pickup_dt.dt.month.astype('float64')
    dayofweek = pickup_dt.dt.dayofweek.astype('float64')
    year = pickup_dt.dt.year.astype('float64')
    dayofyear = pickup_dt.dt.dayofyear.astype('float64')
    time_elapsed = year + (dayofyear - 1.0 + hour / 24.0 + minute / 1440.0) / 365.25

    minute_of_day = hour * 60.0 + minute
    passenger_count = X['passenger_count'].astype('float64')
    is_weekday = (dayofweek <= 4.0).fillna(False)
    is_weekend = (dayofweek >= 5.0).fillna(False).astype('float64')
    is_morning_rush = (is_weekday & (hour >= 7.0).fillna(False) & (hour < 10.0).fillna(False)).astype('float64')
    is_rush_hour = (is_weekday & (hour >= 16.0).fillna(False) & (hour < 20.0).fillna(False)).astype('float64')
    is_overnight = ((hour >= 20.0).fillna(False) | (hour < 6.0).fillna(False)).astype('float64')

    hour_sin = (2.0 * np.pi * hour / 24.0).skb.apply_func(np.sin)
    hour_cos = (2.0 * np.pi * hour / 24.0).skb.apply_func(np.cos)
    mod_sin = (2.0 * np.pi * minute_of_day / 1440.0).skb.apply_func(np.sin)
    mod_cos = (2.0 * np.pi * minute_of_day / 1440.0).skb.apply_func(np.cos)
    month_sin = (2.0 * np.pi * month / 12.0).skb.apply_func(np.sin)
    month_cos = (2.0 * np.pi * month / 12.0).skb.apply_func(np.cos)
    dow_sin = (2.0 * np.pi * dayofweek / 7.0).skb.apply_func(np.sin)
    dow_cos = (2.0 * np.pi * dayofweek / 7.0).skb.apply_func(np.cos)

    temporal_features = spatial_features.assign(
        year=year,
        month=month,
        dayofweek=dayofweek,
        hour=hour,
        minute=minute,
        minute_of_day=minute_of_day,
        time_elapsed=time_elapsed,
        passenger_count=passenger_count,
        is_morning_rush=is_morning_rush,
        is_rush_hour=is_rush_hour,
        is_overnight=is_overnight,
        is_weekend=is_weekend,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        mod_sin=mod_sin,
        mod_cos=mod_cos,
        month_sin=month_sin,
        month_cos=month_cos,
        dow_sin=dow_sin,
        dow_cos=dow_cos,
    ).skb.set_name('temporal_features')

    is_post_2009_mta = (X['pickup_datetime'] >= '2009-11-01').fillna(False).astype('float64')
    is_post_2012_hike = (X['pickup_datetime'] >= '2012-09-04').fillna(False).astype('float64')

    dist_x_post2012 = rot_manhattan * is_post_2012_hike
    dist_x_intra_manhattan = rot_manhattan * is_intra_manhattan
    dist_x_intra_manhattan_post2012 = dist_x_intra_manhattan * is_post_2012_hike
    dist_x_rush_hour = rot_manhattan * is_rush_hour
    dist_x_morning_rush = rot_manhattan * is_morning_rush
    abs_rot_x_intra = abs_rot_x * is_intra_manhattan
    abs_rot_y_intra = abs_rot_y * is_intra_manhattan

    is_jfk_flat = (is_jfk * ((in_manhattan_p == 1.0) | (in_manhattan_d == 1.0)).fillna(False)).astype('float64')
    dist_x_lga = rot_manhattan * is_lga
    dist_x_jfk_metered = rot_manhattan * (is_jfk * (1.0 - is_jfk_flat))
    toll_diff_east_river = qmt_detour - qbb_detour

    est_meter_fare = (
        2.5 +
        0.5 * is_post_2009_mta +
        (1.24275 + 0.31068 * is_post_2012_hike) * rot_manhattan +
        1.0 * is_rush_hour +
        0.5 * is_overnight
    )
    est_jfk_flat_fare = is_jfk_flat * (45.0 + 7.0 * is_post_2012_hike + 0.5 * is_post_2009_mta + 4.5 * is_rush_hour)
    est_ewr_fare = is_ewr * (est_meter_fare + 15.0 + 2.5 * is_post_2012_hike + 11.0)

    statutory_baseline = (
        est_jfk_flat_fare +
        (1.0 - is_jfk_flat) * (is_ewr * (est_meter_fare + 15.0 + 2.5 * is_post_2012_hike + 11.0) + (1.0 - is_ewr) * est_meter_fare)
    )

    sqrt_dist = rot_manhattan ** 0.5
    log_dist = rot_manhattan.skb.apply_func(np.log1p)

    tariff_features = temporal_features.assign(
        is_post_2009_mta=is_post_2009_mta,
        is_post_2012_hike=is_post_2012_hike,
        dist_x_post2012=dist_x_post2012,
        dist_x_intra_manhattan=dist_x_intra_manhattan,
        dist_x_intra_manhattan_post2012=dist_x_intra_manhattan_post2012,
        dist_x_rush_hour=dist_x_rush_hour,
        dist_x_morning_rush=dist_x_morning_rush,
        abs_rot_x_intra=abs_rot_x_intra,
        abs_rot_y_intra=abs_rot_y_intra,
        dist_x_lga=dist_x_lga,
        dist_x_jfk_metered=dist_x_jfk_metered,
        is_jfk_flat=is_jfk_flat,
        toll_diff_east_river=toll_diff_east_river,
        est_meter_fare=est_meter_fare,
        est_jfk_flat_fare=est_jfk_flat_fare,
        est_ewr_fare=est_ewr_fare,
        statutory_baseline=statutory_baseline,
        sqrt_dist=sqrt_dist,
        log_dist=log_dist,
    ).skb.set_name('tariff_features')

    model = BlendedEnsembleRegressor(
        n_estimators=1000,
        num_leaves=255,
        lr_lgb=0.035,
        max_depth=10,
        lr_xgb=0.035,
        subsample=0.8,
        colsample_bytree=0.8,
        hidden_dim=384,
        n_blocks=3,
        epochs=18,
        batch_size=4096,
        lr_nn=0.0015,
        weight_decay=0.0001,
        lgb_weight=0.45,
        xgb_weight=0.45,
        nn_weight=0.1,
        min_fare=2.5,
        random_state=42,
    )
    pred = tariff_features.skb.apply(model, y=y).skb.set_name('model')

    result = {'pred': pred, 'scoring': scoring}
    if row_keys is not None:
        result['row_keys'] = row_keys
    return result