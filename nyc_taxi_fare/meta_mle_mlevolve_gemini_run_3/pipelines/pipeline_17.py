import math
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from scipy.optimize import nnls
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class ResBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class TabularResNet(nn.Module):

    def __init__(self, in_features, hidden_dim=384, n_blocks=3):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).view(-1)

class BlendedEnsembleRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, n_estimators=800, lr_lgb=0.04, num_leaves=127, lr_xgb=0.04, max_depth=10, min_child_weight=20, hidden_dim=384, n_blocks=3, epochs=22, batch_size=4096, lr_nn=0.003, weight_decay=0.0001, min_fare=2.5, val_size=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.lr_lgb = lr_lgb
        self.num_leaves = num_leaves
        self.lr_xgb = lr_xgb
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_nn = lr_nn
        self.weight_decay = weight_decay
        self.min_fare = min_fare
        self.val_size = val_size
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

        x_train, x_val, y_train, y_val = train_test_split(
            x_arr, y_arr, test_size=self.val_size, random_state=self.random_state
        )

        self.lgb_ = LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=self.n_estimators,
            learning_rate=self.lr_lgb,
            num_leaves=self.num_leaves,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=8,
            verbosity=-1
        )
        self.lgb_.fit(x_train, y_train)

        xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xgb_ = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.lr_xgb,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device=xgb_device,
            random_state=self.random_state,
            n_jobs=8
        )
        self.xgb_.fit(x_train, y_train)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mean_ = np.nanmean(x_train, axis=0, keepdims=True)
        self.std_ = np.nanstd(x_train, axis=0, keepdims=True)
        self.std_[self.std_ == 0.0] = 1.0

        self.y_mean_ = float(np.mean(y_train))
        self.y_std_ = float(np.std(y_train))
        if self.y_std_ == 0.0:
            self.y_std_ = 1.0

        x_train_norm = np.nan_to_num((x_train - self.mean_) / self.std_, nan=0.0, posinf=0.0, neginf=0.0)
        y_train_norm = (y_train - self.y_mean_) / self.y_std_

        tensor_x = torch.from_numpy(x_train_norm)
        tensor_y = torch.from_numpy(y_train_norm)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        model = TabularResNet(in_features=x_train_norm.shape[1], hidden_dim=self.hidden_dim, n_blocks=self.n_blocks).to(device)
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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        self.net_ = model.cpu()

        pred_val_lgb = self.lgb_.predict(x_val)
        pred_val_xgb = self.xgb_.predict(x_val)

        x_val_norm = np.nan_to_num((x_val - self.mean_) / self.std_, nan=0.0, posinf=0.0, neginf=0.0)
        tensor_val_x = torch.from_numpy(x_val_norm)
        val_loader = DataLoader(tensor_val_x, batch_size=self.batch_size * 2, shuffle=False)
        self.net_.to(device)
        self.net_.eval()
        nn_preds = []
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                p = self.net_(batch_x)
                nn_preds.append(p.cpu().numpy())
        pred_val_nn = np.concatenate(nn_preds, axis=0) * self.y_std_ + self.y_mean_

        val_matrix = np.column_stack([pred_val_lgb, pred_val_xgb, pred_val_nn])
        weights, _ = nnls(val_matrix, y_val)
        sum_w = np.sum(weights)
        if sum_w > 0:
            self.weights_ = weights / sum_w
        else:
            self.weights_ = np.array([0.45, 0.45, 0.10], dtype=np.float32)

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            x_arr = X.to_numpy(dtype=np.float32)
        else:
            x_arr = np.asarray(X, dtype=np.float32)

        pred_lgb = self.lgb_.predict(x_arr)
        pred_xgb = self.xgb_.predict(x_arr)

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
        pred_nn = np.concatenate(preds_list, axis=0) * self.y_std_ + self.y_mean_

        blended = (
            self.weights_[0] * pred_lgb
            + self.weights_[1] * pred_xgb
            + self.weights_[2] * pred_nn
        )
        return np.clip(blended, self.min_fare, None)

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
    abs_rot_x = rot_x.abs()
    abs_rot_y = rot_y.abs()
    rot_manhattan_km = abs_rot_x + abs_rot_y

    grid_dir_x = rot_x / (euclidean_km + 1e-05)
    grid_dir_y = rot_y / (euclidean_km + 1e-05)

    p_dx_mid = (p_lon - -73.9855) * 84.3
    p_dy_mid = (p_lat - 40.758) * 111.0
    p_rot_x = p_dx_mid * cos_theta - p_dy_mid * sin_theta
    p_rot_y = p_dx_mid * sin_theta + p_dy_mid * cos_theta

    d_dx_mid = (d_lon - -73.9855) * 84.3
    d_dy_mid = (d_lat - 40.758) * 111.0
    d_rot_x = d_dx_mid * cos_theta - d_dy_mid * sin_theta
    d_rot_y = d_dx_mid * sin_theta + d_dy_mid * cos_theta

    midpoint_rot_x = (p_rot_x + d_rot_x) * 0.5
    midpoint_rot_y = (p_rot_y + d_rot_y) * 0.5

    in_manhattan_p = ((p_rot_x >= -3.5) & (p_rot_x <= 3.5) & (p_rot_y >= -12.0) & (p_rot_y <= 12.0)).fillna(False).astype('float64')
    in_manhattan_d = ((d_rot_x >= -3.5) & (d_rot_x <= 3.5) & (d_rot_y >= -12.0) & (d_rot_y <= 12.0)).fillna(False).astype('float64')
    is_intra_manhattan = ((in_manhattan_p == 1.0) & (in_manhattan_d == 1.0)).fillna(False).astype('float64')
    is_borough_crossing = (in_manhattan_p - in_manhattan_d).abs()

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

    downtown_p_dist = (((p_lat - 40.7128) * 111.0) ** 2 + ((p_lon - -74.006) * 84.3) ** 2).skb.apply_func(np.sqrt)
    downtown_d_dist = (((d_lat - 40.7128) * 111.0) ** 2 + ((d_lon - -74.006) * 84.3) ** 2).skb.apply_func(np.sqrt)
    downtown_min_dist = (downtown_p_dist + downtown_d_dist - (downtown_p_dist - downtown_d_dist).abs()) * 0.5

    uptown_p_dist = (((p_lat - 40.81) * 111.0) ** 2 + ((p_lon - -73.945) * 84.3) ** 2).skb.apply_func(np.sqrt)
    uptown_d_dist = (((d_lat - 40.81) * 111.0) ** 2 + ((d_lon - -73.945) * 84.3) ** 2).skb.apply_func(np.sqrt)
    uptown_min_dist = (uptown_p_dist + uptown_d_dist - (uptown_p_dist - uptown_d_dist).abs()) * 0.5

    qmt_p_dist = (((p_lat - 40.7445) * 111.0) ** 2 + ((p_lon - -73.9645) * 84.3) ** 2).skb.apply_func(np.sqrt)
    qmt_d_dist = (((d_lat - 40.7445) * 111.0) ** 2 + ((d_lon - -73.9645) * 84.3) ** 2).skb.apply_func(np.sqrt)
    qmt_min_dist = (qmt_p_dist + qmt_d_dist - (qmt_p_dist - qmt_d_dist).abs()) * 0.5
    qmt_detour = qmt_p_dist + qmt_d_dist - euclidean_km

    rfk_p_dist = (((p_lat - 40.7798) * 111.0) ** 2 + ((p_lon - -73.9282) * 84.3) ** 2).skb.apply_func(np.sqrt)
    rfk_d_dist = (((d_lat - 40.7798) * 111.0) ** 2 + ((d_lon - -73.9282) * 84.3) ** 2).skb.apply_func(np.sqrt)
    rfk_min_dist = (rfk_p_dist + rfk_d_dist - (rfk_p_dist - rfk_d_dist).abs()) * 0.5
    rfk_detour = rfk_p_dist + rfk_d_dist - euclidean_km

    bbt_p_dist = (((p_lat - 40.6958) * 111.0) ** 2 + ((p_lon - -74.0131) * 84.3) ** 2).skb.apply_func(np.sqrt)
    bbt_d_dist = (((d_lat - 40.6958) * 111.0) ** 2 + ((d_lon - -74.0131) * 84.3) ** 2).skb.apply_func(np.sqrt)
    bbt_min_dist = (bbt_p_dist + bbt_d_dist - (bbt_p_dist - bbt_d_dist).abs()) * 0.5
    bbt_detour = bbt_p_dist + bbt_d_dist - euclidean_km

    gwb_p_dist = (((p_lat - 40.8517) * 111.0) ** 2 + ((p_lon - -73.9527) * 84.3) ** 2).skb.apply_func(np.sqrt)
    gwb_d_dist = (((d_lat - 40.8517) * 111.0) ** 2 + ((d_lon - -73.9527) * 84.3) ** 2).skb.apply_func(np.sqrt)
    gwb_min_dist = (gwb_p_dist + gwb_d_dist - (gwb_p_dist - gwb_d_dist).abs()) * 0.5
    gwb_detour = gwb_p_dist + gwb_d_dist - euclidean_km

    verrazzano_p_dist = (((p_lat - 40.6066) * 111.0) ** 2 + ((p_lon - -74.0447) * 84.3) ** 2).skb.apply_func(np.sqrt)
    verrazzano_d_dist = (((d_lat - 40.6066) * 111.0) ** 2 + ((d_lon - -74.0447) * 84.3) ** 2).skb.apply_func(np.sqrt)
    verrazzano_min_dist = (verrazzano_p_dist + verrazzano_d_dist - (verrazzano_p_dist - verrazzano_d_dist).abs()) * 0.5
    verrazzano_detour = verrazzano_p_dist + verrazzano_d_dist - euclidean_km

    lincoln_p_dist = (((p_lat - 40.7615) * 111.0) ** 2 + ((p_lon - -74.0044) * 84.3) ** 2).skb.apply_func(np.sqrt)
    lincoln_d_dist = (((d_lat - 40.7615) * 111.0) ** 2 + ((d_lon - -74.0044) * 84.3) ** 2).skb.apply_func(np.sqrt)
    lincoln_min_dist = (lincoln_p_dist + lincoln_d_dist - (lincoln_p_dist - lincoln_d_dist).abs()) * 0.5
    lincoln_detour = lincoln_p_dist + lincoln_d_dist - euclidean_km

    holland_p_dist = (((p_lat - 40.7277) * 111.0) ** 2 + ((p_lon - -74.0216) * 84.3) ** 2).skb.apply_func(np.sqrt)
    holland_d_dist = (((d_lat - 40.7277) * 111.0) ** 2 + ((d_lon - -74.0216) * 84.3) ** 2).skb.apply_func(np.sqrt)
    holland_min_dist = (holland_p_dist + holland_d_dist - (holland_p_dist - holland_d_dist).abs()) * 0.5
    holland_detour = holland_p_dist + holland_d_dist - euclidean_km

    qbb_p_dist = (((p_lat - 40.757) * 111.0) ** 2 + ((p_lon - -73.9542) * 84.3) ** 2).skb.apply_func(np.sqrt)
    qbb_d_dist = (((d_lat - 40.757) * 111.0) ** 2 + ((d_lon - -73.9542) * 84.3) ** 2).skb.apply_func(np.sqrt)
    qbb_min_dist = (qbb_p_dist + qbb_d_dist - (qbb_p_dist - qbb_d_dist).abs()) * 0.5
    qbb_detour = qbb_p_dist + qbb_d_dist - euclidean_km

    bk_p_dist = (((p_lat - 40.6782) * 111.0) ** 2 + ((p_lon - -73.9442) * 84.3) ** 2).skb.apply_func(np.sqrt)
    bk_d_dist = (((d_lat - 40.6782) * 111.0) ** 2 + ((d_lon - -73.9442) * 84.3) ** 2).skb.apply_func(np.sqrt)
    bk_min_dist = (bk_p_dist + bk_d_dist - (bk_p_dist - bk_d_dist).abs()) * 0.5

    qn_p_dist = (((p_lat - 40.7282) * 111.0) ** 2 + ((p_lon - -73.7949) * 84.3) ** 2).skb.apply_func(np.sqrt)
    qn_d_dist = (((d_lat - 40.7282) * 111.0) ** 2 + ((d_lon - -73.7949) * 84.3) ** 2).skb.apply_func(np.sqrt)
    qn_min_dist = (qn_p_dist + qn_d_dist - (qn_p_dist - qn_d_dist).abs()) * 0.5

    bx_p_dist = (((p_lat - 40.8448) * 111.0) ** 2 + ((p_lon - -73.8648) * 84.3) ** 2).skb.apply_func(np.sqrt)
    bx_d_dist = (((d_lat - 40.8448) * 111.0) ** 2 + ((d_lon - -73.8648) * 84.3) ** 2).skb.apply_func(np.sqrt)
    bx_min_dist = (bx_p_dist + bx_d_dist - (bx_p_dist - bx_d_dist).abs()) * 0.5

    spatial_features = p_lon.to_frame('pickup_longitude').skb.concat([
        p_lat.to_frame('pickup_latitude'),
        d_lon.to_frame('dropoff_longitude'),
        d_lat.to_frame('dropoff_latitude'),
        dx_km.to_frame('dx_km'),
        dy_km.to_frame('dy_km'),
        rot_x.to_frame('rot_x'),
        rot_y.to_frame('rot_y'),
        p_rot_x.to_frame('p_rot_x'),
        p_rot_y.to_frame('p_rot_y'),
        d_rot_x.to_frame('d_rot_x'),
        d_rot_y.to_frame('d_rot_y'),
        midpoint_rot_x.to_frame('midpoint_rot_x'),
        midpoint_rot_y.to_frame('midpoint_rot_y'),
        in_manhattan_p.to_frame('in_manhattan_p'),
        in_manhattan_d.to_frame('in_manhattan_d'),
        is_intra_manhattan.to_frame('is_intra_manhattan'),
        is_borough_crossing.to_frame('is_borough_crossing'),
        abs_dx_km.to_frame('abs_dx_km'),
        abs_dy_km.to_frame('abs_dy_km'),
        euclidean_km.to_frame('euclidean_km'),
        manhattan_km.to_frame('manhattan_km'),
        rot_manhattan_km.to_frame('rot_manhattan_km'),
        abs_rot_x.to_frame('abs_rot_x'),
        abs_rot_y.to_frame('abs_rot_y'),
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
        downtown_p_dist.to_frame('downtown_p_dist'),
        downtown_d_dist.to_frame('downtown_d_dist'),
        downtown_min_dist.to_frame('downtown_min_dist'),
        uptown_p_dist.to_frame('uptown_p_dist'),
        uptown_d_dist.to_frame('uptown_d_dist'),
        uptown_min_dist.to_frame('uptown_min_dist'),
        qmt_min_dist.to_frame('qmt_min_dist'),
        qmt_detour.to_frame('qmt_detour'),
        rfk_min_dist.to_frame('rfk_min_dist'),
        rfk_detour.to_frame('rfk_detour'),
        bbt_min_dist.to_frame('bbt_min_dist'),
        bbt_detour.to_frame('bbt_detour'),
        gwb_min_dist.to_frame('gwb_min_dist'),
        gwb_detour.to_frame('gwb_detour'),
        verrazzano_min_dist.to_frame('verrazzano_min_dist'),
        verrazzano_detour.to_frame('verrazzano_detour'),
        lincoln_min_dist.to_frame('lincoln_min_dist'),
        lincoln_detour.to_frame('lincoln_detour'),
        holland_min_dist.to_frame('holland_min_dist'),
        holland_detour.to_frame('holland_detour'),
        qbb_min_dist.to_frame('qbb_min_dist'),
        qbb_detour.to_frame('qbb_detour'),
        bk_p_dist.to_frame('bk_p_dist'),
        bk_d_dist.to_frame('bk_d_dist'),
        bk_min_dist.to_frame('bk_min_dist'),
        qn_p_dist.to_frame('qn_p_dist'),
        qn_d_dist.to_frame('qn_d_dist'),
        qn_min_dist.to_frame('qn_min_dist'),
        bx_p_dist.to_frame('bx_p_dist'),
        bx_d_dist.to_frame('bx_d_dist'),
        bx_min_dist.to_frame('bx_min_dist')
    ], axis=1).skb.set_name('spatial_features')

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime)
    year = dt.dt.year.astype('float64')
    month = dt.dt.month.astype('float64')
    day = dt.dt.day.astype('float64')
    hour = dt.dt.hour.astype('float64')
    minute = dt.dt.minute.astype('float64')
    dayofweek = dt.dt.dayofweek.astype('float64')
    dayofyear = dt.dt.dayofyear.astype('float64')

    minute_of_day = hour * 60.0 + minute
    hour_of_week = dayofweek * 24.0 + hour
    is_weekday = (dayofweek < 5.0).astype('float64')
    is_weekend = (dayofweek >= 5.0).astype('float64')

    is_morning_rush = ((is_weekday == 1.0) & (hour >= 7.0) & (hour < 10.0)).fillna(False).astype('float64')
    is_midday = ((is_weekday == 1.0) & (hour >= 10.0) & (hour < 16.0)).fillna(False).astype('float64')
    is_rush_hour = ((is_weekday == 1.0) & (hour >= 16.0) & (hour < 20.0)).fillna(False).astype('float64')
    is_overnight = ((hour >= 20.0) | (hour < 6.0)).fillna(False).astype('float64')

    is_post_2012_hike = ((year > 2012.0) | (year == 2012.0) & (month >= 9.0)).fillna(False).astype('float64')
    is_mta_surcharge = ((year > 2009.0) | (year == 2009.0) & (month >= 11.0)).fillna(False).astype('float64')
    time_elapsed = year + (dayofyear - 1.0) / 365.25
    passengers = X['passenger_count'].astype('float64')

    hour_sin = (2.0 * math.pi * hour / 24.0).skb.apply_func(np.sin)
    hour_cos = (2.0 * math.pi * hour / 24.0).skb.apply_func(np.cos)
    mod_sin = (2.0 * math.pi * minute_of_day / 1440.0).skb.apply_func(np.sin)
    mod_cos = (2.0 * math.pi * minute_of_day / 1440.0).skb.apply_func(np.cos)
    month_sin = (2.0 * math.pi * month / 12.0).skb.apply_func(np.sin)
    month_cos = (2.0 * math.pi * month / 12.0).skb.apply_func(np.cos)
    dow_sin = (2.0 * math.pi * dayofweek / 7.0).skb.apply_func(np.sin)
    dow_cos = (2.0 * math.pi * dayofweek / 7.0).skb.apply_func(np.cos)
    how_sin = (2.0 * math.pi * hour_of_week / 168.0).skb.apply_func(np.sin)
    how_cos = (2.0 * math.pi * hour_of_week / 168.0).skb.apply_func(np.cos)

    temporal_features = year.to_frame('year').skb.concat([
        month.to_frame('month'),
        day.to_frame('day'),
        hour.to_frame('hour'),
        dayofweek.to_frame('dayofweek'),
        hour_of_week.to_frame('hour_of_week'),
        minute_of_day.to_frame('minute_of_day'),
        is_weekday.to_frame('is_weekday'),
        is_weekend.to_frame('is_weekend'),
        is_morning_rush.to_frame('is_morning_rush'),
        is_midday.to_frame('is_midday'),
        is_rush_hour.to_frame('is_rush_hour'),
        is_overnight.to_frame('is_overnight'),
        is_post_2012_hike.to_frame('is_post_2012_hike'),
        is_mta_surcharge.to_frame('is_mta_surcharge'),
        time_elapsed.to_frame('time_elapsed'),
        hour_sin.to_frame('hour_sin'),
        hour_cos.to_frame('hour_cos'),
        mod_sin.to_frame('mod_sin'),
        mod_cos.to_frame('mod_cos'),
        month_sin.to_frame('month_sin'),
        month_cos.to_frame('month_cos'),
        dow_sin.to_frame('dow_sin'),
        dow_cos.to_frame('dow_cos'),
        how_sin.to_frame('how_sin'),
        how_cos.to_frame('how_cos'),
        passengers.to_frame('passenger_count')
    ], axis=1).skb.set_name('temporal_features')

    manhattan_p_dist_raw = (midtown_p_dist + downtown_p_dist - (midtown_p_dist - downtown_p_dist).abs()) * 0.5
    manhattan_p_dist = (manhattan_p_dist_raw + uptown_p_dist - (manhattan_p_dist_raw - uptown_p_dist).abs()) * 0.5
    manhattan_d_dist_raw = (midtown_d_dist + downtown_d_dist - (midtown_d_dist - downtown_d_dist).abs()) * 0.5
    manhattan_d_dist = (manhattan_d_dist_raw + uptown_d_dist - (manhattan_d_dist_raw - uptown_d_dist).abs()) * 0.5

    est_meter_fare = 2.5 + 0.5 * is_mta_surcharge + (1.24275 + 0.31068 * is_post_2012_hike) * rot_manhattan_km + 1.0 * is_rush_hour + 0.5 * is_overnight
    is_jfk_flat_p = (jfk_p_dist < 3.5) & (manhattan_d_dist < 7.0)
    is_jfk_flat_d = (jfk_d_dist < 3.5) & (manhattan_p_dist < 7.0)
    is_jfk_flat_route = (is_jfk_flat_p | is_jfk_flat_d).fillna(False).astype('float64')
    est_jfk_flat_fare = is_jfk_flat_route * (45.0 + 7.0 * is_post_2012_hike + 0.5 * is_mta_surcharge + 4.5 * is_rush_hour * is_post_2012_hike)

    is_ewr_route = (ewr_min_dist < 3.5).fillna(False).astype('float64')
    est_ewr_fare = is_ewr_route * (est_meter_fare + 15.0 + 2.5 * is_post_2012_hike + 11.0)
    is_lga_route = (lga_min_dist < 3.0).fillna(False).astype('float64')

    toll_diff_east_river = qmt_detour - qbb_detour
    tortuosity = rot_manhattan_km / (euclidean_km + 0.0001)

    midpoint_lat = (p_lat + d_lat) * 0.5
    midpoint_lon = (p_lon + d_lon) * 0.5
    midpoint_to_midtown = (((midpoint_lat - 40.758) * 111.0) ** 2 + ((midpoint_lon - -73.9855) * 84.3) ** 2).skb.apply_func(np.sqrt)

    manhattan_morning_rush_km = rot_manhattan_km * is_intra_manhattan * is_morning_rush
    manhattan_midday_km = rot_manhattan_km * is_intra_manhattan * is_midday
    manhattan_evening_rush_km = rot_manhattan_km * is_intra_manhattan * is_rush_hour
    manhattan_night_km = rot_manhattan_km * is_intra_manhattan * is_overnight

    tariff_features = est_meter_fare.to_frame('est_meter_fare').skb.concat([
        is_jfk_flat_route.to_frame('is_jfk_flat_route'),
        est_jfk_flat_fare.to_frame('est_jfk_flat_fare'),
        is_ewr_route.to_frame('is_ewr_route'),
        est_ewr_fare.to_frame('est_ewr_fare'),
        is_lga_route.to_frame('is_lga_route'),
        toll_diff_east_river.to_frame('toll_diff_east_river'),
        tortuosity.to_frame('tortuosity'),
        midpoint_to_midtown.to_frame('midpoint_to_midtown'),
        manhattan_morning_rush_km.to_frame('manhattan_morning_rush_km'),
        manhattan_midday_km.to_frame('manhattan_midday_km'),
        manhattan_evening_rush_km.to_frame('manhattan_evening_rush_km'),
        manhattan_night_km.to_frame('manhattan_night_km')
    ], axis=1).skb.set_name('tariff_features')

    features = spatial_features.skb.concat([temporal_features, tariff_features], axis=1).skb.set_name('combined_features')

    model = BlendedEnsembleRegressor(
        n_estimators=800,
        lr_lgb=0.04,
        num_leaves=127,
        lr_xgb=0.04,
        max_depth=10,
        min_child_weight=20,
        hidden_dim=384,
        n_blocks=3,
        epochs=22,
        batch_size=4096,
        lr_nn=0.003,
        weight_decay=0.0001,
        min_fare=2.5,
        val_size=0.1,
        random_state=42
    )

    pred = features.skb.apply(model, y=y).skb.set_name('blended_ensemble')

    result = {'pred': pred, 'scoring': scoring}
    if row_keys is not None:
        result['row_keys'] = row_keys
    return result