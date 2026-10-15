import math
import numpy as np
import pandas as pd
import skrub
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

class PyTorchMLPRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, hidden_dim=256, n_blocks=2, epochs=14, batch_size=4096, lr=0.003, weight_decay=0.0001, random_state=42):
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
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
        model = TabularResNet(in_features=x_norm.shape[1], hidden_dim=self.hidden_dim, n_blocks=self.n_blocks).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(self.epochs):
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
        return np.concatenate(preds_list, axis=0)

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
    d_lon_diff = d_lon - p_lon
    d_lat_diff = d_lat - p_lat
    abs_d_lon = d_lon_diff.abs()
    abs_d_lat = d_lat_diff.abs()
    euclidean_dist = (d_lon_diff ** 2 + d_lat_diff ** 2) ** 0.5
    manhattan_dist = abs_d_lon + abs_d_lat
    jfk_p_dist = ((p_lon - -73.7781) ** 2 + (p_lat - 40.6413) ** 2) ** 0.5
    jfk_d_dist = ((d_lon - -73.7781) ** 2 + (d_lat - 40.6413) ** 2) ** 0.5
    lga_p_dist = ((p_lon - -73.874) ** 2 + (p_lat - 40.7769) ** 2) ** 0.5
    lga_d_dist = ((d_lon - -73.874) ** 2 + (d_lat - 40.7769) ** 2) ** 0.5
    ewr_p_dist = ((p_lon - -74.1745) ** 2 + (p_lat - 40.6895) ** 2) ** 0.5
    ewr_d_dist = ((d_lon - -74.1745) ** 2 + (d_lat - 40.6895) ** 2) ** 0.5
    time_str = X['pickup_datetime'].astype('string')
    year = time_str.str.slice(0, 4).astype('float64')
    month = time_str.str.slice(5, 7).astype('float64')
    hour = time_str.str.slice(11, 13).astype('float64')
    passengers = X['passenger_count'].astype('float64')
    hour_sin = (2.0 * math.pi * hour / 24.0).skb.apply_func(np.sin)
    hour_cos = (2.0 * math.pi * hour / 24.0).skb.apply_func(np.cos)
    month_sin = (2.0 * math.pi * month / 12.0).skb.apply_func(np.sin)
    month_cos = (2.0 * math.pi * month / 12.0).skb.apply_func(np.cos)
    spatial_features = p_lon.to_frame('pickup_longitude').skb.concat([p_lat.to_frame('pickup_latitude'), d_lon.to_frame('dropoff_longitude'), d_lat.to_frame('dropoff_latitude'), abs_d_lon.to_frame('abs_d_lon'), abs_d_lat.to_frame('abs_d_lat'), euclidean_dist.to_frame('euclidean_dist'), manhattan_dist.to_frame('manhattan_dist'), jfk_p_dist.to_frame('jfk_p_dist'), jfk_d_dist.to_frame('jfk_d_dist'), lga_p_dist.to_frame('lga_p_dist'), lga_d_dist.to_frame('lga_d_dist'), ewr_p_dist.to_frame('ewr_p_dist'), ewr_d_dist.to_frame('ewr_d_dist')], axis=1).skb.set_name('spatial_features')
    temporal_features = year.to_frame('year').skb.concat([month.to_frame('month'), hour.to_frame('hour'), hour_sin.to_frame('hour_sin'), hour_cos.to_frame('hour_cos'), month_sin.to_frame('month_sin'), month_cos.to_frame('month_cos'), passengers.to_frame('passenger_count')], axis=1).skb.set_name('temporal_features')
    features = spatial_features.skb.concat([temporal_features], axis=1).skb.set_name('combined_features')
    model = PyTorchMLPRegressor(hidden_dim=256, n_blocks=2, epochs=14, batch_size=4096, lr=0.003, weight_decay=0.0001, random_state=42)
    pred = features.skb.apply(model, y=y).skb.set_name('mlp_regressor')
    result = {'pred': pred, 'scoring': scoring}
    if row_keys is not None:
        result['row_keys'] = row_keys
    return result