import numpy as np
import pandas as pd
import skrub
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRFRegressor
from time import perf_counter


class RouteBatchDataset(torch.utils.data.Dataset):
    def __init__(self, numeric, codes, target):
        self.numeric = torch.from_numpy(numeric)
        self.codes = torch.from_numpy(codes)
        self.target = torch.from_numpy(target)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        return self.numeric[indices], self.codes[indices], self.target[indices]


class AdditiveRouteNetwork(nn.Module):
    def __init__(self, n_numeric, n_hash, buckets):
        super().__init__()
        self.linear = nn.Linear(n_numeric, 1)
        self.effects = nn.Embedding(n_hash * buckets, 1)
        self.interaction_embeddings = nn.Embedding(n_hash * buckets, 4)
        self.residual = nn.Sequential(
            nn.Linear(n_numeric + 4 * n_hash, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.register_buffer('offsets', torch.arange(n_hash) * buckets)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.effects.weight)
        nn.init.normal_(self.interaction_embeddings.weight, std=0.02)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, numeric, codes):
        indices = codes + self.offsets
        additive = self.linear(numeric).squeeze(-1)
        additive = additive + self.effects(indices).squeeze(-1).sum(dim=1)
        embedded = self.interaction_embeddings(indices).flatten(start_dim=1)
        return additive + self.residual(torch.cat([numeric, embedded], dim=1)).squeeze(-1)


class AdditiveRouteRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, epochs=12, batch_size=65536, learning_rate=0.03,
                 buckets=131072, regularization=0.01, random_state=714,
                 residual_learning_rate=0.003, residual_weight_decay=0.0001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.buckets = buckets
        self.regularization = regularization
        self.random_state = random_state
        self.residual_learning_rate = residual_learning_rate
        self.residual_weight_decay = residual_weight_decay

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        torch.set_num_threads(16)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.numeric_columns_ = [c for c in X.columns if not c.startswith('hash_')]
        self.hash_columns_ = [c for c in X.columns if c.startswith('hash_')]
        a = X[self.numeric_columns_].to_numpy(dtype=np.float32, copy=True)
        a[~np.isfinite(a)] = np.nan
        self.mean_ = np.nan_to_num(np.nanmean(a, axis=0))
        self.scale_ = np.maximum(np.nan_to_num(np.nanstd(a, axis=0), nan=1.0), 0.01)
        a -= self.mean_
        a /= self.scale_
        np.nan_to_num(a, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
        np.clip(a, -20, 20, out=a)
        h = X[self.hash_columns_].to_numpy(dtype=np.int64, copy=True)
        h %= self.buckets
        target = np.asarray(y, dtype=np.float32)
        self.net_ = AdditiveRouteNetwork(a.shape[1], h.shape[1], self.buckets).to(self.device_)
        with torch.no_grad():
            self.net_.linear.bias.fill_(float(np.mean(target)))
        optimizer = torch.optim.Adam([
            {
                'params': list(self.net_.linear.parameters()) + list(self.net_.effects.parameters()),
                'lr': self.learning_rate, 'weight_decay': 0.0
            },
            {
                'params': list(self.net_.interaction_embeddings.parameters()) + list(self.net_.residual.parameters()),
                'lr': self.residual_learning_rate, 'weight_decay': self.residual_weight_decay
            }
        ])
        dataset = RouteBatchDataset(a, h, target)
        generator = torch.Generator().manual_seed(self.random_state)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        batches = torch.utils.data.BatchSampler(sampler, batch_size=self.batch_size, drop_last=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, sampler=batches, generator=generator,
            num_workers=0, pin_memory=self.device_ == 'cuda'
        )
        self.net_.train()
        self.training_history_ = []
        for epoch in range(self.epochs):
            start = perf_counter()
            total = torch.zeros((), device=self.device_)
            count = 0
            for numeric, codes, truth in loader:
                numeric = numeric.to(self.device_, non_blocking=True)
                codes = codes.to(self.device_, non_blocking=True)
                truth = truth.to(self.device_, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.net_(numeric, codes)
                mse = (prediction - truth).square().mean()
                selected = self.net_.effects(codes + self.net_.offsets)
                penalty = self.regularization * (
                    selected.square().mean() + self.net_.linear.weight.square().mean()
                )
                (mse + penalty).backward()
                optimizer.step()
                total += mse.detach() * len(truth)
                count += len(truth)
            training_mse = float(total) / count
            seconds = perf_counter() - start
            self.training_history_.append(
                {'epoch': epoch + 1, 'training_mse': training_mse, 'seconds': seconds}
            )
            print('interaction_route_epoch', epoch + 1,
                  'training_mse', training_mse, 'seconds', seconds, flush=True)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        self.net_.eval()
        output = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                part = X.iloc[start:start + self.batch_size]
                a = part[self.numeric_columns_].to_numpy(dtype=np.float32, copy=True)
                a -= self.mean_
                a /= self.scale_
                np.nan_to_num(a, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
                np.clip(a, -20, 20, out=a)
                h = part[self.hash_columns_].to_numpy(dtype=np.int64, copy=True)
                h %= self.buckets
                prediction = self.net_(
                    torch.from_numpy(a).to(self.device_),
                    torch.from_numpy(h).to(self.device_)
                )
                output.append(prediction.cpu().numpy())
        return np.concatenate(output)


class EqualWeightRouteForest(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.parent_columns_ = [c for c in X.columns if not c.startswith('forest__')]
        self.forest_columns_ = [c for c in X.columns if c.startswith('forest__')]
        start = perf_counter()
        self.parent_ = AdditiveRouteRegressor().fit(X[self.parent_columns_], y)
        print('parent_fit_seconds', perf_counter() - start, flush=True)
        self.forest_ = XGBRFRegressor(
            n_estimators=256, max_depth=12, learning_rate=1.0, subsample=0.7,
            colsample_bynode=0.8, min_child_weight=100, reg_lambda=10,
            objective='reg:squarederror', tree_method='hist', device='cuda',
            n_jobs=16, random_state=714
        )
        start = perf_counter()
        self.forest_.fit(X[self.forest_columns_], y)
        print('forest_fit_seconds', perf_counter() - start, flush=True)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return (0.5 * self.parent_.predict(X[self.parent_columns_])
                + 0.5 * self.forest_.predict(X[self.forest_columns_]))


def barrier_features(X, regime):
    regions = []
    positions = []
    for endpoint in ['pickup', 'dropoff']:
        lon = X[endpoint + '_longitude'].astype('float32')
        lat = X[endpoint + '_latitude'].astype('float32')
        valid = (lon.between(-75.5, -72) & lat.between(39.5, 42)).fillna(False)
        local = (valid & lon.between(-74.3, -73.65) & lat.between(40.5, 40.9)).fillna(False)
        west = (local & (lon < -74.01 + 0.45 * (lat - 40.71))).fillna(False)
        east = (local & ~west & (lon > -73.96 + 0.65 * (lat - 40.75))).fillna(False)
        central = local & ~west & ~east
        regions.append([central, east, west, ~local])
        positions.append((lon.where(valid, -73.98), lat.where(valid, 40.75)))
    block = X[['passenger_count']].drop(columns=['passenger_count'])
    for i, origin in enumerate(regions[0]):
        for j, destination in enumerate(regions[1]):
            block = block.assign(**{
                'barrier_pair_%d_%d' % (i, j): (origin & destination).astype('float32')
            })
    p, q = positions
    direct = (((p[0] - q[0]) * 84.3) ** 2 + ((p[1] - q[1]) * 111.2) ** 2) ** 0.5
    portals = {
        'hudson': [
            (-74.0116, 40.7258, -74.031, 40.7274),
            (-73.9969, 40.759, -74.0234, 40.7663),
            (-73.9527, 40.8505, -73.9638, 40.8544)
        ],
        'east': [
            (-74.0048, 40.7113, -73.9897, 40.7004),
            (-73.9969, 40.7156, -73.985, 40.7033),
            (-73.9841, 40.7185, -73.9707, 40.7134),
            (-73.9607, 40.759, -73.944, 40.753),
            (-73.9719, 40.745, -73.9499, 40.7428)
        ]
    }
    hudson = (regions[0][2] & (regions[1][0] | regions[1][1])
              | regions[1][2] & (regions[0][0] | regions[0][1]))
    east = (regions[0][1] & (regions[1][0] | regions[1][2])
            | regions[1][1] & (regions[0][0] | regions[0][2]))
    for family, crossing in [('hudson', hudson), ('east', east)]:
        best = None
        for alon, alat, blon, blat in portals[family]:
            length = (((alon - blon) * 84.3) ** 2 + ((alat - blat) * 111.2) ** 2) ** 0.5
            pa = (((p[0] - alon) * 84.3) ** 2 + ((p[1] - alat) * 111.2) ** 2) ** 0.5
            pb = (((p[0] - blon) * 84.3) ** 2 + ((p[1] - blat) * 111.2) ** 2) ** 0.5
            qa = (((q[0] - alon) * 84.3) ** 2 + ((q[1] - alat) * 111.2) ** 2) ** 0.5
            qb = (((q[0] - blon) * 84.3) ** 2 + ((q[1] - blat) * 111.2) ** 2) ** 0.5
            alternative = (pa + qb).skb.apply_func(np.minimum, pb + qa) + length
            best = alternative if best is None else best.skb.apply_func(np.minimum, alternative)
        flag = crossing.astype('float32')
        block = block.assign(**{
            'barrier_' + family: flag,
            'barrier_' + family + '_route_km': best.clip(0, 200) * flag,
            'barrier_' + family + '_excess_km': (best - direct).clip(0, 100) * flag,
            'barrier_' + family + '_regime': flag * regime
        })
        for airport, lon, lat in [
            ('jfk', -73.7781, 40.6413), ('lga', -73.874, 40.7769),
            ('ewr', -74.1745, 40.6895)
        ]:
            pdist = (((p[0] - lon) * 84.3) ** 2 + ((p[1] - lat) * 111.2) ** 2) ** 0.5
            qdist = (((q[0] - lon) * 84.3) ** 2 + ((q[1] - lat) * 111.2) ** 2) ** 0.5
            block = block.assign(**{
                'barrier_' + family + '_' + airport:
                    flag * ((1 - pdist / 4).clip(0, 1) + (1 - qdist / 4).clip(0, 1))
            })
    return block.astype('float32').skb.set_name('river_crossing_costs')


def build():
    setup = build_evaluation()
    X, y = setup['X'], setup['y']
    coordinates = X[[
        'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
        'dropoff_latitude', 'passenger_count'
    ]].astype('float32').skb.set_name('raw_coordinates')
    timestamp = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True, errors='coerce')
    dates = coordinates[['passenger_count']].assign(
        year=timestamp.dt.year, month=timestamp.dt.month, day=timestamp.dt.day,
        weekday=timestamp.dt.dayofweek, hour=timestamp.dt.hour,
        minute=timestamp.dt.minute, second=timestamp.dt.second
    ).drop(columns=['passenger_count']).astype('float32').skb.set_name('dates')
    dx = (X['dropoff_longitude'] - X['pickup_longitude']) * 84.3
    dy = (X['dropoff_latitude'] - X['pickup_latitude']) * 111.2
    geometry = coordinates[['passenger_count']].assign(
        delta_x=dx, delta_y=dy, absolute_x=dx.abs(), absolute_y=dy.abs(),
        euclidean_distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan_distance=dx.abs() + dy.abs(),
        midpoint_longitude=(X['pickup_longitude'] + X['dropoff_longitude']) / 2,
        midpoint_latitude=(X['pickup_latitude'] + X['dropoff_latitude']) / 2
    ).drop(columns=['passenger_count']).astype('float32').skb.set_name('geometry')
    regime = (X['pickup_datetime'].str.slice(0, 10) >= '2012-09-04').fillna(False).astype('float32')
    barriers = barrier_features(X, regime)
    forest_features = coordinates.skb.concat([dates, geometry, barriers], axis=1)
    forest_features = forest_features.replace(
        [float('inf'), -float('inf')], float('nan')
    ).add_prefix('forest__').skb.set_name('forest_finite_features')
    pvalid = (
        X['pickup_longitude'].between(-75.5, -72)
        & X['pickup_latitude'].between(39.5, 42)
    ).fillna(False)
    dvalid = (
        X['dropoff_longitude'].between(-75.5, -72)
        & X['dropoff_latitude'].between(39.5, 42)
    ).fillna(False)
    plon = X['pickup_longitude'].where(pvalid, -73.98)
    plat = X['pickup_latitude'].where(pvalid, 40.75)
    dlon = X['dropoff_longitude'].where(dvalid, -73.98)
    dlat = X['dropoff_latitude'].where(dvalid, 40.75)
    ex, ey = (dlon - plon) * 84.3, (dlat - plat) * 111.2
    rx, ry = ex * 0.846 + ey * 0.533, ey * 0.846 - ex * 0.533
    road = (rx.abs() + ry.abs()).clip(0, 200)
    rad = np.pi / 180
    hav = (
        ((dlat - plat) * rad / 2).skb.apply_func(np.sin) ** 2
        + (plat * rad).skb.apply_func(np.cos)
        * (dlat * rad).skb.apply_func(np.cos)
        * ((dlon - plon) * rad / 2).skb.apply_func(np.sin) ** 2
    )
    distance = (
        hav.clip(0, 1).skb.apply_func(np.sqrt).skb.apply_func(np.arcsin) * 12742
    ).clip(0, 200)
    year = timestamp.dt.year.fillna(2012).astype('int64') - 2009
    basis = coordinates[['passenger_count']].assign(
        pickup_invalid=(~pvalid).astype('float32'),
        dropoff_invalid=(~dvalid).astype('float32'), haversine=distance,
        rotated_x=rx.clip(-200, 200), rotated_y=ry.clip(-200, 200),
        road_distance=road, tariff_change=regime,
        tariff_distance=road * regime, year_distance=road * year
    ).drop(columns=['passenger_count'])
    for knot in [1, 2, 4, 7, 10, 15, 25, 40, 70]:
        hinge = (road - knot).clip(lower=0)
        basis = basis.assign(**{
            'hinge_%d' % knot: hinge,
            'tariff_hinge_%d' % knot: hinge * regime,
            'year_hinge_%d' % knot: hinge * year
        })
    for name, lon, lat in [
        ('jfk', -73.7781, 40.6413), ('lga', -73.874, 40.7769),
        ('ewr', -74.1745, 40.6895)
    ]:
        pdist = (((plon - lon) * 84.3) ** 2 + ((plat - lat) * 111.2) ** 2) ** 0.5
        ddist = (((dlon - lon) * 84.3) ** 2 + ((dlat - lat) * 111.2) ** 2) ** 0.5
        pp = (1 - pdist / 4).clip(0, 1) * pvalid.astype('float32')
        dp = (1 - ddist / 4).clip(0, 1) * dvalid.astype('float32')
        basis = basis.assign(**{
            name + '_pickup_distance': pdist.clip(0, 100),
            name + '_dropoff_distance': ddist.clip(0, 100),
            name + '_pickup_proximity': pp, name + '_dropoff_proximity': dp,
            name + '_pickup_trip': pp * road, name + '_dropoff_trip': dp * road,
            name + '_tariff_pickup': pp * regime, name + '_tariff_dropoff': dp * regime
        })
    basis = basis.astype('float32').skb.set_name('geographic_pricing_basis')
    hashes = coordinates[['passenger_count']].assign(
        hash_hour_of_week=(timestamp.dt.dayofweek * 24 + timestamp.dt.hour).fillna(0),
        hash_year_month=year * 12 + timestamp.dt.month.fillna(1),
        hash_passengers=X['passenger_count'].fillna(0).clip(0, 20)
    ).drop(columns=['passenger_count'])
    for name, resolution in [('fine', 0.005), ('coarse', 0.02)]:
        px = ((plon + 76) / resolution).skb.apply_func(np.floor).astype('int64')
        py = ((plat - 39) / resolution).skb.apply_func(np.floor).astype('int64')
        qx = ((dlon + 76) / resolution).skb.apply_func(np.floor).astype('int64')
        qy = ((dlat - 39) / resolution).skb.apply_func(np.floor).astype('int64')
        pickup, dropoff = px * 1009 + py, qx * 1009 + qy
        route = pickup * 73856093 + dropoff * 19349663
        hashes = hashes.assign(**{
            'hash_pickup_' + name: pickup % 131072,
            'hash_dropoff_' + name: dropoff % 131072,
            'hash_route_' + name: route % 131072,
            'hash_route_year_' + name: (route * 31 + year * 83492791) % 131072,
            'hash_route_tariff_' + name:
                (route * 17 + regime.astype('int64') * 49999) % 131072
        })
    hashes = hashes.astype('float32').skb.set_name('hashed_route_effects')
    numeric = coordinates.skb.concat([dates, geometry, basis], axis=1)
    numeric = numeric.replace([float('inf'), -float('inf')], float('nan'))
    numeric = numeric.clip(-10000, 10000).astype('float32').skb.set_name('bounded_numeric_basis')
    features = numeric.skb.concat([barriers, hashes, forest_features], axis=1)
    pred = features.skb.apply(EqualWeightRouteForest(), y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}