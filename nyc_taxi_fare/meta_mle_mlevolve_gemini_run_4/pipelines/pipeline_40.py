import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

class TaxiFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        p_lon = X['pickup_longitude'].to_numpy(dtype=np.float64)
        p_lat = X['pickup_latitude'].to_numpy(dtype=np.float64)
        d_lon = X['dropoff_longitude'].to_numpy(dtype=np.float64)
        d_lat = X['dropoff_latitude'].to_numpy(dtype=np.float64)
        passengers = X['passenger_count'].to_numpy(dtype=np.float64)
        dlon = d_lon - p_lon
        dlat = d_lat - p_lat
        abs_dlon = np.abs(dlon)
        abs_dlat = np.abs(dlat)
        lat1, lon1 = (np.radians(p_lat), np.radians(p_lon))
        lat2, lon2 = (np.radians(d_lat), np.radians(d_lon))
        h_dlat = lat2 - lat1
        h_dlon = lon2 - lon1
        a = np.sin(h_dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(h_dlon / 2.0) ** 2
        haversine_km = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        rot_rad = 29.0 * np.pi / 180.0
        cos_rot = np.cos(rot_rad)
        sin_rot = np.sin(rot_rad)
        mean_lat = np.radians(40.75)
        dx_km = dlon * 111.32 * np.cos(mean_lat)
        dy_km = dlat * 110.57
        dx_rot = dx_km * cos_rot - dy_km * sin_rot
        dy_rot = dx_km * sin_rot + dy_km * cos_rot
        rot_raw_dx = dx_rot
        rot_raw_dy = dy_rot
        rot_dx = np.abs(dx_rot)
        rot_dy = np.abs(dy_rot)
        manhattan_km = rot_dx + rot_dy
        euclidean_km = np.sqrt(dx_km ** 2 + dy_km ** 2)
        pick_x_km = (p_lon - -74.0) * 84.18
        pick_y_km = (p_lat - 40.7) * 111.13
        drop_x_km = (d_lon - -74.0) * 84.18
        drop_y_km = (d_lat - 40.7) * 111.13
        rot_pick_x = pick_x_km * cos_rot - pick_y_km * sin_rot
        rot_pick_y = pick_x_km * sin_rot + pick_y_km * cos_rot
        rot_drop_x = drop_x_km * cos_rot - drop_y_km * sin_rot
        rot_drop_y = drop_x_km * sin_rot + drop_y_km * cos_rot
        center_rot_x = (rot_pick_x + rot_drop_x) / 2.0
        center_rot_y = (rot_pick_y + rot_drop_y) / 2.0
        avenue_dist_frac = rot_dy / (manhattan_km + 0.05)
        crosstown_dist_frac = rot_dx / (manhattan_km + 0.05)
        tortuosity = manhattan_km / (haversine_km + 0.05)
        grid_bearing = np.arctan2(dy_rot, dx_rot)
        jfk_p_dist = np.sqrt((p_lon - -73.7781) ** 2 + (p_lat - 40.6413) ** 2)
        jfk_d_dist = np.sqrt((d_lon - -73.7781) ** 2 + (d_lat - 40.6413) ** 2)
        is_jfk = ((jfk_p_dist < 0.03) | (jfk_d_dist < 0.03)).astype(np.float64)
        lga_p_dist = np.sqrt((p_lon - -73.874) ** 2 + (p_lat - 40.7769) ** 2)
        lga_d_dist = np.sqrt((d_lon - -73.874) ** 2 + (d_lat - 40.7769) ** 2)
        is_lga = ((lga_p_dist < 0.03) | (lga_d_dist < 0.03)).astype(np.float64)
        ewr_p_dist = np.sqrt((p_lon - -74.1745) ** 2 + (p_lat - 40.6895) ** 2)
        ewr_d_dist = np.sqrt((d_lon - -74.1745) ** 2 + (d_lat - 40.6895) ** 2)
        is_ewr = ((ewr_p_dist < 0.03) | (ewr_d_dist < 0.03)).astype(np.float64)
        p_manhattan = ((p_lon >= -74.02) & (p_lon <= -73.93) & (p_lat >= 40.7) & (p_lat <= 40.87)).astype(np.float64)
        d_manhattan = ((d_lon >= -74.02) & (d_lon <= -73.93) & (d_lat >= 40.7) & (d_lat <= 40.87)).astype(np.float64)
        is_both_manhattan = p_manhattan * d_manhattan
        is_cross_borough = (p_manhattan != d_manhattan).astype(np.float64)
        p_brooklyn = ((p_lon >= -74.04) & (p_lon <= -73.85) & (p_lat >= 40.57) & (p_lat <= 40.73)).astype(np.float64)
        d_brooklyn = ((d_lon >= -74.04) & (d_lon <= -73.85) & (d_lat >= 40.57) & (d_lat <= 40.73)).astype(np.float64)
        p_queens = ((p_lon >= -73.96) & (p_lon <= -73.7) & (p_lat >= 40.54) & (p_lat <= 40.8)).astype(np.float64)
        d_queens = ((d_lon >= -73.96) & (d_lon <= -73.7) & (d_lat >= 40.54) & (d_lat <= 40.8)).astype(np.float64)
        p_bronx = ((p_lon >= -73.93) & (p_lon <= -73.76) & (p_lat >= 40.8) & (p_lat <= 40.91)).astype(np.float64)
        d_bronx = ((d_lon >= -73.93) & (d_lon <= -73.76) & (d_lat >= 40.8) & (d_lat <= 40.91)).astype(np.float64)
        is_manhattan_brooklyn = ((p_manhattan > 0) & (d_brooklyn > 0) | (d_manhattan > 0) & (p_brooklyn > 0)).astype(np.float64)
        is_manhattan_queens = ((p_manhattan > 0) & (d_queens > 0) | (d_manhattan > 0) & (p_queens > 0)).astype(np.float64)
        is_manhattan_bronx = ((p_manhattan > 0) & (d_bronx > 0) | (d_manhattan > 0) & (p_bronx > 0)).astype(np.float64)
        is_jfk_manhattan = ((jfk_p_dist < 0.03) & (d_manhattan > 0) | (jfk_d_dist < 0.03) & (p_manhattan > 0)).astype(np.float64)
        is_lga_manhattan = ((lga_p_dist < 0.03) & (d_manhattan > 0) | (lga_d_dist < 0.03) & (p_manhattan > 0)).astype(np.float64)
        is_ewr_manhattan = ((ewr_p_dist < 0.03) & (d_manhattan > 0) | (ewr_d_dist < 0.03) & (p_manhattan > 0)).astype(np.float64)
        p_ues = ((p_lon >= -73.965) & (p_lon <= -73.945) & (p_lat >= 40.765) & (p_lat <= 40.8)).astype(np.float64)
        d_ues = ((d_lon >= -73.965) & (d_lon <= -73.945) & (d_lat >= 40.765) & (d_lat <= 40.8)).astype(np.float64)
        p_uws = ((p_lon >= -73.988) & (p_lon <= -73.968) & (p_lat >= 40.765) & (p_lat <= 40.8)).astype(np.float64)
        d_uws = ((d_lon >= -73.988) & (d_lon <= -73.968) & (d_lat >= 40.765) & (d_lat <= 40.8)).astype(np.float64)
        is_cp_crossing = ((p_ues > 0) & (d_uws > 0) | (p_uws > 0) & (d_ues > 0)).astype(np.float64)
        cp_crossing_dist = is_cp_crossing * rot_dx
        center_lat = (p_lat + d_lat) / 2.0
        center_lon = (p_lon + d_lon) / 2.0
        center_dist_midtown = np.sqrt(((center_lat - 40.755) * 111.13) ** 2 + ((center_lon - -73.985) * 84.18) ** 2)
        mid_midtown_tunnel = np.sqrt(((center_lat - 40.744) * 111.13) ** 2 + ((center_lon - -73.968) * 84.18) ** 2)
        mid_rfk = np.sqrt(((center_lat - 40.78) * 111.13) ** 2 + ((center_lon - -73.924) * 84.18) ** 2)
        mid_queensboro = np.sqrt(((center_lat - 40.757) * 111.13) ** 2 + ((center_lon - -73.955) * 84.18) ** 2)
        mid_battery_tunnel = np.sqrt(((center_lat - 40.697) * 111.13) ** 2 + ((center_lon - -74.015) * 84.18) ** 2)
        mid_brooklyn_bridge = np.sqrt(((center_lat - 40.706) * 111.13) ** 2 + ((center_lon - -73.997) * 84.18) ** 2)
        mid_lincoln = np.sqrt(((center_lat - 40.76) * 111.13) ** 2 + ((center_lon - -74.004) * 84.18) ** 2)
        mid_holland = np.sqrt(((center_lat - 40.726) * 111.13) ** 2 + ((center_lon - -74.016) * 84.18) ** 2)
        mid_gwb = np.sqrt(((center_lat - 40.852) * 111.13) ** 2 + ((center_lon - -73.953) * 84.18) ** 2)
        tolled_midtown_tunnel = ((is_manhattan_queens > 0) & (mid_midtown_tunnel < 3.0)).astype(np.float64)
        tolled_rfk = (((is_manhattan_queens > 0) | (is_manhattan_bronx > 0)) & (mid_rfk < 3.0)).astype(np.float64)
        tolled_battery_tunnel = ((is_manhattan_brooklyn > 0) & (mid_battery_tunnel < 2.5)).astype(np.float64)
        tolled_hudson_tunnel = ((is_ewr_manhattan > 0) | (d_lon < -74.03)).astype(np.float64)
        free_queensboro = ((is_manhattan_queens > 0) & (mid_queensboro < 3.0)).astype(np.float64)
        free_brooklyn = ((is_manhattan_brooklyn > 0) & (mid_brooklyn_bridge < 2.5)).astype(np.float64)
        midtown_p_dist = np.sqrt(((p_lat - 40.755) * 111.13) ** 2 + ((p_lon - -73.985) * 84.18) ** 2)
        midtown_d_dist = np.sqrt(((d_lat - 40.755) * 111.13) ** 2 + ((d_lon - -73.985) * 84.18) ** 2)
        fidi_p_dist = np.sqrt(((p_lat - 40.7128) * 111.13) ** 2 + ((p_lon - -74.006) * 84.18) ** 2)
        fidi_d_dist = np.sqrt(((d_lat - 40.7128) * 111.13) ** 2 + ((d_lon - -74.006) * 84.18) ** 2)
        gc_p_dist = np.sqrt(((p_lat - 40.7527) * 111.13) ** 2 + ((p_lon - -73.9772) * 84.18) ** 2)
        gc_d_dist = np.sqrt(((d_lat - 40.7527) * 111.13) ** 2 + ((d_lon - -73.9772) * 84.18) ** 2)
        penn_p_dist = np.sqrt(((p_lat - 40.7505) * 111.13) ** 2 + ((p_lon - -73.9935) * 84.18) ** 2)
        penn_d_dist = np.sqrt(((d_lat - 40.7505) * 111.13) ** 2 + ((d_lon - -73.9935) * 84.18) ** 2)
        ues_p_dist = np.sqrt(((p_lat - 40.7736) * 111.13) ** 2 + ((p_lon - -73.9566) * 84.18) ** 2)
        ues_d_dist = np.sqrt(((d_lat - 40.7736) * 111.13) ** 2 + ((d_lon - -73.9566) * 84.18) ** 2)
        uws_p_dist = np.sqrt(((p_lat - 40.787) * 111.13) ** 2 + ((p_lon - -73.9754) * 84.18) ** 2)
        uws_d_dist = np.sqrt(((d_lat - 40.787) * 111.13) ** 2 + ((d_lon - -73.9754) * 84.18) ** 2)
        bk_downtown_p_dist = np.sqrt(((p_lat - 40.689) * 111.13) ** 2 + ((p_lon - -73.985) * 84.18) ** 2)
        bk_downtown_d_dist = np.sqrt(((d_lat - 40.689) * 111.13) ** 2 + ((d_lon - -73.985) * 84.18) ** 2)
        lic_p_dist = np.sqrt(((p_lat - 40.7447) * 111.13) ** 2 + ((p_lon - -73.9485) * 84.18) ** 2)
        lic_d_dist = np.sqrt(((d_lat - 40.7447) * 111.13) ** 2 + ((d_lon - -73.9485) * 84.18) ** 2)
        is_zero_dist = (manhattan_km < 0.05).astype(np.float64)
        is_westchester = (d_lat > 40.915).astype(np.float64)
        is_nassau = (d_lon > -73.7).astype(np.float64)
        is_suburban = ((is_westchester > 0) | (is_nassau > 0)).astype(np.float64)
        dt = pd.to_datetime(X['pickup_datetime'].astype(str).str.slice(0, 19), format='%Y-%m-%d %H:%M:%S')
        year = dt.dt.year.to_numpy(dtype=np.float64)
        month = dt.dt.month.to_numpy(dtype=np.float64)
        day = dt.dt.day.to_numpy(dtype=np.float64)
        hour = dt.dt.hour.to_numpy(dtype=np.float64)
        minute = dt.dt.minute.to_numpy(dtype=np.float64)
        weekday = dt.dt.weekday.to_numpy(dtype=np.float64)
        hour_float = hour + minute / 60.0
        sin_diurnal = np.sin(2.0 * np.pi * hour_float / 24.0)
        cos_diurnal = np.cos(2.0 * np.pi * hour_float / 24.0)
        sin_hour = np.sin(2.0 * np.pi * hour / 24.0)
        cos_hour = np.cos(2.0 * np.pi * hour / 24.0)
        sin_month = np.sin(2.0 * np.pi * month / 12.0)
        cos_month = np.cos(2.0 * np.pi * month / 12.0)
        sin_weekday = np.sin(2.0 * np.pi * weekday / 7.0)
        cos_weekday = np.cos(2.0 * np.pi * weekday / 7.0)
        is_weekend = (weekday >= 5).astype(np.float64)
        is_rush_hour = ((weekday < 5) & (hour >= 16) & (hour < 20)).astype(np.float64)
        is_morning_rush = ((weekday < 5) & (hour >= 7) & (hour < 10)).astype(np.float64)
        is_midday = ((weekday < 5) & (hour >= 10) & (hour < 16)).astype(np.float64)
        is_overnight = ((hour >= 20) | (hour < 6)).astype(np.float64)
        is_post_nov_2009 = ((year > 2009) | (year == 2009) & (month >= 11)).astype(np.float64)
        is_post_2012 = ((year > 2012) | (year == 2012) & ((month > 9) | (month == 9) & (day >= 4))).astype(np.float64)
        is_post_2015 = (year >= 2015).astype(np.float64)
        manhattan_crosstown_km = is_both_manhattan * rot_dx
        manhattan_avenue_km = is_both_manhattan * rot_dy
        manhattan_rush_crosstown = is_both_manhattan * is_rush_hour * rot_dx
        manhattan_morning_rush_crosstown = is_both_manhattan * is_morning_rush * rot_dx
        manhattan_midday_crosstown = is_both_manhattan * is_midday * rot_dx
        manhattan_rush_avenue = is_both_manhattan * is_rush_hour * rot_dy
        manhattan_morning_rush_avenue = is_both_manhattan * is_morning_rush * rot_dy
        manhattan_midday_avenue = is_both_manhattan * is_midday * rot_dy
        lga_rush_dist = is_lga * is_rush_hour * haversine_km
        ewr_rush_dist = is_ewr * is_rush_hour * haversine_km
        non_jfk_manhattan_km = (1.0 - is_jfk_manhattan) * manhattan_km
        non_jfk_haversine_km = (1.0 - is_jfk_manhattan) * haversine_km
        jfk_toll = is_jfk_manhattan * (5.33 * is_post_2012 + 4.5 * (1.0 - is_post_2012))
        jfk_flat_rate = is_jfk_manhattan * (52.0 * is_post_2012 + 45.0 * (1.0 - is_post_2012)) + jfk_toll
        ewr_surcharge = is_ewr_manhattan * (17.5 * is_post_2012 + 15.0 * (1.0 - is_post_2012))
        statutory_surcharge = 0.5 * is_post_nov_2009 + 0.5 * is_overnight + 1.0 * is_rush_hour + 0.3 * is_post_2015
        post_2012_dist = is_post_2012 * non_jfk_manhattan_km
        post_2012_hav = is_post_2012 * non_jfk_haversine_km
        sqrt_manhattan_km = np.sqrt(non_jfk_manhattan_km)
        log_manhattan_km = np.log1p(non_jfk_manhattan_km)
        sqrt_haversine_km = np.sqrt(non_jfk_haversine_km)
        log_haversine_km = np.log1p(non_jfk_haversine_km)
        rush_hour_dist = is_rush_hour * non_jfk_manhattan_km
        morning_rush_dist = is_morning_rush * non_jfk_manhattan_km
        midday_dist = is_midday * non_jfk_manhattan_km
        overnight_dist = is_overnight * non_jfk_manhattan_km
        suburban_dist = is_suburban * non_jfk_manhattan_km
        post_2012_rush = is_post_2012 * is_rush_hour
        rate_per_km = 1.2427 * (1.0 - is_post_2012) + 1.5534 * is_post_2012
        effective_dist_km = is_both_manhattan * manhattan_km + (1.0 - is_both_manhattan) * (0.6 * manhattan_km + 0.4 * haversine_km)
        metered_dist_fare = effective_dist_km * rate_per_km
        base_fare = 2.5 + metered_dist_fare
        statutory_fare_estimate = (1.0 - is_jfk_manhattan) * base_fare + jfk_flat_rate + statutory_surcharge + ewr_surcharge
        feats = {'pickup_longitude': p_lon, 'pickup_latitude': p_lat, 'dropoff_longitude': d_lon, 'dropoff_latitude': d_lat, 'passenger_count': passengers, 'dlon': dlon, 'dlat': dlat, 'abs_dlon': abs_dlon, 'abs_dlat': abs_dlat, 'haversine_km': haversine_km, 'manhattan_km': manhattan_km, 'euclidean_km': euclidean_km, 'rot_raw_dx': rot_raw_dx, 'rot_raw_dy': rot_raw_dy, 'rot_dx': rot_dx, 'rot_dy': rot_dy, 'rot_pick_x': rot_pick_x, 'rot_pick_y': rot_pick_y, 'rot_drop_x': rot_drop_x, 'rot_drop_y': rot_drop_y, 'center_rot_x': center_rot_x, 'center_rot_y': center_rot_y, 'center_dist_midtown': center_dist_midtown, 'avenue_dist_frac': avenue_dist_frac, 'crosstown_dist_frac': crosstown_dist_frac, 'tortuosity': tortuosity, 'grid_bearing': grid_bearing, 'is_cp_crossing': is_cp_crossing, 'cp_crossing_dist': cp_crossing_dist, 'manhattan_crosstown_km': manhattan_crosstown_km, 'manhattan_avenue_km': manhattan_avenue_km, 'manhattan_rush_crosstown': manhattan_rush_crosstown, 'manhattan_morning_rush_crosstown': manhattan_morning_rush_crosstown, 'manhattan_midday_crosstown': manhattan_midday_crosstown, 'manhattan_rush_avenue': manhattan_rush_avenue, 'manhattan_morning_rush_avenue': manhattan_morning_rush_avenue, 'manhattan_midday_avenue': manhattan_midday_avenue, 'lga_rush_dist': lga_rush_dist, 'ewr_rush_dist': ewr_rush_dist, 'non_jfk_manhattan_km': non_jfk_manhattan_km, 'non_jfk_haversine_km': non_jfk_haversine_km, 'post_2012_dist': post_2012_dist, 'post_2012_hav': post_2012_hav, 'sqrt_manhattan_km': sqrt_manhattan_km, 'log_manhattan_km': log_manhattan_km, 'sqrt_haversine_km': sqrt_haversine_km, 'log_haversine_km': log_haversine_km, 'rush_hour_dist': rush_hour_dist, 'morning_rush_dist': morning_rush_dist, 'midday_dist': midday_dist, 'overnight_dist': overnight_dist, 'suburban_dist': suburban_dist, 'post_2012_rush': post_2012_rush, 'jfk_toll': jfk_toll, 'jfk_flat_rate': jfk_flat_rate, 'ewr_surcharge': ewr_surcharge, 'statutory_surcharge': statutory_surcharge, 'statutory_fare_estimate': statutory_fare_estimate, 'jfk_p_dist': jfk_p_dist, 'jfk_d_dist': jfk_d_dist, 'is_jfk': is_jfk, 'is_jfk_manhattan': is_jfk_manhattan, 'lga_p_dist': lga_p_dist, 'lga_d_dist': lga_d_dist, 'is_lga': is_lga, 'is_lga_manhattan': is_lga_manhattan, 'ewr_p_dist': ewr_p_dist, 'ewr_d_dist': ewr_d_dist, 'is_ewr': is_ewr, 'is_ewr_manhattan': is_ewr_manhattan, 'p_manhattan': p_manhattan, 'd_manhattan': d_manhattan, 'is_both_manhattan': is_both_manhattan, 'is_cross_borough': is_cross_borough, 'is_manhattan_brooklyn': is_manhattan_brooklyn, 'is_manhattan_queens': is_manhattan_queens, 'is_manhattan_bronx': is_manhattan_bronx, 'mid_midtown_tunnel': mid_midtown_tunnel, 'mid_rfk': mid_rfk, 'mid_queensboro': mid_queensboro, 'mid_battery_tunnel': mid_battery_tunnel, 'mid_brooklyn_bridge': mid_brooklyn_bridge, 'mid_lincoln': mid_lincoln, 'mid_holland': mid_holland, 'mid_gwb': mid_gwb, 'tolled_midtown_tunnel': tolled_midtown_tunnel, 'tolled_rfk': tolled_rfk, 'tolled_battery_tunnel': tolled_battery_tunnel, 'tolled_hudson_tunnel': tolled_hudson_tunnel, 'free_queensboro': free_queensboro, 'free_brooklyn': free_brooklyn, 'midtown_p_dist': midtown_p_dist, 'midtown_d_dist': midtown_d_dist, 'fidi_p_dist': fidi_p_dist, 'fidi_d_dist': fidi_d_dist, 'gc_p_dist': gc_p_dist, 'gc_d_dist': gc_d_dist, 'penn_p_dist': penn_p_dist, 'penn_d_dist': penn_d_dist, 'ues_p_dist': ues_p_dist, 'ues_d_dist': ues_d_dist, 'uws_p_dist': uws_p_dist, 'uws_d_dist': uws_d_dist, 'bk_downtown_p_dist': bk_downtown_p_dist, 'bk_downtown_d_dist': bk_downtown_d_dist, 'lic_p_dist': lic_p_dist, 'lic_d_dist': lic_d_dist, 'is_zero_dist': is_zero_dist, 'is_westchester': is_westchester, 'is_nassau': is_nassau, 'is_suburban': is_suburban, 'year': year, 'month': month, 'day': day, 'hour': hour, 'weekday': weekday, 'hour_float': hour_float, 'sin_diurnal': sin_diurnal, 'cos_diurnal': cos_diurnal, 'sin_hour': sin_hour, 'cos_hour': cos_hour, 'sin_month': sin_month, 'cos_month': cos_month, 'sin_weekday': sin_weekday, 'cos_weekday': cos_weekday, 'is_weekend': is_weekend, 'is_rush_hour': is_rush_hour, 'is_morning_rush': is_morning_rush, 'is_midday': is_midday, 'is_overnight': is_overnight, 'is_post_nov_2009': is_post_nov_2009, 'is_post_2012': is_post_2012, 'is_post_2015': is_post_2015}
        for wl in [1.0, 2.5, 5.0, 10.0]:
            freq = 2.0 * np.pi / wl
            wl_str = str(wl).replace('.', '_')
            feats[f'sin_pick_x_{wl_str}'] = np.sin(rot_pick_x * freq)
            feats[f'cos_pick_x_{wl_str}'] = np.cos(rot_pick_x * freq)
            feats[f'sin_pick_y_{wl_str}'] = np.sin(rot_pick_y * freq)
            feats[f'cos_pick_y_{wl_str}'] = np.cos(rot_pick_y * freq)
            feats[f'sin_drop_x_{wl_str}'] = np.sin(rot_drop_x * freq)
            feats[f'cos_drop_x_{wl_str}'] = np.cos(rot_drop_x * freq)
            feats[f'sin_drop_y_{wl_str}'] = np.sin(rot_drop_y * freq)
            feats[f'cos_drop_y_{wl_str}'] = np.cos(rot_drop_y * freq)
        return pd.DataFrame(feats)

class GatedResBlock(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, 2 * dim)
        self.glu1 = nn.GLU(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, 2 * dim)
        self.glu2 = nn.GLU(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.glu1(self.linear1(h))
        h = self.dropout(h)
        h = self.glu2(self.linear2(h))
        return residual + h

class WideAndDeepTabularResNet(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int=256, num_blocks: int=3, dropout: float=0.1):
        super().__init__()
        self.wide = nn.Linear(in_features, 1, bias=False)
        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([GatedResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_out = self.wide(x)
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        h = self.gelu(h)
        deep_out = self.head(h)
        return wide_out + deep_out

class TabularResNetRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, hidden_dim: int=256, num_blocks: int=3, dropout: float=0.1, lr: float=0.003, epochs: int=25, batch_size: int=2048, random_state: int=42):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        X_mat = np.asarray(X, dtype=np.float32)
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_mat).astype(np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        y_mean = float(y_arr.mean())
        in_dim = X_scaled.shape[1]
        net = WideAndDeepTabularResNet(in_features=in_dim, hidden_dim=self.hidden_dim, num_blocks=self.num_blocks, dropout=self.dropout).to(device)
        with torch.no_grad():
            net.head.bias.fill_(y_mean)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=0.0001)
        dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_arr).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        total_steps = self.epochs * len(loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-05)
        criterion = nn.MSELoss()
        net.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                out = net(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
        self.net_ = net
        self.device_ = device
        return self

    def predict(self, X):
        X_mat = np.asarray(X, dtype=np.float32)
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler_.transform(X_mat).astype(np.float32)
        if len(X_scaled) == 0:
            return np.array([], dtype=np.float32)
        self.net_.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), 4096):
                batch_x = torch.from_numpy(X_scaled[i:i + 4096]).to(self.device_)
                batch_pred = self.net_(batch_x).squeeze(1).cpu().numpy()
                preds.append(batch_pred)
        return np.concatenate(preds)

class TriEnsembleResidualRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, hidden_dim: int=256, num_blocks: int=3, dropout: float=0.1, lr: float=0.003, epochs: int=25, batch_size: int=2048, lgb_n_estimators: int=1000, lgb_learning_rate: float=0.04, lgb_num_leaves: int=63, lgb_min_child_samples: int=50, lgb_reg_alpha: float=1.0, lgb_reg_lambda: float=5.0, lgb_colsample_bytree: float=0.8, lgb_subsample: float=0.8, lgb_subsample_freq: int=1, xgb_n_estimators: int=1000, xgb_learning_rate: float=0.04, xgb_max_depth: int=8, xgb_reg_alpha: float=1.0, xgb_reg_lambda: float=5.0, xgb_colsample_bytree: float=0.8, xgb_subsample: float=0.8, w_resnet: float=0.34, w_lgb: float=0.33, w_xgb: float=0.33, fare_floor: float=2.5, fare_cap: float=500.0, random_state: int=42, n_jobs: int=8):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lgb_n_estimators = lgb_n_estimators
        self.lgb_learning_rate = lgb_learning_rate
        self.lgb_num_leaves = lgb_num_leaves
        self.lgb_min_child_samples = lgb_min_child_samples
        self.lgb_reg_alpha = lgb_reg_alpha
        self.lgb_reg_lambda = lgb_reg_lambda
        self.lgb_colsample_bytree = lgb_colsample_bytree
        self.lgb_subsample = lgb_subsample
        self.lgb_subsample_freq = lgb_subsample_freq
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_max_depth = xgb_max_depth
        self.xgb_reg_alpha = xgb_reg_alpha
        self.xgb_reg_lambda = xgb_reg_lambda
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_subsample = xgb_subsample
        self.w_resnet = w_resnet
        self.w_lgb = w_lgb
        self.w_xgb = w_xgb
        self.fare_floor = fare_floor
        self.fare_cap = fare_cap
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            statutory_baseline = X['statutory_fare_estimate'].to_numpy(dtype=np.float64)
            self.baseline_idx_ = X.columns.get_loc('statutory_fare_estimate')
        else:
            self.baseline_idx_ = -1
            statutory_baseline = X[:, self.baseline_idx_].astype(np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        residual = y_arr - statutory_baseline
        X_mat = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.resnet_ = TabularResNetRegressor(hidden_dim=self.hidden_dim, num_blocks=self.num_blocks, dropout=self.dropout, lr=self.lr, epochs=self.epochs, batch_size=self.batch_size, random_state=self.random_state)
        self.resnet_.fit(X_mat, residual)
        self.lgb_ = LGBMRegressor(n_estimators=self.lgb_n_estimators, learning_rate=self.lgb_learning_rate, num_leaves=self.lgb_num_leaves, min_child_samples=self.lgb_min_child_samples, reg_alpha=self.lgb_reg_alpha, reg_lambda=self.lgb_reg_lambda, colsample_bytree=self.lgb_colsample_bytree, subsample=self.lgb_subsample, subsample_freq=self.lgb_subsample_freq, random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1)
        self.lgb_.fit(X, residual)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.xgb_ = XGBRegressor(n_estimators=self.xgb_n_estimators, learning_rate=self.xgb_learning_rate, max_depth=self.xgb_max_depth, reg_alpha=self.xgb_reg_alpha, reg_lambda=self.xgb_reg_lambda, colsample_bytree=self.xgb_colsample_bytree, subsample=self.xgb_subsample, random_state=self.random_state, n_jobs=self.n_jobs, tree_method='hist', device=device)
            self.xgb_.fit(X_mat, residual)
        except Exception:
            self.xgb_ = XGBRegressor(n_estimators=self.xgb_n_estimators, learning_rate=self.xgb_learning_rate, max_depth=self.xgb_max_depth, reg_alpha=self.xgb_reg_alpha, reg_lambda=self.xgb_reg_lambda, colsample_bytree=self.xgb_colsample_bytree, subsample=self.xgb_subsample, random_state=self.random_state, n_jobs=self.n_jobs, tree_method='hist')
            self.xgb_.fit(X_mat, residual)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            statutory_baseline = X['statutory_fare_estimate'].to_numpy(dtype=np.float64)
        else:
            statutory_baseline = X[:, self.baseline_idx_].astype(np.float64)
        X_mat = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        pred_resnet = self.resnet_.predict(X_mat)
        pred_lgb = self.lgb_.predict(X)
        pred_xgb = self.xgb_.predict(X_mat)
        pred_res = self.w_resnet * pred_resnet + self.w_lgb * pred_lgb + self.w_xgb * pred_xgb
        pred = statutory_baseline + pred_res
        return np.clip(pred, self.fare_floor, self.fare_cap)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']
    features = X.skb.apply(TaxiFeatureTransformer()).skb.set_name('features')
    chosen_model = TriEnsembleResidualRegressor(w_resnet=0.34, w_lgb=0.33, w_xgb=0.33)
    pred = features.skb.apply(chosen_model, y=y).skb.set_name('model')
    result = {'pred': pred, 'scoring': scoring}
    if 'row_keys' in setup:
        result['row_keys'] = setup['row_keys']
    return result