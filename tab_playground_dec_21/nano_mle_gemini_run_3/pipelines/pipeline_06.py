import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import skrub
from lightgbm import LGBMClassifier

TRAIN_PATH = '/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv'


def locked_setup_helper():
    train_df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    y = train_df['Cover_Type'].skb.mark_as_y()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X = train_df.drop(columns=['Cover_Type', 'Id']).skb.mark_as_X(cv=cv, split_kwargs={})
    audit = {
        'target_distribution': train_df.groupby('Cover_Type').size().reset_index(name='count'),
        'train_head': train_df.head(5),
    }
    return {'X': X, 'y': y, 'scoring': 'accuracy', 'audit': audit}


def locked_setup_entry():
    return locked_setup_helper()


def build_evaluation():
    return locked_setup_entry()


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    # Remove zero-variance constant columns
    X_clean = X.drop(columns=['Soil_Type7', 'Soil_Type15'])

    elev = X['Elevation']
    slope = X['Slope']
    vd_hyd = X['Vertical_Distance_To_Hydrology']
    hd_hyd = X['Horizontal_Distance_To_Hydrology']
    hd_road = X['Horizontal_Distance_To_Roadways']
    hd_fire = X['Horizontal_Distance_To_Fire_Points']

    # Physical spatial hydrology and distance features
    water_elevation = elev.sub(vd_hyd)
    hyd_dist_sq = hd_hyd.pow(2).add(vd_hyd.pow(2))
    euclidean_distance_hydrology = hyd_dist_sq.pow(0.5)
    abs_vd_hyd = vd_hyd.abs()

    # Pairwise and aggregate horizontal distances
    hyd_road_sum = hd_hyd.add(hd_road)
    hyd_fire_sum = hd_hyd.add(hd_fire)
    road_fire_sum = hd_road.add(hd_fire)
    hyd_road_diff = hd_hyd.sub(hd_road).abs()
    hyd_fire_diff = hd_hyd.sub(hd_fire).abs()
    road_fire_diff = hd_road.sub(hd_fire).abs()
    total_horizontal_distance = hd_hyd.add(hd_road).add(hd_fire)

    # Terrain aspect decomposition and solar orientation
    aspect_mod_360 = X['Aspect'].mod(360)
    aspect_rad = X['Aspect'].mul(np.pi / 180.0)
    aspect_northness = aspect_rad.skb.apply_func(np.cos)
    aspect_eastness = aspect_rad.skb.apply_func(np.sin)
    slope_northness = slope.mul(aspect_northness)
    slope_eastness = slope.mul(aspect_eastness)
    elevation_aspect_northness = elev.mul(aspect_northness)
    elevation_aspect_eastness = elev.mul(aspect_eastness)

    # Diurnal solar dynamics
    hs_9am = X['Hillshade_9am']
    hs_noon = X['Hillshade_Noon']
    hs_3pm = X['Hillshade_3pm']
    hillshade_diff_3pm_9am = hs_3pm.sub(hs_9am)
    hillshade_diff_noon_3pm = hs_noon.sub(hs_3pm)
    hillshade_diff_noon_9am = hs_noon.sub(hs_9am)
    hillshade_total = hs_9am.add(hs_noon).add(hs_3pm)

    # Synthetic CTGAN generation artifact counts
    wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
    wilderness_area_sum = X[wilderness_cols].sum(axis=1)
    soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]
    soil_type_sum = X[soil_cols].sum(axis=1)

    hd_hyd_neg = hd_hyd.lt(0).astype('int32')
    vd_hyd_neg = vd_hyd.lt(0).astype('int32')
    hd_road_neg = hd_road.lt(0).astype('int32')
    hd_fire_neg = hd_fire.lt(0).astype('int32')
    neg_distance_count = hd_hyd_neg.add(vd_hyd_neg).add(hd_road_neg).add(hd_fire_neg)

    hs_9am_oob = hs_9am.lt(0).astype('int32').add(hs_9am.gt(255).astype('int32'))
    hs_noon_oob = hs_noon.lt(0).astype('int32').add(hs_noon.gt(255).astype('int32'))
    hs_3pm_oob = hs_3pm.lt(0).astype('int32').add(hs_3pm.gt(255).astype('int32'))
    out_of_bound_hillshade_count = hs_9am_oob.add(hs_noon_oob).add(hs_3pm_oob)

    # Targeted ecological gradients separating Spruce/Fir and Lodgepole Pine
    hydrology_slope_ratio = vd_hyd.div(hd_hyd.abs().add(10.0))
    hd_fire_clipped = hd_fire.clip(lower=0)
    moisture_fire_ratio = hd_hyd.div(hd_fire_clipped.add(100.0))
    elevation_slope = elev.mul(slope)

    X_features = X_clean.assign(
        water_elevation=water_elevation,
        euclidean_distance_hydrology=euclidean_distance_hydrology,
        abs_vertical_distance_to_hydrology=abs_vd_hyd,
        hydrology_roadways_sum=hyd_road_sum,
        hydrology_firepoints_sum=hyd_fire_sum,
        roadways_firepoints_sum=road_fire_sum,
        hydrology_roadways_diff=hyd_road_diff,
        hydrology_firepoints_diff=hyd_fire_diff,
        roadways_firepoints_diff=road_fire_diff,
        total_horizontal_distance=total_horizontal_distance,
        aspect_mod_360=aspect_mod_360,
        aspect_northness=aspect_northness,
        aspect_eastness=aspect_eastness,
        slope_northness=slope_northness,
        slope_eastness=slope_eastness,
        elevation_aspect_northness=elevation_aspect_northness,
        elevation_aspect_eastness=elevation_aspect_eastness,
        hillshade_diff_3pm_9am=hillshade_diff_3pm_9am,
        hillshade_diff_noon_3pm=hillshade_diff_noon_3pm,
        hillshade_diff_noon_9am=hillshade_diff_noon_9am,
        hillshade_total=hillshade_total,
        wilderness_area_sum=wilderness_area_sum,
        soil_type_sum=soil_type_sum,
        neg_distance_count=neg_distance_count,
        out_of_bound_hillshade_count=out_of_bound_hillshade_count,
        hydrology_slope_ratio=hydrology_slope_ratio,
        moisture_fire_ratio=moisture_fire_ratio,
        elevation_slope=elevation_slope,
    )

    models = {
        'lgbm_750_leaves63_mc100': LGBMClassifier(
            n_estimators=750,
            learning_rate=0.06,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=100,
            reg_lambda=5.0,
            reg_alpha=0.5,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        'lgbm_650_leaves127_mc150': LGBMClassifier(
            n_estimators=650,
            learning_rate=0.06,
            num_leaves=127,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=150,
            reg_lambda=10.0,
            reg_alpha=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    estimator = skrub.choose_from(models, name='model')
    pred = X_features.skb.apply(estimator, y=y)
    return {'pred': pred, 'scoring': setup['scoring']}