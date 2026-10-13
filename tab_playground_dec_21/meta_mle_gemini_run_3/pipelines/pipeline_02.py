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
        'train_head': train_df.head(5)
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

    # Drop constant columns identified in exploration
    X_clean = X.drop(columns=['Soil_Type7', 'Soil_Type15'])

    # Coordinates and primary distances
    elev = X['Elevation']
    vd_hyd = X['Vertical_Distance_To_Hydrology']
    hd_hyd = X['Horizontal_Distance_To_Hydrology']
    hd_road = X['Horizontal_Distance_To_Roadways']
    hd_fire = X['Horizontal_Distance_To_Fire_Points']

    # Hydrology geometry interactions
    water_elevation = elev.sub(vd_hyd)
    hyd_dist_sq = hd_hyd.pow(2).add(vd_hyd.pow(2))
    euclidean_distance_hydrology = hyd_dist_sq.pow(0.5)
    manhattan_distance_hydrology = hd_hyd.abs().add(vd_hyd.abs())

    # Distance sums and differences across spatial landmarks
    hyd_road_sum = hd_hyd.add(hd_road)
    hyd_fire_sum = hd_hyd.add(hd_fire)
    road_fire_sum = hd_road.add(hd_fire)
    hyd_road_diff = hd_hyd.sub(hd_road).abs()
    hyd_fire_diff = hd_hyd.sub(hd_fire).abs()
    road_fire_diff = hd_road.sub(hd_fire).abs()

    # Cyclic aspect transformations (modulo 360 wrap-around, sine and cosine encodings)
    aspect_mod = X['Aspect'].mod(360)
    aspect_rad = aspect_mod.mul(np.pi / 180.0)
    aspect_sin = aspect_rad.skb.apply_func(np.sin)
    aspect_cos = aspect_rad.skb.apply_func(np.cos)

    # Diurnal hillshade dynamics and summary statistics
    hs_9 = X['Hillshade_9am']
    hs_noon = X['Hillshade_Noon']
    hs_3 = X['Hillshade_3pm']
    hillshade_9_minus_3 = hs_9.sub(hs_3)
    hillshade_noon_minus_3 = hs_noon.sub(hs_3)
    hillshade_9_minus_noon = hs_9.sub(hs_noon)
    hillshade_mean = hs_9.add(hs_noon).add(hs_3).div(3.0)

    hs_cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    hs_max = X[hs_cols].max(axis=1)
    hs_min = X[hs_cols].min(axis=1)
    hillshade_range = hs_max.sub(hs_min)

    # Wilderness area structural features (active count and dominant code)
    wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
    wilderness_sum = X[wilderness_cols].sum(axis=1)
    wilderness_code = (
        X['Wilderness_Area1'].mul(1)
        .add(X['Wilderness_Area2'].mul(2))
        .add(X['Wilderness_Area3'].mul(3))
        .add(X['Wilderness_Area4'].mul(4))
    )

    # Soil type structural features (active count and categorical index code)
    soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]
    soil_sum = X[soil_cols].sum(axis=1)
    soil_code = X['Soil_Type1'].mul(1)
    for i in range(2, 41):
        soil_code = soil_code.add(X[f'Soil_Type{i}'].mul(i))

    X_features = X_clean.assign(
        water_elevation=water_elevation,
        euclidean_distance_hydrology=euclidean_distance_hydrology,
        manhattan_distance_hydrology=manhattan_distance_hydrology,
        hydrology_roadways_sum=hyd_road_sum,
        hydrology_firepoints_sum=hyd_fire_sum,
        roadways_firepoints_sum=road_fire_sum,
        hydrology_roadways_diff=hyd_road_diff,
        hydrology_firepoints_diff=hyd_fire_diff,
        roadways_firepoints_diff=road_fire_diff,
        aspect_sin=aspect_sin,
        aspect_cos=aspect_cos,
        hillshade_9_minus_3=hillshade_9_minus_3,
        hillshade_noon_minus_3=hillshade_noon_minus_3,
        hillshade_9_minus_noon=hillshade_9_minus_noon,
        hillshade_mean=hillshade_mean,
        hillshade_range=hillshade_range,
        wilderness_sum=wilderness_sum,
        wilderness_code=wilderness_code,
        soil_sum=soil_sum,
        soil_code=soil_code
    )

    models = {
        'lgbm_500_leaves63': LGBMClassifier(
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'lgbm_500_leaves127': LGBMClassifier(
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=127,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    }
    estimator = skrub.choose_from(models, name='model')

    pred = X_features.skb.apply(estimator, y=y)
    return {'pred': pred, 'scoring': setup['scoring']}