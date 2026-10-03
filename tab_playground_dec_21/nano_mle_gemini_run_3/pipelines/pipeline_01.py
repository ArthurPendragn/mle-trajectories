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
    audit = {'target_distribution': train_df.groupby('Cover_Type').size().reset_index(name='count'), 'train_head': train_df.head(5)}
    return {'X': X, 'y': y, 'scoring': 'accuracy', 'audit': audit}

def locked_setup_entry():
    return locked_setup_helper()

def build_evaluation():
    return locked_setup_entry()

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    X_clean = X.drop(columns=['Soil_Type7', 'Soil_Type15'])

    elev = X['Elevation']
    vd_hyd = X['Vertical_Distance_To_Hydrology']
    hd_hyd = X['Horizontal_Distance_To_Hydrology']
    hd_road = X['Horizontal_Distance_To_Roadways']
    hd_fire = X['Horizontal_Distance_To_Fire_Points']

    water_elevation = elev.sub(vd_hyd)
    hyd_dist_sq = hd_hyd.pow(2).add(vd_hyd.pow(2))
    euclidean_distance_hydrology = hyd_dist_sq.pow(0.5)

    hyd_road_sum = hd_hyd.add(hd_road)
    hyd_fire_sum = hd_hyd.add(hd_fire)
    road_fire_sum = hd_road.add(hd_fire)

    hyd_road_diff = hd_hyd.sub(hd_road).abs()
    hyd_fire_diff = hd_hyd.sub(hd_fire).abs()
    road_fire_diff = hd_road.sub(hd_fire).abs()

    X_features = X_clean.assign(
        water_elevation=water_elevation,
        euclidean_distance_hydrology=euclidean_distance_hydrology,
        hydrology_roadways_sum=hyd_road_sum,
        hydrology_firepoints_sum=hyd_fire_sum,
        roadways_firepoints_sum=road_fire_sum,
        hydrology_roadways_diff=hyd_road_diff,
        hydrology_firepoints_diff=hyd_fire_diff,
        roadways_firepoints_diff=road_fire_diff,
    )

    models = {
        'lgbm_150': LGBMClassifier(
            n_estimators=150,
            learning_rate=0.1,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        'lgbm_300': LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }
    estimator = skrub.choose_from(models, name='model')

    pred = X_features.skb.apply(estimator, y=y)

    return {
        'pred': pred,
        'scoring': setup['scoring'],
    }