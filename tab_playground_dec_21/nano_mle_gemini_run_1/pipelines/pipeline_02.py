import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
import skrub

TRAIN_PATH = '/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv'


def mark_X(data, cv, split_kwargs=None):
    if split_kwargs is None:
        split_kwargs = {}
    if hasattr(data, 'skb') and hasattr(data.skb, 'mark_as_X'):
        return data.skb.mark_as_X(cv=cv, split_kwargs=split_kwargs)
    if hasattr(skrub, 'mark_as_X'):
        return skrub.mark_as_X(data, cv=cv, split_kwargs=split_kwargs)
    return data.mark_as_X(cv=cv, split_kwargs=split_kwargs)


def mark_y(data):
    if hasattr(data, 'skb') and hasattr(data.skb, 'mark_as_y'):
        return data.skb.mark_as_y()
    if hasattr(skrub, 'mark_as_y'):
        return skrub.mark_as_y(data)
    return data.mark_as_y()


def locked_setup_helper():
    raw_df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    X = raw_df.drop(columns=['Cover_Type', 'Id'])
    y = raw_df['Cover_Type']
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    split_kwargs = {}
    X_marked = mark_X(X, cv=cv, split_kwargs=split_kwargs)
    y_marked = mark_y(y)
    audit = {'class_counts': raw_df[['Cover_Type']].groupby('Cover_Type').size().reset_index(name='count')}
    return {'X': X_marked, 'y': y_marked, 'scoring': 'accuracy', 'audit': audit}


def locked_setup_entry():
    return locked_setup_helper()


def build_evaluation():
    return locked_setup_entry()


class SubsampledClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator=None, train_fraction=1.0, random_state=42):
        self.estimator = estimator
        self.train_fraction = train_fraction
        self.random_state = random_state

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator) if self.estimator is not None else None
        n_samples = len(X)
        fraction = float(self.train_fraction)
        if fraction < 1.0:
            subsample_size = int(n_samples * fraction)
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(n_samples, size=subsample_size, replace=False)
            if hasattr(X, 'iloc'):
                X_sub = X.iloc[indices]
            else:
                X_sub = X[indices]
            if hasattr(y, 'iloc'):
                y_sub = y.iloc[indices]
            else:
                y_sub = y[indices]
        else:
            X_sub = X
            y_sub = y
        self.estimator_.fit(X_sub, y_sub)
        self.classes_ = np.array([1, 2, 3, 4, 5, 6, 7])
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        raw_probs = self.estimator_.predict_proba(X)
        if hasattr(self.estimator_, 'classes_') and len(self.estimator_.classes_) == len(self.classes_) and np.array_equal(self.estimator_.classes_, self.classes_):
            return raw_probs
        full_probs = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        for col_idx, cls in enumerate(self.estimator_.classes_):
            target_cols = np.where(self.classes_ == cls)[0]
            if len(target_cols) > 0:
                full_probs[:, target_cols[0]] = raw_probs[:, col_idx]
        return full_probs


def build():
    eval_dict = build_evaluation()
    X = eval_dict['X']
    y = eval_dict['y']
    scoring = eval_dict['scoring']

    drop_cols = ['Soil_Type7', 'Soil_Type15']
    X_clean = X.drop(columns=drop_cols)

    wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
    soil_cols = [f'Soil_Type{i}' for i in range(1, 41) if i not in (7, 15)]
    binary_cols = wilderness_cols + soil_cols
    numeric_cols = [
        'Elevation',
        'Aspect',
        'Slope',
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am',
        'Hillshade_Noon',
        'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
    ]

    dtype_map = {col: 'int8' for col in binary_cols}
    dtype_map.update({col: 'int16' for col in numeric_cols})
    X_base = X_clean.astype(dtype_map)

    # 1. Hydrology geometry features
    h_dist = X_clean['Horizontal_Distance_To_Hydrology'].astype('float32')
    v_dist = X_clean['Vertical_Distance_To_Hydrology'].astype('float32')
    hydro_dist_euclidean = ((h_dist ** 2) + (v_dist ** 2)) ** 0.5
    water_elevation = (X_clean['Elevation'] - X_clean['Vertical_Distance_To_Hydrology']).astype('int16')

    # 2. Cyclic Aspect and solar exposure
    aspect_norm = (X_clean['Aspect'] % 360).astype('int16')
    aspect_rad = (X_clean['Aspect'] % 360) * (float(np.pi) / 180.0)
    aspect_sin = aspect_rad.skb.apply_func(np.sin).astype('float32')
    aspect_cos = aspect_rad.skb.apply_func(np.cos).astype('float32')

    hillshade_9_minus_3 = (X_clean['Hillshade_9am'] - X_clean['Hillshade_3pm']).astype('int16')
    hillshade_noon_minus_3 = (X_clean['Hillshade_Noon'] - X_clean['Hillshade_3pm']).astype('int16')
    hillshade_mean = (
        (X_clean['Hillshade_9am'].astype('float32') + X_clean['Hillshade_Noon'].astype('float32') + X_clean['Hillshade_3pm'].astype('float32')) / 3.0
    )

    # 3. Pairwise spatial distance arithmetic
    hydro_h = X_clean['Horizontal_Distance_To_Hydrology']
    road_h = X_clean['Horizontal_Distance_To_Roadways']
    fire_h = X_clean['Horizontal_Distance_To_Fire_Points']

    hydro_plus_road = (hydro_h + road_h).astype('int16')
    hydro_minus_road = (hydro_h - road_h).astype('int16')
    road_plus_fire = (road_h + fire_h).astype('int16')
    road_minus_fire = (road_h - fire_h).astype('int16')
    hydro_plus_fire = (hydro_h + fire_h).astype('int16')
    hydro_minus_fire = (hydro_h - fire_h).astype('int16')

    # 4. Indicator counts
    wilderness_sum = X_clean[wilderness_cols].sum(axis=1).astype('int8')
    soil_sum = X_clean[soil_cols].sum(axis=1).astype('int8')

    X_all = X_base.assign(
        Hydrology_Distance_Euclidean=hydro_dist_euclidean,
        Water_Elevation=water_elevation,
        Aspect_Modulo_360=aspect_norm,
        Aspect_Sin=aspect_sin,
        Aspect_Cos=aspect_cos,
        Hillshade_9am_minus_3pm=hillshade_9_minus_3,
        Hillshade_Noon_minus_3pm=hillshade_noon_minus_3,
        Hillshade_Mean=hillshade_mean,
        Hydro_Plus_Road=hydro_plus_road,
        Hydro_Minus_Road=hydro_minus_road,
        Road_Plus_Fire=road_plus_fire,
        Road_Minus_Fire=road_minus_fire,
        Hydro_Plus_Fire=hydro_plus_fire,
        Hydro_Minus_Fire=hydro_minus_fire,
        Wilderness_Area_Sum=wilderness_sum,
        Soil_Type_Sum=soil_sum,
    )

    cols_baseline = list(dtype_map.keys())

    cols_physical = cols_baseline + [
        'Hydrology_Distance_Euclidean',
        'Water_Elevation',
        'Aspect_Modulo_360',
        'Aspect_Sin',
        'Aspect_Cos',
        'Hillshade_9am_minus_3pm',
        'Hillshade_Noon_minus_3pm',
        'Hillshade_Mean',
    ]

    cols_all_domain = cols_physical + [
        'Hydro_Plus_Road',
        'Hydro_Minus_Road',
        'Road_Plus_Fire',
        'Road_Minus_Fire',
        'Hydro_Plus_Fire',
        'Hydro_Minus_Fire',
        'Wilderness_Area_Sum',
        'Soil_Type_Sum',
    ]

    feature_set_choice = skrub.choose_from(
        {
            'baseline': cols_baseline,
            'physical': cols_physical,
            'all_domain': cols_all_domain,
        },
        name='feature_set',
    )

    X_features = X_all[feature_set_choice]

    base_clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model = SubsampledClassifier(estimator=base_clf, train_fraction=1.0, random_state=42)
    pred = X_features.skb.apply(model, y=y)
    out = {'pred': pred, 'scoring': scoring}
    if 'row_keys' in eval_dict:
        out['row_keys'] = eval_dict['row_keys']
    return out