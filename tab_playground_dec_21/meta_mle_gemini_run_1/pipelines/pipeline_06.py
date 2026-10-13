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
        'Horizontal_Distance_To_Fire_Points'
    ]
    wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
    soil_cols = [f'Soil_Type{i}' for i in range(1, 41) if i not in (7, 15)]
    base_cols = numeric_cols + wilderness_cols + soil_cols

    base_dtype_map = {col: 'int8' for col in wilderness_cols + soil_cols}
    base_dtype_map.update({col: 'int16' for col in numeric_cols})
    X_base = X_clean.astype(base_dtype_map)

    h = X_base['Horizontal_Distance_To_Hydrology'].astype('float32')
    v = X_base['Vertical_Distance_To_Hydrology'].astype('float32')
    elev = X_base['Elevation'].astype('float32')

    euclidean_hydro = ((h * h + v * v) ** 0.5).astype('float32')
    hydro_elevation = (elev - v).astype('float32')
    hydro_dist_diff = (h - v).astype('float32')

    hs_9am = X_base['Hillshade_9am'].astype('float32')
    hs_noon = X_base['Hillshade_Noon'].astype('float32')
    hs_3pm = X_base['Hillshade_3pm'].astype('float32')

    diff_3pm_9am = (hs_3pm - hs_9am).astype('float32')
    diff_3pm_noon = (hs_3pm - hs_noon).astype('float32')
    diff_noon_9am = (hs_noon - hs_9am).astype('float32')

    aspect_rad = X_base['Aspect'].astype('float32') * float(np.pi / 180.0)
    aspect_sin = aspect_rad.skb.apply_func(np.sin).astype('float32')
    aspect_cos = aspect_rad.skb.apply_func(np.cos).astype('float32')

    wilderness_sum = X_base[wilderness_cols].sum(axis=1).astype('int16')
    soil_sum = X_base[soil_cols].sum(axis=1).astype('int16')

    X_all = X_base.assign(
        Euclidean_Distance_To_Hydrology=euclidean_hydro,
        Hydrology_Elevation=hydro_elevation,
        Hydrology_Distance_Diff=hydro_dist_diff,
        Hillshade_Diff_3pm_9am=diff_3pm_9am,
        Hillshade_Diff_3pm_Noon=diff_3pm_noon,
        Hillshade_Diff_Noon_9am=diff_noon_9am,
        Aspect_Sin=aspect_sin,
        Aspect_Cos=aspect_cos,
        Wilderness_Area_Sum=wilderness_sum,
        Soil_Type_Sum=soil_sum,
    )

    cols_hydro = [
        'Euclidean_Distance_To_Hydrology',
        'Hydrology_Elevation',
        'Hydrology_Distance_Diff'
    ]
    cols_solar_aspect = [
        'Hillshade_Diff_3pm_9am',
        'Hillshade_Diff_3pm_Noon',
        'Hillshade_Diff_Noon_9am',
        'Aspect_Sin',
        'Aspect_Cos'
    ]
    cols_indicators = [
        'Wilderness_Area_Sum',
        'Soil_Type_Sum'
    ]

    cols_hydro_spatial = base_cols + cols_hydro
    cols_hydro_solar_aspect = base_cols + cols_hydro + cols_solar_aspect
    cols_full_terrain = base_cols + cols_hydro + cols_solar_aspect + cols_indicators

    feature_choice = skrub.choose_from(
        {
            'hydrology_spatial': cols_hydro_spatial,
            'hydro_solar_aspect': cols_hydro_solar_aspect,
            'full_terrain_indicators': cols_full_terrain,
        },
        name='feature_set'
    )
    X_features = X_all[feature_choice]

    base_lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=63,
        colsample_bytree=0.7,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=100,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model = SubsampledClassifier(
        estimator=base_lgbm,
        train_fraction=1.0,
        random_state=42
    )

    pred = X_features.skb.apply(model, y=y)
    out = {'pred': pred, 'scoring': scoring}
    if 'row_keys' in eval_dict:
        out['row_keys'] = eval_dict['row_keys']
    return out