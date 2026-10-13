import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
import skrub

TRAIN_PATH = '/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv'


def locked_setup_helper():
    raw = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X = raw.drop(columns=['Id', 'Cover_Type']).skb.mark_as_X(cv=cv, split_kwargs={})
    y = raw['Cover_Type'].skb.mark_as_y()
    row_keys = raw['Id']
    return {
        'X': X,
        'y': y,
        'scoring': 'accuracy',
        'row_keys': row_keys,
        'audit': {'target_counts': raw['Cover_Type'].value_counts().to_frame().reset_index()},
    }


def locked_setup_entry():
    return locked_setup_helper()


def build_evaluation():
    return locked_setup_entry()


class DomainFeatureLGBMClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        feature_set='all',
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=63,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ):
        self.feature_set = feature_set
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _transform(self, X):
        if self.feature_set == 'baseline':
            return X
        new_cols = {}
        if self.feature_set in ('physical', 'all'):
            h_hydro = X['Horizontal_Distance_To_Hydrology'].to_numpy(dtype=np.float32)
            v_hydro = X['Vertical_Distance_To_Hydrology'].to_numpy(dtype=np.float32)
            elevation = X['Elevation'].to_numpy(dtype=np.float32)
            aspect = X['Aspect'].to_numpy(dtype=np.float32)
            h9 = X['Hillshade_9am'].to_numpy(dtype=np.float32)
            h12 = X['Hillshade_Noon'].to_numpy(dtype=np.float32)
            h15 = X['Hillshade_3pm'].to_numpy(dtype=np.float32)
            new_cols['Euclidean_Distance_To_Hydrology'] = np.hypot(h_hydro, v_hydro)
            new_cols['Water_Elevation'] = elevation - v_hydro
            rad_aspect = np.radians(aspect % 360.0)
            new_cols['Aspect_Sin'] = np.sin(rad_aspect)
            new_cols['Aspect_Cos'] = np.cos(rad_aspect)
            new_cols['Hillshade_9am_minus_Noon'] = h9 - h12
            new_cols['Hillshade_Noon_minus_3pm'] = h12 - h15
            new_cols['Hillshade_9am_minus_3pm'] = h9 - h15
            new_cols['Hillshade_Mean'] = (h9 + h12 + h15) / 3.0
        if self.feature_set == 'all':
            h_hydro = X['Horizontal_Distance_To_Hydrology'].to_numpy(dtype=np.float32)
            h_road = X['Horizontal_Distance_To_Roadways'].to_numpy(dtype=np.float32)
            h_fire = X['Horizontal_Distance_To_Fire_Points'].to_numpy(dtype=np.float32)
            new_cols['Distance_Hydro_plus_Road'] = h_hydro + h_road
            new_cols['Distance_Hydro_plus_Fire'] = h_hydro + h_fire
            new_cols['Distance_Road_plus_Fire'] = h_road + h_fire
            new_cols['Distance_Hydro_minus_Road'] = np.abs(h_hydro - h_road)
            new_cols['Distance_Hydro_minus_Fire'] = np.abs(h_hydro - h_fire)
            new_cols['Distance_Road_minus_Fire'] = np.abs(h_road - h_fire)
            wild_cols = [c for c in X.columns if c.startswith('Wilderness_Area')]
            if wild_cols:
                new_cols['Wilderness_Count'] = X[wild_cols].sum(axis=1).to_numpy(dtype=np.int32)
            soil_cols = [c for c in X.columns if c.startswith('Soil_Type')]
            if soil_cols:
                new_cols['Soil_Count'] = X[soil_cols].sum(axis=1).to_numpy(dtype=np.int32)
        new_df = pd.DataFrame(new_cols, index=X.index)
        return pd.concat([X, new_df], axis=1)

    def fit(self, X, y):
        X_trans = self._transform(X)
        self.model_ = lgb.LGBMClassifier(
            objective='multiclass',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        self.model_.fit(X_trans, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        X_trans = self._transform(X)
        return self.model_.predict(X_trans)

    def predict_proba(self, X):
        X_trans = self._transform(X)
        return self.model_.predict_proba(X_trans)


def build():
    evaluation = build_evaluation()
    X = evaluation['X'].drop(columns=['Soil_Type7', 'Soil_Type15'])
    y = evaluation['y']

    models = {
        'capacity_127_200': DomainFeatureLGBMClassifier(
            feature_set='all',
            n_estimators=200,
            learning_rate=0.08,
            num_leaves=127,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        'capacity_255_200': DomainFeatureLGBMClassifier(
            feature_set='all',
            n_estimators=200,
            learning_rate=0.08,
            num_leaves=255,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        'capacity_255_300': DomainFeatureLGBMClassifier(
            feature_set='all',
            n_estimators=300,
            learning_rate=0.06,
            num_leaves=255,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    model = skrub.choose_from(models, name='model_capacity')
    pred = X.skb.apply(model, y=y)
    return {
        'pred': pred,
        'scoring': evaluation['scoring'],
        'row_keys': evaluation['row_keys'],
    }