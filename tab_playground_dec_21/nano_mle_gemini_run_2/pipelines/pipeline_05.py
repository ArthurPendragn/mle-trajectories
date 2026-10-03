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
    X = raw.drop(columns=['Id', 'Cover_Type']).skb.mark_as_X(
        cv=cv, split_kwargs={}
    )
    y = raw['Cover_Type'].skb.mark_as_y()
    row_keys = raw['Id']
    return {
        'X': X,
        'y': y,
        'scoring': 'accuracy',
        'row_keys': row_keys,
        'audit': {
            'target_counts': (
                raw['Cover_Type'].value_counts().to_frame().reset_index()
            )
        },
    }


def locked_setup_entry():
    return locked_setup_helper()


def build_evaluation():
    return locked_setup_entry()


class DomainFeatureLGBMClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        feature_set='parent_all',
        n_estimators=300,
        learning_rate=0.06,
        num_leaves=255,
        colsample_bytree=0.8,
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
        new_cols = {}

        # 1. Base physical and distance interaction features
        h_hydro = X['Horizontal_Distance_To_Hydrology'].to_numpy(
            dtype=np.float32
        )
        v_hydro = X['Vertical_Distance_To_Hydrology'].to_numpy(dtype=np.float32)
        elevation = X['Elevation'].to_numpy(dtype=np.float32)
        aspect = X['Aspect'].to_numpy(dtype=np.float32)
        h9 = X['Hillshade_9am'].to_numpy(dtype=np.float32)
        h12 = X['Hillshade_Noon'].to_numpy(dtype=np.float32)
        h15 = X['Hillshade_3pm'].to_numpy(dtype=np.float32)
        h_road = X['Horizontal_Distance_To_Roadways'].to_numpy(dtype=np.float32)
        h_fire = X['Horizontal_Distance_To_Fire_Points'].to_numpy(
            dtype=np.float32
        )

        new_cols['Euclidean_Distance_To_Hydrology'] = np.hypot(h_hydro, v_hydro)
        new_cols['Water_Elevation'] = elevation - v_hydro
        rad_aspect = np.radians(aspect % 360.0)
        new_cols['Aspect_Sin'] = np.sin(rad_aspect)
        new_cols['Aspect_Cos'] = np.cos(rad_aspect)
        new_cols['Hillshade_9am_minus_Noon'] = h9 - h12
        new_cols['Hillshade_Noon_minus_3pm'] = h12 - h15
        new_cols['Hillshade_9am_minus_3pm'] = h9 - h15
        new_cols['Hillshade_Mean'] = (h9 + h12 + h15) / 3.0

        new_cols['Distance_Hydro_plus_Road'] = h_hydro + h_road
        new_cols['Distance_Hydro_plus_Fire'] = h_hydro + h_fire
        new_cols['Distance_Road_plus_Fire'] = h_road + h_fire
        new_cols['Distance_Hydro_minus_Road'] = np.abs(h_hydro - h_road)
        new_cols['Distance_Hydro_minus_Fire'] = np.abs(h_hydro - h_fire)
        new_cols['Distance_Road_minus_Fire'] = np.abs(h_road - h_fire)

        wild_cols = [
            f'Wilderness_Area{i}'
            for i in range(1, 5)
            if f'Wilderness_Area{i}' in X.columns
        ]
        if wild_cols:
            new_cols['Wilderness_Count'] = (
                X[wild_cols].sum(axis=1).to_numpy(dtype=np.int32)
            )

        soil_cols = [
            f'Soil_Type{i}'
            for i in range(1, 41)
            if f'Soil_Type{i}' in X.columns
        ]
        if soil_cols:
            new_cols['Soil_Count'] = (
                X[soil_cols].sum(axis=1).to_numpy(dtype=np.int32)
            )

        # 2. Categorical indices and solar composite features
        if self.feature_set in ('categorical_and_solar', 'full_engineered'):
            new_cols['Aspect_Mod'] = aspect % 360.0
            h_min = np.minimum(np.minimum(h9, h12), h15)
            h_max = np.maximum(np.maximum(h9, h12), h15)
            new_cols['Hillshade_Min'] = h_min
            new_cols['Hillshade_Max'] = h_max
            new_cols['Hillshade_Range'] = h_max - h_min

            if wild_cols:
                wild_mat = X[wild_cols].to_numpy(dtype=np.int32)
                wild_ids = np.array(
                    [int(c.replace('Wilderness_Area', '')) for c in wild_cols],
                    dtype=np.int32,
                )
                has_wild = wild_mat.max(axis=1) > 0
                new_cols['Wilderness_Type'] = np.where(
                    has_wild, wild_ids[wild_mat.argmax(axis=1)], 0
                ).astype(np.int32)

            if soil_cols:
                soil_mat = X[soil_cols].to_numpy(dtype=np.int32)
                soil_ids = np.array(
                    [int(c.replace('Soil_Type', '')) for c in soil_cols],
                    dtype=np.int32,
                )
                has_soil = soil_mat.max(axis=1) > 0
                new_cols['Soil_Type_Index'] = np.where(
                    has_soil, soil_ids[soil_mat.argmax(axis=1)], 0
                ).astype(np.int32)

        # 3. Multi-distance infrastructure summary and smoothed ratios
        if self.feature_set == 'full_engineered':
            new_cols['Distance_Amenity_Mean'] = (
                h_hydro + h_road + h_fire
            ) / 3.0
            new_cols['Distance_Amenity_Min'] = np.minimum(
                np.minimum(h_hydro, h_road), h_fire
            )
            new_cols['Distance_Amenity_Max'] = np.maximum(
                np.maximum(h_hydro, h_road), h_fire
            )
            new_cols['Distance_Hydro_Road_Ratio'] = (h_hydro + 500.0) / (
                h_road + 500.0
            )
            new_cols['Distance_Hydro_Fire_Ratio'] = (h_hydro + 500.0) / (
                h_fire + 500.0
            )
            new_cols['Distance_Road_Fire_Ratio'] = (h_road + 500.0) / (
                h_fire + 500.0
            )

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
        'parent_all': DomainFeatureLGBMClassifier(
            feature_set='parent_all',
            n_estimators=300,
            learning_rate=0.06,
            num_leaves=255,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        'categorical_and_solar': DomainFeatureLGBMClassifier(
            feature_set='categorical_and_solar',
            n_estimators=300,
            learning_rate=0.06,
            num_leaves=255,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        'full_engineered': DomainFeatureLGBMClassifier(
            feature_set='full_engineered',
            n_estimators=300,
            learning_rate=0.06,
            num_leaves=255,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    model = skrub.choose_from(models, name='feature_ablation')
    pred = X.skb.apply(model, y=y)

    return {
        'pred': pred,
        'scoring': evaluation['scoring'],
        'row_keys': evaluation['row_keys'],
    }