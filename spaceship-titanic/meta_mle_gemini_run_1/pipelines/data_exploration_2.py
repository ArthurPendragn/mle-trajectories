import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skrub
from xgboost import XGBClassifier

TRAIN_PATH = '/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv'


class PartyFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        group_ids = df['PassengerId'].astype(str).str.slice(0, 4)
        cabins = df['Cabin'].fillna('Missing').astype(str)
        surnames = df['Name'].fillna('Missing').astype(str).str.split().str[-1]
        amenity_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        total_spends = df[amenity_cols].fillna(0.0).sum(axis=1)
        self.group_size_map_ = group_ids.value_counts().to_dict()
        self.cabin_occupants_map_ = cabins.value_counts().to_dict()
        self.surname_freq_map_ = surnames.value_counts().to_dict()
        df_stats = pd.DataFrame({'group_id': group_ids, 'spend': total_spends})
        group_agg = df_stats.groupby('group_id')['spend'].agg(['sum', 'mean'])
        self.group_sum_spend_map_ = group_agg['sum'].to_dict()
        self.group_mean_spend_map_ = group_agg['mean'].to_dict()
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        amenity_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        spend_sum = df[amenity_cols].fillna(0.0).sum(axis=1)
        zero_spend = (spend_sum == 0.0).astype(float)
        luxury_spend = df['RoomService'].fillna(0.0) + df['Spa'].fillna(0.0) + df['VRDeck'].fillna(0.0)
        log_total_spend = np.log1p(spend_sum)
        log_luxury_spend = np.log1p(luxury_spend)
        luxury_ratio = luxury_spend / (spend_sum + 1.0)
        room_service_ratio = df['RoomService'].fillna(0.0) / (spend_sum + 1.0)
        spa_ratio = df['Spa'].fillna(0.0) / (spend_sum + 1.0)
        vr_ratio = df['VRDeck'].fillna(0.0) / (spend_sum + 1.0)
        cabin_str = df['Cabin'].astype(str)
        cabin_parts = cabin_str.str.split('/', expand=True)
        deck = cabin_parts[0].where(df['Cabin'].notna(), 'Missing')
        side = cabin_parts[2].where(df['Cabin'].notna(), 'Missing')
        cabin_num = pd.to_numeric(cabin_parts[1], errors='coerce').fillna(-1.0)
        cabin_region = (cabin_num // 300.0).astype(float)
        group_ids = df['PassengerId'].astype(str).str.slice(0, 4)
        group_num = pd.to_numeric(df['PassengerId'].astype(str).str.slice(5), errors='coerce').fillna(1.0)
        cabins = df['Cabin'].fillna('Missing').astype(str)
        surnames = df['Name'].fillna('Missing').astype(str).str.split().str[-1]
        group_size = group_ids.map(self.group_size_map_).fillna(1.0).astype(float)
        is_alone = (group_size == 1.0).astype(float)
        group_total_spend = group_ids.map(self.group_sum_spend_map_).fillna(spend_sum).astype(float)
        group_mean_spend = group_ids.map(self.group_mean_spend_map_).fillna(spend_sum).astype(float)
        cabin_occupants = cabins.map(self.cabin_occupants_map_).fillna(1.0).astype(float)
        family_size = surnames.map(self.surname_freq_map_).fillna(1.0).astype(float)
        is_child = (df['Age'].fillna(999.0) <= 12.0).astype(float)
        home_planet = df['HomePlanet'].fillna('Missing').astype(str)
        cryo_sleep = df['CryoSleep'].fillna('Missing').astype(str)
        destination = df['Destination'].fillna('Missing').astype(str)
        vip = df['VIP'].fillna('Missing').astype(str)
        out_df = pd.DataFrame(
            {
                'HomePlanet': home_planet,
                'CryoSleep': cryo_sleep,
                'Destination': destination,
                'VIP': vip,
                'Deck': deck,
                'Side': side,
                'Age': df['Age'],
                'CabinNum': cabin_num,
                'CabinRegion': cabin_region,
                'CabinOccupants': cabin_occupants,
                'GroupNum': group_num,
                'GroupSize': group_size,
                'IsAlone': is_alone,
                'GroupTotalSpend': group_total_spend,
                'GroupMeanSpend': group_mean_spend,
                'FamilySize': family_size,
                'IsChild': is_child,
                'RoomService': df['RoomService'],
                'FoodCourt': df['FoodCourt'],
                'ShoppingMall': df['ShoppingMall'],
                'Spa': df['Spa'],
                'VRDeck': df['VRDeck'],
                'TotalSpend': spend_sum,
                'ZeroSpend': zero_spend,
                'LogTotalSpend': log_total_spend,
                'LuxurySpend': luxury_spend,
                'LogLuxurySpend': log_luxury_spend,
                'LuxuryRatio': luxury_ratio,
                'RoomServiceRatio': room_service_ratio,
                'SpaRatio': spa_ratio,
                'VRDeckRatio': vr_ratio,
            },
            index=df.index,
        )
        return out_df


class SafeCatBoostClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = 'classifier'

    def __init__(self, iterations=300, depth=4, learning_rate=0.05, random_seed=42, verbose=0):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
            verbose=self.verbose,
        )
        y_arr = np.asarray(y).astype(int)
        self.model_.fit(X, y_arr)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        preds = proba[:, 1] >= 0.5
        if hasattr(self, 'classes_') and self.classes_.dtype == bool:
            return preds.astype(bool)
        elif hasattr(self, 'classes_'):
            return preds.astype(self.classes_.dtype)
        return preds


class ThresholdSweepAnalyzer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        y_arr = np.asarray(y).astype(bool)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
        num_cols = [
            'Age', 'CabinNum', 'CabinRegion', 'CabinOccupants', 'GroupNum', 'GroupSize',
            'IsAlone', 'GroupTotalSpend', 'GroupMeanSpend', 'FamilySize', 'IsChild',
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend',
            'ZeroSpend', 'LogTotalSpend', 'LuxurySpend', 'LogLuxurySpend', 'LuxuryRatio',
            'RoomServiceRatio', 'SpaRatio', 'VRDeckRatio',
        ]
        preprocessor = Pipeline([
            ('party_fe', PartyFeatureEngineer()),
            ('prep', ColumnTransformer(transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ])),
        ])
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=4, eval_metric='logloss')),
                ('lgbm', LGBMClassifier(learning_rate=0.05, num_leaves=15, n_estimators=300, random_state=42, n_jobs=4, verbose=-1)),
            ],
            voting='soft',
        )

        oof_probs = np.zeros(len(y_arr))
        for train_idx, val_idx in cv.split(X, y_arr):
            X_tr, y_tr = X.iloc[train_idx], y_arr[train_idx]
            X_val = X.iloc[val_idx]
            pipe = Pipeline([('prep', preprocessor), ('model', ensemble)])
            pipe.fit(X_tr, y_tr)
            oof_probs[val_idx] = pipe.predict_proba(X_val)[:, 1]

        thresholds = np.round(np.arange(0.45, 0.585, 0.01), 2)
        records = []
        base_acc = float(accuracy_score(y_arr, oof_probs >= 0.50))
        for t in thresholds:
            pred_t = oof_probs >= t
            acc = float(accuracy_score(y_arr, pred_t))
            records.append({
                'threshold': float(t),
                'oof_accuracy': acc,
                'delta_vs_0_50': acc - base_acc,
                'predicted_positive_rate': float(np.mean(pred_t)),
            })
        self.result_df_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.result_df_


class CatBoostDiagnosticAnalyzer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        y_arr = np.asarray(y).astype(bool)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(cv.split(X, y_arr))
        X_tr, y_tr = X.iloc[train_idx], y_arr[train_idx]
        X_val, y_val = X.iloc[val_idx], y_arr[val_idx]

        cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
        num_cols = [
            'Age', 'CabinNum', 'CabinRegion', 'CabinOccupants', 'GroupNum', 'GroupSize',
            'IsAlone', 'GroupTotalSpend', 'GroupMeanSpend', 'FamilySize', 'IsChild',
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend',
            'ZeroSpend', 'LogTotalSpend', 'LuxurySpend', 'LogLuxurySpend', 'LuxuryRatio',
            'RoomServiceRatio', 'SpaRatio', 'VRDeckRatio',
        ]
        preprocessor = Pipeline([
            ('party_fe', PartyFeatureEngineer()),
            ('prep', ColumnTransformer(transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ])),
        ])
        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)

        raw_cb = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, random_seed=42, verbose=0)
        raw_cb.fit(X_tr_proc, y_tr)
        raw_pred = raw_cb.predict(X_val_proc)
        raw_proba = raw_cb.predict_proba(X_val_proc)

        raw_acc_status = 'ok'
        try:
            raw_acc = float(accuracy_score(y_val, raw_pred))
        except Exception as e:
            raw_acc = -1.0
            raw_acc_status = f'failed: {str(e)}'

        safe_cb = SafeCatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, random_seed=42, verbose=0)
        safe_cb.fit(X_tr_proc, y_tr)
        safe_pred = safe_cb.predict(X_val_proc)
        safe_acc = float(accuracy_score(y_val, safe_pred))

        diag_record = [{
            'raw_predict_shape': str(raw_pred.shape),
            'raw_predict_ndim': int(raw_pred.ndim),
            'raw_predict_dtype': str(raw_pred.dtype),
            'raw_proba_shape': str(raw_proba.shape),
            'raw_acc_status': raw_acc_status,
            'raw_accuracy': raw_acc,
            'safe_predict_shape': str(safe_pred.shape),
            'safe_predict_dtype': str(safe_pred.dtype),
            'safe_accuracy': safe_acc,
        }]
        self.result_df_ = pd.DataFrame(diag_record)
        return self

    def transform(self, X):
        return self.result_df_


class OrthogonalModelsBenchmark(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        y_arr = np.asarray(y).astype(bool)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
        num_cols = [
            'Age', 'CabinNum', 'CabinRegion', 'CabinOccupants', 'GroupNum', 'GroupSize',
            'IsAlone', 'GroupTotalSpend', 'GroupMeanSpend', 'FamilySize', 'IsChild',
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend',
            'ZeroSpend', 'LogTotalSpend', 'LuxurySpend', 'LogLuxurySpend', 'LuxuryRatio',
            'RoomServiceRatio', 'SpaRatio', 'VRDeckRatio',
        ]
        preprocessor = Pipeline([
            ('party_fe', PartyFeatureEngineer()),
            ('prep', ColumnTransformer(transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ])),
        ])

        models = {
            'XGBoost': XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=4, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(learning_rate=0.05, num_leaves=15, n_estimators=300, random_state=42, n_jobs=4, verbose=-1),
            'CatBoost': SafeCatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, random_seed=42, verbose=0),
            'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=15, learning_rate=0.05, random_state=42),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_split=5, random_state=42, n_jobs=4),
            'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42),
        }

        benchmarks = []
        for name, model in models.items():
            fold_accs = []
            for train_idx, val_idx in cv.split(X, y_arr):
                X_tr, y_tr = X.iloc[train_idx], y_arr[train_idx]
                X_val, y_val = X.iloc[val_idx], y_arr[val_idx]
                pipe = Pipeline([('prep', preprocessor), ('model', model)])
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_val)
                fold_accs.append(float(accuracy_score(y_val, preds)))
            benchmarks.append({
                'model': name,
                'mean_accuracy': float(np.mean(fold_accs)),
                'std_accuracy': float(np.std(fold_accs)),
                'fold_0': fold_accs[0],
                'fold_1': fold_accs[1],
                'fold_2': fold_accs[2],
                'fold_3': fold_accs[3],
                'fold_4': fold_accs[4],
            })
        self.result_df_ = pd.DataFrame(benchmarks)
        return self

    def transform(self, X):
        return self.result_df_


class ThreeWayEnsembleBenchmark(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        y_arr = np.asarray(y).astype(bool)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
        num_cols = [
            'Age', 'CabinNum', 'CabinRegion', 'CabinOccupants', 'GroupNum', 'GroupSize',
            'IsAlone', 'GroupTotalSpend', 'GroupMeanSpend', 'FamilySize', 'IsChild',
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend',
            'ZeroSpend', 'LogTotalSpend', 'LuxurySpend', 'LogLuxurySpend', 'LuxuryRatio',
            'RoomServiceRatio', 'SpaRatio', 'VRDeckRatio',
        ]
        preprocessor = Pipeline([
            ('party_fe', PartyFeatureEngineer()),
            ('prep', ColumnTransformer(transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ])),
        ])
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=4, eval_metric='logloss')),
                ('lgbm', LGBMClassifier(learning_rate=0.05, num_leaves=15, n_estimators=300, random_state=42, n_jobs=4, verbose=-1)),
                ('cat', SafeCatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, random_seed=42, verbose=0)),
            ],
            voting='soft',
        )

        oof_probs = np.zeros(len(y_arr))
        fold_scores_0_5 = []
        for train_idx, val_idx in cv.split(X, y_arr):
            X_tr, y_tr = X.iloc[train_idx], y_arr[train_idx]
            X_val, y_val = X.iloc[val_idx], y_arr[val_idx]
            pipe = Pipeline([('prep', preprocessor), ('model', ensemble)])
            pipe.fit(X_tr, y_tr)
            val_probs = pipe.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = val_probs
            fold_scores_0_5.append(float(accuracy_score(y_val, val_probs >= 0.50)))

        records = [{
            'ensemble': 'XGB_LGB_CatBoost_soft',
            'mean_accuracy_0_5': float(np.mean(fold_scores_0_5)),
            'std_accuracy_0_5': float(np.std(fold_scores_0_5)),
            'fold_0': fold_scores_0_5[0],
            'fold_1': fold_scores_0_5[1],
            'fold_2': fold_scores_0_5[2],
            'fold_3': fold_scores_0_5[3],
            'fold_4': fold_scores_0_5[4],
        }]
        for t in np.round(np.arange(0.46, 0.565, 0.01), 2):
            acc_t = float(accuracy_score(y_arr, oof_probs >= t))
            records.append({
                'ensemble': f'threshold_{t}',
                'mean_accuracy_0_5': acc_t,
                'std_accuracy_0_5': 0.0,
                'fold_0': float(t),
                'fold_1': float(np.mean(oof_probs >= t)),
                'fold_2': acc_t - float(np.mean(fold_scores_0_5)),
                'fold_3': 0.0,
                'fold_4': 0.0,
            })
        self.result_df_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.result_df_


def build():
    df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    X = df.drop(columns=['Transported'])
    y = df['Transported'].astype(bool)

    threshold_sweep_op = X.skb.apply(ThresholdSweepAnalyzer(), y=y)
    catboost_diagnostics_op = X.skb.apply(CatBoostDiagnosticAnalyzer(), y=y)
    orthogonal_benchmark_op = X.skb.apply(OrthogonalModelsBenchmark(), y=y)
    three_way_ensemble_op = X.skb.apply(ThreeWayEnsembleBenchmark(), y=y)

    return {
        'threshold_sweep': threshold_sweep_op,
        'catboost_diagnostics': catboost_diagnostics_op,
        'orthogonal_benchmark': orthogonal_benchmark_op,
        'three_way_ensemble': three_way_ensemble_op,
    }