import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skrub
from xgboost import XGBClassifier

TRAIN_PATH = '/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv'


class PartyFeatureEngineer(BaseEstimator, TransformerMixin):
    """Fits party and cabin context statistics on training rows (group spend,

    group sizes, cabin occupancy, surname frequencies) and maps them cleanly
    at transform time without data leakage.
    """

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
        luxury_spend = (
            df['RoomService'].fillna(0.0)
            + df['Spa'].fillna(0.0)
            + df['VRDeck'].fillna(0.0)
        )
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
        group_num = pd.to_numeric(
            df['PassengerId'].astype(str).str.slice(5), errors='coerce'
        ).fillna(1.0)
        cabins = df['Cabin'].fillna('Missing').astype(str)
        surnames = df['Name'].fillna('Missing').astype(str).str.split().str[-1]

        group_size = (
            group_ids.map(self.group_size_map_).fillna(1.0).astype(float)
        )
        is_alone = (group_size == 1.0).astype(float)
        group_total_spend = (
            group_ids.map(self.group_sum_spend_map_)
            .fillna(spend_sum)
            .astype(float)
        )
        group_mean_spend = (
            group_ids.map(self.group_mean_spend_map_)
            .fillna(spend_sum)
            .astype(float)
        )
        cabin_occupants = (
            cabins.map(self.cabin_occupants_map_).fillna(1.0).astype(float)
        )
        family_size = (
            surnames.map(self.surname_freq_map_).fillna(1.0).astype(float)
        )
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


class OOFConsensusAnalyzer(BaseEstimator, TransformerMixin):
    """Executes 5-fold StratifiedKFold validation to reconstruct out-of-fold

    predictions across model families and evaluates consensus aggregations,
    fold 4 error profiles, and intermediate decile calibration.
    """

    def fit(self, X, y):
        X_df = pd.DataFrame(X).copy().reset_index(drop=True)
        y_arr = np.array(y, dtype=bool)
        n_samples = len(X_df)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        oof_lgb = np.zeros(n_samples, dtype=float)
        oof_cat = np.zeros(n_samples, dtype=float)
        oof_xgb = np.zeros(n_samples, dtype=float)
        oof_hgb = np.zeros(n_samples, dtype=float)
        oof_fold = np.zeros(n_samples, dtype=int)

        cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
        num_cols = [
            'Age',
            'CabinNum',
            'CabinRegion',
            'CabinOccupants',
            'GroupNum',
            'GroupSize',
            'IsAlone',
            'GroupTotalSpend',
            'GroupMeanSpend',
            'FamilySize',
            'IsChild',
            'RoomService',
            'FoodCourt',
            'ShoppingMall',
            'Spa',
            'VRDeck',
            'TotalSpend',
            'ZeroSpend',
            'LogTotalSpend',
            'LuxurySpend',
            'LogLuxurySpend',
            'LuxuryRatio',
            'RoomServiceRatio',
            'SpaRatio',
            'VRDeckRatio',
        ]

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_df, y_arr)):
            oof_fold[val_idx] = fold

            fe = PartyFeatureEngineer()
            X_tr_fe = fe.fit_transform(X_df.iloc[train_idx])
            X_val_fe = fe.transform(X_df.iloc[val_idx])

            cat_tf = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            num_tf = Pipeline(
                [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]
            )
            prep = ColumnTransformer(
                transformers=[
                    ('cat', cat_tf, cat_cols),
                    ('num', num_tf, num_cols),
                ]
            )

            X_tr_prep = prep.fit_transform(X_tr_fe)
            X_val_prep = prep.transform(X_val_fe)
            y_tr = y_arr[train_idx]

            clf_lgb = LGBMClassifier(
                learning_rate=0.05,
                num_leaves=15,
                n_estimators=300,
                random_state=42,
                n_jobs=4,
                verbose=-1,
            )
            clf_lgb.fit(X_tr_prep, y_tr)
            oof_lgb[val_idx] = clf_lgb.predict_proba(X_val_prep)[:, 1]

            clf_cat = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=5,
                random_seed=42,
                thread_count=4,
                verbose=0,
                allow_writing_files=False,
            )
            clf_cat.fit(X_tr_prep, y_tr)
            oof_cat[val_idx] = clf_cat.predict_proba(X_val_prep)[:, 1]

            clf_xgb = XGBClassifier(
                max_depth=4,
                learning_rate=0.05,
                n_estimators=300,
                random_state=42,
                n_jobs=4,
                eval_metric='logloss',
            )
            clf_xgb.fit(X_tr_prep, y_tr)
            oof_xgb[val_idx] = clf_xgb.predict_proba(X_val_prep)[:, 1]

            clf_hgb = HistGradientBoostingClassifier(
                max_iter=300,
                max_leaf_nodes=15,
                learning_rate=0.05,
                random_state=42,
            )
            clf_hgb.fit(X_tr_prep, y_tr)
            oof_hgb[val_idx] = clf_hgb.predict_proba(X_val_prep)[:, 1]

        # Model correlation matrix
        corr_matrix = pd.DataFrame(
            {
                'LightGBM': oof_lgb,
                'CatBoost': oof_cat,
                'XGBoost': oof_xgb,
                'HistGB': oof_hgb,
            }
        ).corr()
        self.corr_df_ = corr_matrix.reset_index().rename(
            columns={'index': 'model'}
        )

        # Parent weighted ensemble
        parent_oof = (1.0 * oof_lgb + 2.0 * oof_cat) / 3.0

        # Consensus variants calculation
        rank_lgb = np.zeros(n_samples)
        rank_cat = np.zeros(n_samples)
        for fold in range(5):
            idx = oof_fold == fold
            rank_lgb[idx] = (
                pd.Series(oof_lgb[idx]).rank(pct=True).to_numpy()
            )
            rank_cat[idx] = (
                pd.Series(oof_cat[idx]).rank(pct=True).to_numpy()
            )

        eps = 1e-6
        logit_lgb = np.log(
            np.clip(oof_lgb, eps, 1 - eps) / (1 - np.clip(oof_lgb, eps, 1 - eps))
        )
        logit_cat = np.log(
            np.clip(oof_cat, eps, 1 - eps) / (1 - np.clip(oof_cat, eps, 1 - eps))
        )
        logit_consensus = 1.0 / (
            1.0 + np.exp(-(logit_lgb + 2.0 * logit_cat) / 3.0)
        )

        consensus_dict = {
            'LightGBM_standalone': oof_lgb,
            'CatBoost_standalone': oof_cat,
            'XGBoost_standalone': oof_xgb,
            'HistGB_standalone': oof_hgb,
            'LGB_Cat_equal_1_1': (oof_lgb + oof_cat) / 2.0,
            'LGB_Cat_parent_1_2': parent_oof,
            'LGB_Cat_XGB_2_3_1': (
                2.0 * oof_lgb + 3.0 * oof_cat + 1.0 * oof_xgb
            )
            / 6.0,
            'LGB_Cat_HGB_2_3_1': (
                2.0 * oof_lgb + 3.0 * oof_cat + 1.0 * oof_hgb
            )
            / 6.0,
            'Power_p0.5': ((oof_lgb**0.5 + 2.0 * oof_cat**0.5) / 3.0) ** 2.0,
            'Power_p1.5': ((oof_lgb**1.5 + 2.0 * oof_cat**1.5) / 3.0)
            ** (1.0 / 1.5),
            'Power_p2.0': ((oof_lgb**2.0 + 2.0 * oof_cat**2.0) / 3.0)
            ** (1.0 / 2.0),
            'Power_p3.0': ((oof_lgb**3.0 + 2.0 * oof_cat**3.0) / 3.0)
            ** (1.0 / 3.0),
            'Percentile_Rank_1_2': (rank_lgb + 2.0 * rank_cat) / 3.0,
            'Logit_Consensus_1_2': logit_consensus,
        }

        consensus_rows = []
        for name, preds in consensus_dict.items():
            f_accs = []
            for fold in range(5):
                idx = oof_fold == fold
                f_accs.append(accuracy_score(y_arr[idx], preds[idx] >= 0.50))
            consensus_rows.append(
                {
                    'method': name,
                    'fold_0': f_accs[0],
                    'fold_1': f_accs[1],
                    'fold_2': f_accs[2],
                    'fold_3': f_accs[3],
                    'fold_4': f_accs[4],
                    'cv_mean': np.mean(f_accs),
                    'cv_std': np.std(f_accs),
                }
            )
        self.consensus_df_ = pd.DataFrame(consensus_rows).sort_values(
            by='cv_mean', ascending=False
        )

        # Question 1: Fold 4 vs Folds 0-3 diagnostics
        f4_mask = oof_fold == 4
        f03_mask = oof_fold != 4

        diag_rows = []
        # Threshold sensitivity comparison
        for th in [0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54]:
            acc_f4 = accuracy_score(y_arr[f4_mask], parent_oof[f4_mask] >= th)
            acc_f03 = accuracy_score(y_arr[f03_mask], parent_oof[f03_mask] >= th)
            diag_rows.append(
                {
                    'diagnostic_type': 'threshold_sweep',
                    'category': f'threshold_{th:.2f}',
                    'fold_4_metric': acc_f4,
                    'folds_0_3_metric': acc_f03,
                    'gap_f4_minus_f03': acc_f4 - acc_f03,
                }
            )

        # Subgroup error breakdown
        # Compute misclassification flag
        is_err = (parent_oof >= 0.50) != y_arr
        subgroups = [
            ('CryoSleep_True', X_df['CryoSleep'] == True),
            ('CryoSleep_False', X_df['CryoSleep'] == False),
            ('CryoSleep_Missing', X_df['CryoSleep'].isna()),
            ('HomePlanet_Earth', X_df['HomePlanet'] == 'Earth'),
            ('HomePlanet_Europa', X_df['HomePlanet'] == 'Europa'),
            ('HomePlanet_Mars', X_df['HomePlanet'] == 'Mars'),
            ('Deck_B', X_df['Cabin'].fillna('').str.startswith('B')),
            ('Deck_C', X_df['Cabin'].fillna('').str.startswith('C')),
            ('Deck_E', X_df['Cabin'].fillna('').str.startswith('E')),
            ('Deck_F', X_df['Cabin'].fillna('').str.startswith('F')),
            ('Deck_G', X_df['Cabin'].fillna('').str.startswith('G')),
            ('Side_Port', X_df['Cabin'].fillna('').str.endswith('P')),
            ('Side_Starboard', X_df['Cabin'].fillna('').str.endswith('S')),
            (
                'ZeroSpend_True',
                X_df[
                    ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
                ]
                .fillna(0)
                .sum(axis=1)
                == 0,
            ),
            (
                'ZeroSpend_False',
                X_df[
                    ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
                ]
                .fillna(0)
                .sum(axis=1)
                > 0,
            ),
            ('Child_Age_le_12', X_df['Age'].fillna(99) <= 12),
        ]
        for name, mask in subgroups:
            f4_sub = f4_mask & mask
            f03_sub = f03_mask & mask
            err_f4 = np.mean(is_err[f4_sub]) if f4_sub.sum() > 0 else 0.0
            err_f03 = np.mean(is_err[f03_sub]) if f03_sub.sum() > 0 else 0.0
            diag_rows.append(
                {
                    'diagnostic_type': 'subgroup_error_rate',
                    'category': name,
                    'fold_4_metric': err_f4,
                    'folds_0_3_metric': err_f03,
                    'gap_f4_minus_f03': err_f4 - err_f03,
                }
            )
        self.fold4_df_ = pd.DataFrame(diag_rows)

        # Question 3: Decile calibration across probability spectrum
        decile_rows = []
        for model_name, preds in [
            ('Parent_Weighted_LGB_Cat', parent_oof),
            ('Logit_Consensus', logit_consensus),
            ('Rank_Consensus', (rank_lgb + 2.0 * rank_cat) / 3.0),
        ]:
            decile_bins = pd.qcut(
                preds, q=10, labels=False, duplicates='drop'
            )
            for d in range(10):
                d_mask = decile_bins == d
                if d_mask.sum() > 0:
                    cnt = int(d_mask.sum())
                    pred_mean = float(np.mean(preds[d_mask]))
                    pos_rate = float(np.mean(y_arr[d_mask]))
                    gap = pred_mean - pos_rate
                    decile_rows.append(
                        {
                            'model': model_name,
                            'decile': d,
                            'count': cnt,
                            'mean_pred_prob': pred_mean,
                            'empirical_pos_rate': pos_rate,
                            'overconfidence_gap': gap,
                        }
                    )
        self.decile_df_ = pd.DataFrame(decile_rows)
        return self

    def transform(self, X):
        return {
            'fold4_diagnostics': self.fold4_df_,
            'consensus_benchmark': self.consensus_df_,
            'decile_calibration': self.decile_df_,
            'model_correlations': self.corr_df_,
        }

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def build():
    train_df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    X = train_df.drop(columns=['Transported'])
    y = train_df['Transported'].astype(bool)

    parts = X.skb.apply(OOFConsensusAnalyzer(), y=y, no_wrap=True)

    return {
        'fold4_diagnostics': parts['fold4_diagnostics'],
        'consensus_benchmark': parts['consensus_benchmark'],
        'decile_calibration': parts['decile_calibration'],
        'model_correlations': parts['model_correlations'],
    }