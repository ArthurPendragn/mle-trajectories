TRAIN_PATH = '/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv'

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

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

        cryo_series = df['CryoSleep'].copy()
        cryo_series = cryo_series.mask(cryo_series.isna() & (total_spends > 0.0), False)
        cryo_bool = (cryo_series == True) | (cryo_series == 'True')
        zero_spend_bool = total_spends == 0.0

        df_group_behave = pd.DataFrame({
            'group_id': group_ids,
            'cryo': cryo_bool.astype(float),
            'zero': zero_spend_bool.astype(float)
        })
        group_behave_agg = df_group_behave.groupby('group_id').agg(
            cryo_ratio=('cryo', 'mean'),
            any_cryo=('cryo', 'max'),
            zero_ratio=('zero', 'mean')
        )
        self.group_cryo_ratio_map_ = group_behave_agg['cryo_ratio'].to_dict()
        self.group_any_cryo_map_ = group_behave_agg['any_cryo'].to_dict()
        self.group_zero_spend_map_ = group_behave_agg['zero_ratio'].to_dict()

        hp_valid = df[df['HomePlanet'].notna() & (df['HomePlanet'] != 'Missing')]
        hp_groups = hp_valid.groupby(hp_valid['PassengerId'].astype(str).str.slice(0, 4))['HomePlanet'].first()
        self.group_to_hp_map_ = hp_groups.to_dict()

        valid_surnames = surnames[(surnames != 'Missing') & df['HomePlanet'].notna() & (df['HomePlanet'] != 'Missing')]
        valid_hp = df.loc[valid_surnames.index, 'HomePlanet']
        df_surname_hp = pd.DataFrame({'surname': valid_surnames, 'HomePlanet': valid_hp})
        surname_groups = df_surname_hp.groupby('surname')['HomePlanet'].first()
        self.surname_to_hp_map_ = surname_groups.to_dict()

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

        local_counts = group_ids.value_counts().to_dict()
        group_size = group_ids.map(self.group_size_map_).fillna(group_ids.map(local_counts)).fillna(1.0).astype(float)
        is_alone = (group_size == 1.0).astype(float)

        group_total_spend = group_ids.map(self.group_sum_spend_map_).fillna(spend_sum).astype(float)
        group_mean_spend = group_ids.map(self.group_mean_spend_map_).fillna(spend_sum).astype(float)
        cabin_occupants = cabins.map(self.cabin_occupants_map_).fillna(1.0).astype(float)
        family_size = surnames.map(self.surname_freq_map_).fillna(1.0).astype(float)

        age_val = df['Age']
        is_child = (age_val.fillna(999.0) <= 12.0).astype(float)

        destination = df['Destination'].fillna('Missing').astype(str)
        vip = df['VIP'].fillna('Missing').astype(str)

        hp_series = df['HomePlanet'].copy()
        hp_series = hp_series.fillna(group_ids.map(self.group_to_hp_map_))
        hp_series = hp_series.fillna(surnames.map(self.surname_to_hp_map_))
        home_planet = hp_series.fillna('Missing').astype(str)

        cryo_series = df['CryoSleep'].copy()
        cryo_series = cryo_series.mask(cryo_series.isna() & (spend_sum > 0.0), False)
        cryo_sleep = cryo_series.fillna('Missing').astype(str)
        cryo_bool = (cryo_series == True) | (cryo_series == 'True')

        df_local_behave = pd.DataFrame({
            'group_id': group_ids,
            'cryo': cryo_bool.astype(float),
            'zero': zero_spend
        })
        local_behave_agg = df_local_behave.groupby('group_id').agg(
            cryo_ratio=('cryo', 'mean'),
            any_cryo=('cryo', 'max'),
            zero_ratio=('zero', 'mean')
        )
        local_cryo_ratio = local_behave_agg['cryo_ratio'].to_dict()
        local_any_cryo = local_behave_agg['any_cryo'].to_dict()
        local_zero_ratio = local_behave_agg['zero_ratio'].to_dict()

        group_cryo_ratio = group_ids.map(self.group_cryo_ratio_map_).fillna(group_ids.map(local_cryo_ratio)).fillna(cryo_bool.astype(float)).astype(float)
        group_any_cryo = group_ids.map(self.group_any_cryo_map_).fillna(group_ids.map(local_any_cryo)).fillna(cryo_bool.astype(float)).astype(float)
        group_zero_spend_ratio = group_ids.map(self.group_zero_spend_map_).fillna(group_ids.map(local_zero_ratio)).fillna(zero_spend).astype(float)

        earth_destination = np.where(home_planet == 'Earth', 'Earth_' + destination, 'Other')
        earth_side = np.where(home_planet == 'Earth', 'Earth_' + side, 'Other')
        earth_minor_pso = ((home_planet == 'Earth') & (age_val.fillna(999.0) <= 12.0) & (destination == 'PSO J318.5-22')).astype(float)
        earth_noncryo_zerospend_side = ((home_planet == 'Earth') & (cryo_sleep == 'False') & (spend_sum == 0.0) & (side == 'S')).astype(float)

        out_data = {
            'HomePlanet': home_planet,
            'CryoSleep': cryo_sleep,
            'Destination': destination,
            'VIP': vip,
            'Deck': deck,
            'Side': side,
            'Age': age_val,
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
            'DeckSide': deck + '/' + side,
            'DeckGStarboardHotspot': ((deck == 'G') & (side == 'S') & (cabin_num >= 750.0) & (cabin_num <= 1200.0)).astype(float),
            'HomePlanet_Cryo': home_planet + '_' + cryo_sleep,
            'IsEarthChild': ((home_planet == 'Earth') & (age_val.fillna(999.0) <= 12.0)).astype(float),
            'Earth_NonCryo_ZeroSpend': ((home_planet == 'Earth') & (cryo_sleep == 'False') & (spend_sum == 0.0)).astype(float),
            'HomePlanet_Deck': home_planet + '_' + deck,
            'GroupCryoRatio': group_cryo_ratio,
            'GroupAnyCryo': group_any_cryo,
            'GroupZeroSpendRatio': group_zero_spend_ratio,
            'Earth_Destination': earth_destination,
            'Earth_Side': earth_side,
            'Earth_Minor_PSO': earth_minor_pso,
            'Earth_NonCryo_ZeroSpend_Side': earth_noncryo_zerospend_side
        }
        return pd.DataFrame(out_data, index=df.index)

class ConsensusDiagnosticsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.results_

    def fit_transform(self, X, y):
        df_raw = pd.DataFrame(X).copy()
        y_true = np.asarray(y, dtype=bool)
        n_rows = len(df_raw)

        base_cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide', 'HomePlanet_Cryo', 'HomePlanet_Deck']
        base_num_cols = [
            'Age', 'CabinNum', 'CabinRegion', 'CabinOccupants', 'GroupNum', 'GroupSize', 'IsAlone',
            'GroupTotalSpend', 'GroupMeanSpend', 'FamilySize', 'IsChild', 'RoomService', 'FoodCourt',
            'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'ZeroSpend', 'LogTotalSpend', 'LuxurySpend',
            'LogLuxurySpend', 'LuxuryRatio', 'RoomServiceRatio', 'SpaRatio', 'VRDeckRatio',
            'DeckGStarboardHotspot', 'IsEarthChild', 'Earth_NonCryo_ZeroSpend'
        ]
        group_num_cols = ['GroupCryoRatio', 'GroupAnyCryo', 'GroupZeroSpendRatio']

        cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        prep_transformer = ColumnTransformer(transformers=[
            ('cat', cat_transformer, base_cat_cols),
            ('num', num_transformer, base_num_cols + group_num_cols)
        ])

        fe = PartyFeatureEngineer()

        lgb_clf = LGBMClassifier(learning_rate=0.05, num_leaves=15, n_estimators=300, random_state=42, n_jobs=4, verbose=-1)
        cat_clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, random_seed=42, thread_count=4, verbose=0, allow_writing_files=False)
        hgb_clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, random_state=42)
        xgb_clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4, eval_metric='logloss')

        weights = [2.0, 3.0, 1.0, 1.0]
        power = 2.0

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(n_rows, dtype=float)
        oof_folds = np.zeros(n_rows, dtype=int)

        # Cross-validation fitting to obtain OOF consensus probabilities
        for fold, (train_idx, val_idx) in enumerate(cv.split(df_raw, y_true)):
            X_tr_raw = df_raw.iloc[train_idx]
            y_tr = y_true[train_idx]
            X_val_raw = df_raw.iloc[val_idx]

            fe_fold = clone(fe).fit(X_tr_raw)
            X_tr_fe = fe_fold.transform(X_tr_raw)
            X_val_fe = fe_fold.transform(X_val_raw)

            prep_fold = clone(prep_transformer).fit(X_tr_fe)
            X_tr_prep = prep_fold.transform(X_tr_fe)
            X_val_prep = prep_fold.transform(X_val_fe)

            p_list = []
            for clf in [lgb_clf, cat_clf, hgb_clf, xgb_clf]:
                fitted = clone(clf).fit(X_tr_prep, y_tr)
                p_val = fitted.predict_proba(X_val_prep)[:, 1]
                p_list.append(np.clip(p_val, 1e-12, 1.0))

            # 4-way consensus power voting (p=2.0)
            pow_list = [np.power(p, power) for p in p_list]
            weighted_pow_true = sum(w * pw for w, pw in zip(weights, pow_list))
            pow_false_list = [np.power(1.0 - p, power) for p in p_list]
            weighted_pow_false = sum(w * pw for w, pw in zip(weights, pow_false_list))
            total_pow = weighted_pow_true + weighted_pow_false
            p_consensus = weighted_pow_true / total_pow

            oof_probs[val_idx] = p_consensus
            oof_folds[val_idx] = fold

        # -------------------------------------------------------------
        # 1. Group Concordance & Probability Smoothing Analysis
        # -------------------------------------------------------------
        group_ids = df_raw['PassengerId'].astype(str).str.slice(0, 4)
        df_diag = pd.DataFrame({
            'group_id': group_ids,
            'fold': oof_folds,
            'prob': oof_probs,
            'y_true': y_true,
            'pred_raw': (oof_probs >= 0.50)
        })

        # Calculate within-fold group co-presence and target concordance
        fold_group_sizes = df_diag.groupby(['fold', 'group_id'])['prob'].transform('count')
        fold_group_mean_prob = df_diag.groupby(['fold', 'group_id'])['prob'].transform('mean')
        fold_group_y_mean = df_diag.groupby(['fold', 'group_id'])['y_true'].transform('mean')
        fold_group_pred_mean = df_diag.groupby(['fold', 'group_id'])['pred_raw'].transform('mean')

        df_diag['in_fold_group_size'] = fold_group_sizes
        df_diag['in_fold_group_mean_prob'] = fold_group_mean_prob
        df_diag['in_fold_group_y_mean'] = fold_group_y_mean
        df_diag['in_fold_group_pred_mean'] = fold_group_pred_mean

        multi_groups_mask = df_diag['in_fold_group_size'] >= 2
        df_multi = df_diag[multi_groups_mask].copy()

        multi_group_agg = df_multi.groupby(['fold', 'group_id']).agg(
            size=('y_true', 'count'),
            y_mean=('y_true', 'mean'),
            pred_mean=('pred_raw', 'mean')
        )
        total_co_groups = len(multi_group_agg)
        total_co_passengers = int(multi_group_agg['size'].sum())
        concordant_target_groups = int(((multi_group_agg['y_mean'] == 1.0) | (multi_group_agg['y_mean'] == 0.0)).sum())
        concordant_pred_groups = int(((multi_group_agg['pred_mean'] == 1.0) | (multi_group_agg['pred_mean'] == 0.0)).sum())

        q1_rows = [
            {'metric': 'total_co_present_groups_size_gte_2', 'value': float(total_co_groups), 'fold_0': np.nan, 'fold_1': np.nan, 'fold_2': np.nan, 'fold_3': np.nan, 'fold_4': np.nan},
            {'metric': 'total_co_present_passengers', 'value': float(total_co_passengers), 'fold_0': np.nan, 'fold_1': np.nan, 'fold_2': np.nan, 'fold_3': np.nan, 'fold_4': np.nan},
            {'metric': 'target_concordant_groups_pct', 'value': float(concordant_target_groups / total_co_groups), 'fold_0': np.nan, 'fold_1': np.nan, 'fold_2': np.nan, 'fold_3': np.nan, 'fold_4': np.nan},
            {'metric': 'pred_concordant_groups_pct', 'value': float(concordant_pred_groups / total_co_groups), 'fold_0': np.nan, 'fold_1': np.nan, 'fold_2': np.nan, 'fold_3': np.nan, 'fold_4': np.nan},
        ]

        # Evaluate group probability smoothing p_adj = (1 - alpha) * p + alpha * p_group
        alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        for alpha in alphas:
            p_adj = np.where(multi_groups_mask, (1.0 - alpha) * df_diag['prob'] + alpha * df_diag['in_fold_group_mean_prob'], df_diag['prob'])
            preds_adj = (p_adj >= 0.50)
            overall_acc = float((preds_adj == y_true).mean())
            fold_accs = []
            for f in range(5):
                m_f = (oof_folds == f)
                fold_accs.append(float((preds_adj[m_f] == y_true[m_f]).mean()))
            q1_rows.append({
                'metric': f'smoothing_alpha_{alpha:.2f}',
                'value': overall_acc,
                'fold_0': fold_accs[0],
                'fold_1': fold_accs[1],
                'fold_2': fold_accs[2],
                'fold_3': fold_accs[3],
                'fold_4': fold_accs[4]
            })
        group_concordance_df = pd.DataFrame(q1_rows)

        # -------------------------------------------------------------
        # 2. Temperature Scaling & Probability Calibration Analysis
        # -------------------------------------------------------------
        eps = 1e-12
        logits = np.log(np.clip(oof_probs, eps, 1.0 - eps) / (1.0 - np.clip(oof_probs, eps, 1.0 - eps)))
        temps = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
        q2_rows = []

        # Temperature scaling accuracy & intermediate decile overconfidence
        for T in temps:
            p_t = 1.0 / (1.0 + np.exp(-logits / T))
            preds_t = (p_t >= 0.50)
            acc = float((preds_t == y_true).mean())
            f4_acc = float((preds_t[oof_folds == 4] == y_true[oof_folds == 4]).mean())

            # Evaluate decile 5 (0.50 - 0.60) and decile 6 (0.60 - 0.70)
            mid_mask = (p_t >= 0.50) & (p_t < 0.70)
            if mid_mask.sum() > 0:
                mid_mean_p = float(p_t[mid_mask].mean())
                mid_emp_pos = float(y_true[mid_mask].mean())
                overconfidence_gap = mid_mean_p - mid_emp_pos
            else:
                overconfidence_gap = 0.0

            q2_rows.append({
                'method': f'temp_T_{T:.1f}',
                'overall_acc': acc,
                'fold_4_acc': f4_acc,
                'mid_deciles_5_6_mean_prob': mid_mean_p,
                'mid_deciles_5_6_empirical_pos': mid_emp_pos,
                'mid_overconfidence_gap': overconfidence_gap
            })

        # Out-of-fold Isotonic and Logistic Calibration
        oof_isotonic = np.zeros(n_rows, dtype=float)
        oof_platt = np.zeros(n_rows, dtype=float)
        for f in range(5):
            val_m = (oof_folds == f)
            tr_m = ~val_m
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(oof_probs[tr_m], y_true[tr_m])
            oof_isotonic[val_m] = iso.predict(oof_probs[val_m])

            lr = LogisticRegression(C=1.0)
            lr.fit(logits[tr_m].reshape(-1, 1), y_true[tr_m])
            oof_platt[val_m] = lr.predict_proba(logits[val_m].reshape(-1, 1))[:, 1]

        for name, p_cal in [('isotonic_calibrated', oof_isotonic), ('platt_logistic_calibrated', oof_platt)]:
            preds_cal = (p_cal >= 0.50)
            acc = float((preds_cal == y_true).mean())
            f4_acc = float((preds_cal[oof_folds == 4] == y_true[oof_folds == 4]).mean())
            mid_mask = (p_cal >= 0.50) & (p_cal < 0.70)
            mid_mean_p = float(p_cal[mid_mask].mean()) if mid_mask.sum() > 0 else 0.0
            mid_emp_pos = float(y_true[mid_mask].mean()) if mid_mask.sum() > 0 else 0.0
            overconfidence_gap = mid_mean_p - mid_emp_pos
            q2_rows.append({
                'method': name,
                'overall_acc': acc,
                'fold_4_acc': f4_acc,
                'mid_deciles_5_6_mean_prob': mid_mean_p,
                'mid_deciles_5_6_empirical_pos': mid_emp_pos,
                'mid_overconfidence_gap': overconfidence_gap
            })
        temp_cal_df = pd.DataFrame(q2_rows)

        # -------------------------------------------------------------
        # 3. Subgroup Decision Thresholds & Calibrated Rules Analysis
        # -------------------------------------------------------------
        spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        total_spend = df_raw[spend_cols].fillna(0.0).sum(axis=1)
        cryo = (df_raw['CryoSleep'] == True) | (df_raw['CryoSleep'] == 'True')
        home_planet = df_raw['HomePlanet'].astype(str)

        is_earth = (home_planet == 'Earth')
        is_earth_zero_noncryo = is_earth & (~cryo) & (total_spend == 0.0)

        thresholds = [0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58]
        q3_rows = []

        # Sweep Earth threshold
        for th in thresholds:
            earth_preds = (oof_probs[is_earth] >= th)
            earth_acc = float((earth_preds == y_true[is_earth]).mean())
            q3_rows.append({
                'segment': 'All_Earth_Passengers',
                'threshold': th,
                'segment_acc': earth_acc,
                'segment_count': int(is_earth.sum()),
                'optimal_delta_vs_default': earth_acc - float(((oof_probs[is_earth] >= 0.50) == y_true[is_earth]).mean())
            })

        # Sweep Earth Zero-Spend Non-Cryo threshold
        for th in thresholds:
            ezn_preds = (oof_probs[is_earth_zero_noncryo] >= th)
            ezn_acc = float((ezn_preds == y_true[is_earth_zero_noncryo]).mean())
            q3_rows.append({
                'segment': 'Earth_NonCryo_ZeroSpend',
                'threshold': th,
                'segment_acc': ezn_acc,
                'segment_count': int(is_earth_zero_noncryo.sum()),
                'optimal_delta_vs_default': ezn_acc - float(((oof_probs[is_earth_zero_noncryo] >= 0.50) == y_true[is_earth_zero_noncryo]).mean())
            })

        # Evaluate combined subgroup decision rule on overall & fold 4 accuracy
        # Rule A: Default 0.50 everywhere
        default_acc = float(((oof_probs >= 0.50) == y_true).mean())
        default_f4_acc = float(((oof_probs[oof_folds == 4] >= 0.50) == y_true[oof_folds == 4]).mean())

        # Test varying Earth threshold while holding non-Earth at 0.50
        for th_earth in [0.46, 0.48, 0.50, 0.52, 0.54]:
            p_comb = np.where(is_earth, (oof_probs >= th_earth), (oof_probs >= 0.50))
            overall_acc = float((p_comb == y_true).mean())
            f4_acc = float((p_comb[oof_folds == 4] == y_true[oof_folds == 4]).mean())
            q3_rows.append({
                'segment': f'Full_Dataset_Earth_th_{th_earth:.2f}_Other_0.50',
                'threshold': th_earth,
                'segment_acc': overall_acc,
                'segment_count': n_rows,
                'optimal_delta_vs_default': overall_acc - default_acc
            })

        subgroup_th_df = pd.DataFrame(q3_rows)

        self.results_ = {
            'group_concordance_analysis': group_concordance_df,
            'temperature_calibration_analysis': temp_cal_df,
            'subgroup_threshold_analysis': subgroup_th_df
        }
        return self.results_

def build():
    train_op = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    X = train_op.drop(columns=['Transported'])
    y = train_op['Transported'].astype(bool)

    analyzer = ConsensusDiagnosticsTransformer()
    diagnostics = X.skb.apply(analyzer, y=y, no_wrap=True)

    return {
        'group_concordance_analysis': diagnostics['group_concordance_analysis'],
        'temperature_calibration_analysis': diagnostics['temperature_calibration_analysis'],
        'subgroup_threshold_analysis': diagnostics['subgroup_threshold_analysis']
    }