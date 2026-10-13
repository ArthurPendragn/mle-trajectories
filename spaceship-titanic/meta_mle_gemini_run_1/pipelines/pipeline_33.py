import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import skrub

class PartyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Fits party, cabin, surname, and co-traveler behavioral statistics on training rows without leakage.
    Provides group-level behavioral concordance aggregates (CryoSleep ratio, zero-spend ratio),
    group/surname origin propagation, deterministic CryoSleep deduction,
    Deck G spatial coordinates, and targeted Earth cohort interaction features.
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

        cryo_series = df['CryoSleep'].copy()
        cryo_series = cryo_series.mask(cryo_series.isna() & (total_spends > 0.0), False)
        cryo_bool = (cryo_series == True) | (cryo_series == 'True')
        zero_spend_bool = total_spends == 0.0
        df_group_behave = pd.DataFrame({
            'group_id': group_ids,
            'cryo': cryo_bool.astype(float),
            'zero': zero_spend_bool.astype(float),
        })
        group_behave_agg = df_group_behave.groupby('group_id').agg(
            cryo_ratio=('cryo', 'mean'),
            any_cryo=('cryo', 'max'),
            zero_ratio=('zero', 'mean'),
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
            'zero': zero_spend,
        })
        local_behave_agg = df_local_behave.groupby('group_id').agg(
            cryo_ratio=('cryo', 'mean'),
            any_cryo=('cryo', 'max'),
            zero_ratio=('zero', 'mean'),
        )
        local_cryo_ratio = local_behave_agg['cryo_ratio'].to_dict()
        local_any_cryo = local_behave_agg['any_cryo'].to_dict()
        local_zero_ratio = local_behave_agg['zero_ratio'].to_dict()

        group_cryo_ratio = group_ids.map(self.group_cryo_ratio_map_).fillna(group_ids.map(local_cryo_ratio)).fillna(cryo_bool.astype(float)).astype(float)
        group_any_cryo = group_ids.map(self.group_any_cryo_map_).fillna(group_ids.map(local_any_cryo)).fillna(cryo_bool.astype(float)).astype(float)
        group_zero_spend_ratio = group_ids.map(self.group_zero_spend_map_).fillna(group_ids.map(local_zero_ratio)).fillna(zero_spend).astype(float)

        earth_destination = np.where(home_planet == 'Earth', 'Earth_' + destination, 'Other')
        earth_side = np.where(home_planet == 'Earth', 'Earth_' + side, 'Other')
        earth_cryo_side = np.where(home_planet == 'Earth', 'Earth_' + cryo_sleep + '_' + side, 'Other')
        earth_minor_pso = ((home_planet == 'Earth') & (age_val.fillna(999.0) <= 12.0) & (destination == 'PSO J318.5-22')).astype(float)
        earth_cryo_s = ((home_planet == 'Earth') & (cryo_sleep == 'True') & (side == 'S')).astype(float)
        earth_cryo_p = ((home_planet == 'Earth') & (cryo_sleep == 'True') & (side == 'P')).astype(float)
        earth_noncryo_zerospend_side = ((home_planet == 'Earth') & (cryo_sleep == 'False') & (spend_sum == 0.0) & (side == 'S')).astype(float)

        deck_g_cabin_num = np.where(deck == 'G', cabin_num, -1.0)
        deck_g_low_cabin = ((deck == 'G') & (cabin_num < 600.0)).astype(float)
        deck_g_starboard_mid = ((deck == 'G') & (side == 'S') & (cabin_num >= 600.0) & (cabin_num <= 1200.0)).astype(float)

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
            'DeckG_CabinNum': deck_g_cabin_num,
            'DeckG_LowCabin': deck_g_low_cabin,
            'DeckG_Starboard_MidHotspot': deck_g_starboard_mid,
            'HomePlanet_Cryo': home_planet + '_' + cryo_sleep,
            'IsEarthChild': ((home_planet == 'Earth') & (age_val.fillna(999.0) <= 12.0)).astype(float),
            'Earth_NonCryo_ZeroSpend': ((home_planet == 'Earth') & (cryo_sleep == 'False') & (spend_sum == 0.0)).astype(float),
            'HomePlanet_Deck': home_planet + '_' + deck,
            'GroupCryoRatio': group_cryo_ratio,
            'GroupAnyCryo': group_any_cryo,
            'GroupZeroSpendRatio': group_zero_spend_ratio,
            'Earth_Destination': earth_destination,
            'Earth_Side': earth_side,
            'Earth_Cryo_Side': earth_cryo_side,
            'Earth_Minor_PSO': earth_minor_pso,
            'Earth_Cryo_S': earth_cryo_s,
            'Earth_Cryo_P': earth_cryo_p,
            'Earth_NonCryo_ZeroSpend_Side': earth_noncryo_zerospend_side,
        }
        return pd.DataFrame(out_data, index=df.index)

class PowerVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Weighted soft-voting and non-linear power-transformed probability consensus classifier.
    Sharpens consensus across diverse base models and regularizes intermediate overconfidence.
    """

    def __init__(self, estimators=None, weights=None, power=1.0):
        self.estimators = estimators
        self.weights = weights
        self.power = power

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.named_estimators_ = {}
        for name, est in self.estimators:
            fitted_est = clone(est).fit(X, y)
            self.estimators_.append(fitted_est)
            self.named_estimators_[name] = fitted_est
        return self

    def predict_proba(self, X):
        if self.weights is None:
            w = [1.0] * len(self.estimators_)
        else:
            w = [float(val) for val in self.weights]

        probas = [est.predict_proba(X) for est in self.estimators_]
        if self.power == 1.0:
            weighted_p = sum(wi * pi for wi, pi in zip(w, probas))
            total_w = sum(w)
            return weighted_p / total_w
        else:
            p_powers = [np.power(np.clip(pi, 1e-12, 1.0), self.power) for pi in probas]
            weighted_pow = sum(wi * p_pow for wi, p_pow in zip(w, p_powers))
            denom = np.sum(weighted_pow, axis=1, keepdims=True)
            denom = np.where(denom == 0, 1.0, denom)
            return weighted_pow / denom

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    features = X.skb.set_name('raw_features')

    base_cat_cols = [
        'HomePlanet', 'CryoSleep', 'Destination', 'VIP',
        'Deck', 'Side', 'DeckSide', 'HomePlanet_Cryo', 'HomePlanet_Deck'
    ]
    base_num_cols = [
        'Age', 'CabinNum', 'CabinRegion', 'CabinOccupants', 'GroupNum',
        'GroupSize', 'IsAlone', 'GroupTotalSpend', 'GroupMeanSpend',
        'FamilySize', 'IsChild', 'RoomService', 'FoodCourt', 'ShoppingMall',
        'Spa', 'VRDeck', 'TotalSpend', 'ZeroSpend', 'LogTotalSpend',
        'LuxurySpend', 'LogLuxurySpend', 'LuxuryRatio', 'RoomServiceRatio',
        'SpaRatio', 'VRDeckRatio', 'DeckGStarboardHotspot', 'IsEarthChild',
        'Earth_NonCryo_ZeroSpend'
    ]
    group_num_cols = ['GroupCryoRatio', 'GroupAnyCryo', 'GroupZeroSpendRatio']

    deck_g_num_cols = ['DeckG_CabinNum', 'DeckG_LowCabin', 'DeckG_Starboard_MidHotspot']
    earth_cat_cols = ['Earth_Destination', 'Earth_Side', 'Earth_Cryo_Side']
    earth_num_cols = ['Earth_Minor_PSO', 'Earth_Cryo_S', 'Earth_Cryo_P', 'Earth_NonCryo_ZeroSpend_Side']

    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    prep_baseline = ColumnTransformer(transformers=[
        ('cat', cat_transformer, base_cat_cols),
        ('num', num_transformer, base_num_cols + group_num_cols)
    ])

    prep_deck_g = ColumnTransformer(transformers=[
        ('cat', cat_transformer, base_cat_cols),
        ('num', num_transformer, base_num_cols + group_num_cols + deck_g_num_cols)
    ])

    prep_earth = ColumnTransformer(transformers=[
        ('cat', cat_transformer, base_cat_cols + earth_cat_cols),
        ('num', num_transformer, base_num_cols + group_num_cols + earth_num_cols)
    ])

    prep_comp = ColumnTransformer(transformers=[
        ('cat', cat_transformer, base_cat_cols + earth_cat_cols),
        ('num', num_transformer, base_num_cols + group_num_cols + deck_g_num_cols + earth_num_cols)
    ])

    prep_variants = {
        'baseline_parent': prep_baseline,
        'deck_g_spatial_partitioning': prep_deck_g,
        'earth_cohort_disambiguation': prep_earth,
        'comprehensive_earth_deck_g': prep_comp
    }
    chosen_prep = skrub.choose_from(prep_variants, name='feature_enhancement_variant')

    fe = PartyFeatureEngineer()

    lgb_clf = LGBMClassifier(
        learning_rate=0.05,
        num_leaves=15,
        n_estimators=300,
        random_state=42,
        n_jobs=4,
        verbose=-1
    )
    cat_clf = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        random_seed=42,
        thread_count=4,
        verbose=0,
        allow_writing_files=False
    )
    hgb_clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        random_state=42
    )
    xgb_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        eval_metric='logloss'
    )

    chosen_consensus = PowerVotingClassifier(
        estimators=[('lgbm', lgb_clf), ('cat', cat_clf), ('hgb', hgb_clf), ('xgb', xgb_clf)],
        weights=[2, 3, 1, 1],
        power=2.0
    )

    pipeline_choice = Pipeline([
        ('party_fe', fe),
        ('prep', chosen_prep),
        ('model', chosen_consensus)
    ])

    pred = features.skb.apply(pipeline_choice, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}