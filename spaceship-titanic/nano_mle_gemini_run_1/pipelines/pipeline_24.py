import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skrub

class PartyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Fits party, cabin, and relational statistics on training rows without leakage.
    Provides targeted Earth subgroup calibration, group/surname origin propagation,
    deterministic CryoSleep deduction, and Deck G Starboard spatial indicators.
    """

    def __init__(self, mode='earth_comprehensive'):
        self.mode = mode

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

        group_size = group_ids.map(self.group_size_map_).fillna(1.0).astype(float)
        is_alone = (group_size == 1.0).astype(float)
        group_total_spend = group_ids.map(self.group_sum_spend_map_).fillna(spend_sum).astype(float)
        group_mean_spend = group_ids.map(self.group_mean_spend_map_).fillna(spend_sum).astype(float)
        cabin_occupants = cabins.map(self.cabin_occupants_map_).fillna(1.0).astype(float)
        family_size = surnames.map(self.surname_freq_map_).fillna(1.0).astype(float)

        age_val = df['Age']
        is_child = (age_val.fillna(999.0) <= 12.0).astype(float)

        destination = df['Destination'].fillna('Missing').astype(str)
        vip = df['VIP'].fillna('Missing').astype(str)

        if self.mode == 'parent_baseline':
            home_planet = df['HomePlanet'].fillna('Missing').astype(str)
            cryo_series = df['CryoSleep'].copy()
            cryo_series = cryo_series.mask(cryo_series.isna() & (spend_sum > 0.0), False)
            cryo_sleep = cryo_series.fillna('Missing').astype(str)
        else:
            hp_series = df['HomePlanet'].copy()
            if self.mode == 'earth_comprehensive':
                hp_series = hp_series.fillna(group_ids.map(self.group_to_hp_map_))
                hp_series = hp_series.fillna(surnames.map(self.surname_to_hp_map_))
            home_planet = hp_series.fillna('Missing').astype(str)

            cryo_series = df['CryoSleep'].copy()
            cryo_series = cryo_series.mask(cryo_series.isna() & (spend_sum > 0.0), False)
            cryo_sleep = cryo_series.fillna('Missing').astype(str)

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
        }

        if self.mode != 'parent_baseline':
            out_data['HomePlanet_Cryo'] = home_planet + '_' + cryo_sleep
            out_data['IsEarthChild'] = ((home_planet == 'Earth') & (age_val.fillna(999.0) <= 12.0)).astype(float)
            out_data['Earth_NonCryo_ZeroSpend'] = ((home_planet == 'Earth') & (cryo_sleep == 'False') & (spend_sum == 0.0)).astype(float)
            out_data['HomePlanet_Deck'] = home_planet + '_' + deck

        return pd.DataFrame(out_data, index=df.index)

class PowerVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Weighted soft-voting and non-linear power-transformed probability consensus classifier.
    Sharpens consensus across base models and regularizes intermediate overconfidence.
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
            weighted_p = sum((wi * pi for wi, pi in zip(w, probas)))
            total_w = sum(w)
            return weighted_p / total_w
        else:
            p_powers = [np.power(np.clip(pi, 1e-12, 1.0), self.power) for pi in probas]
            weighted_pow = sum((wi * p_pow for wi, p_pow in zip(w, p_powers)))
            denom = np.sum(weighted_pow, axis=1, keepdims=True)
            denom = np.where(denom == 0, 1.0, denom)
            return weighted_pow / denom

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

def make_candidate_pipeline(variant_name, base_cat, base_num, cat_tx, num_tx, ensemble_model):
    if variant_name == 'parent_baseline':
        cat_cols = base_cat + ['DeckSide']
        num_cols = base_num + ['DeckGStarboardHotspot']
        fe = PartyFeatureEngineer(mode='parent_baseline')
    elif variant_name == 'earth_core':
        cat_cols = base_cat + ['DeckSide', 'HomePlanet_Cryo']
        num_cols = base_num + ['DeckGStarboardHotspot', 'IsEarthChild']
        fe = PartyFeatureEngineer(mode='earth_core')
    elif variant_name == 'earth_spend_calibrated':
        cat_cols = base_cat + ['DeckSide', 'HomePlanet_Cryo']
        num_cols = base_num + ['DeckGStarboardHotspot', 'IsEarthChild', 'Earth_NonCryo_ZeroSpend']
        fe = PartyFeatureEngineer(mode='earth_spend_calibrated')
    else:
        cat_cols = base_cat + ['DeckSide', 'HomePlanet_Cryo', 'HomePlanet_Deck']
        num_cols = base_num + ['DeckGStarboardHotspot', 'IsEarthChild', 'Earth_NonCryo_ZeroSpend']
        fe = PartyFeatureEngineer(mode='earth_comprehensive')

    preprocessor = ColumnTransformer(transformers=[
        ('cat', cat_tx, cat_cols),
        ('num', num_tx, num_cols)
    ])
    return Pipeline([('party_fe', fe), ('prep', preprocessor), ('model', ensemble_model)])

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    features = X.skb.set_name('raw_features')

    base_cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    base_num_cols = [
        'Age', 'CabinNum', 'CabinRegion', 'CabinOccupants', 'GroupNum', 'GroupSize',
        'IsAlone', 'GroupTotalSpend', 'GroupMeanSpend', 'FamilySize', 'IsChild',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'TotalSpend', 'ZeroSpend', 'LogTotalSpend', 'LuxurySpend', 'LogLuxurySpend',
        'LuxuryRatio', 'RoomServiceRatio', 'SpaRatio', 'VRDeckRatio'
    ]

    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    lgb_clf = LGBMClassifier(learning_rate=0.05, num_leaves=15, n_estimators=300, random_state=42, n_jobs=4, verbose=-1)
    cat_clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, random_seed=42, thread_count=4, verbose=0, allow_writing_files=False)
    model_2way = [('lgbm', lgb_clf), ('cat', cat_clf)]
    power_ensemble = PowerVotingClassifier(estimators=model_2way, weights=[1, 2], power=3.0)

    variant_pipelines = {
        'parent_baseline': make_candidate_pipeline('parent_baseline', base_cat_cols, base_num_cols, cat_transformer, num_transformer, power_ensemble),
        'earth_core': make_candidate_pipeline('earth_core', base_cat_cols, base_num_cols, cat_transformer, num_transformer, power_ensemble),
        'earth_spend_calibrated': make_candidate_pipeline('earth_spend_calibrated', base_cat_cols, base_num_cols, cat_transformer, num_transformer, power_ensemble),
        'earth_comprehensive': make_candidate_pipeline('earth_comprehensive', base_cat_cols, base_num_cols, cat_transformer, num_transformer, power_ensemble)
    }

    chosen_pipeline = skrub.choose_from(variant_pipelines, name='earth_feature_variant')

    pred = features.skb.apply(chosen_pipeline, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}