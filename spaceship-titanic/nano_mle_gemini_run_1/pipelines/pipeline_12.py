import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skrub

class PartyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Extracts travel party context, cabin structures, and spending profiles.
    Supports logical deduction of CryoSleep from amenity spend, group-level
    propagation of HomePlanet, and luxury vs communal expenditure partitioning.
    """

    def __init__(self, feature_mode='full_relational'):
        self.feature_mode = feature_mode

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

        known_hp_mask = df['HomePlanet'].notna()
        if known_hp_mask.any():
            self.group_home_planet_map_ = df[known_hp_mask].groupby(group_ids[known_hp_mask])['HomePlanet'].first().to_dict()
        else:
            self.group_home_planet_map_ = {}

        known_cryo_mask = df['CryoSleep'].notna()
        if known_cryo_mask.any():
            cryo_bool = df.loc[known_cryo_mask, 'CryoSleep'].astype(bool)
            self.group_cryo_ratio_map_ = cryo_bool.groupby(group_ids[known_cryo_mask]).mean().to_dict()
        else:
            self.group_cryo_ratio_map_ = {}

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

        destination = df['Destination'].fillna('Missing').astype(str)
        vip = df['VIP'].fillna('Missing').astype(str)

        raw_cryo = df['CryoSleep']
        can_deduce_cryo = raw_cryo.isna() & (spend_sum > 0.0)
        inferred_cryo_flag = can_deduce_cryo.astype(float)

        if self.feature_mode == 'baseline':
            home_planet = df['HomePlanet'].fillna('Missing').astype(str)
            cryo_sleep = raw_cryo.fillna('Missing').astype(str)
        else:
            cryo_deduced = raw_cryo.copy()
            cryo_deduced = cryo_deduced.mask(can_deduce_cryo, False)

            if self.feature_mode == 'inferred_cryo':
                home_planet = df['HomePlanet'].fillna('Missing').astype(str)
                cryo_sleep = cryo_deduced.fillna('Missing').astype(str)
            else:
                known_hp_mask_test = df['HomePlanet'].notna()
                known_hp_intra = df[known_hp_mask_test].groupby(group_ids[known_hp_mask_test])['HomePlanet'].first().to_dict()
                hp_propagated = df['HomePlanet'].fillna(group_ids.map(known_hp_intra)).fillna(group_ids.map(self.group_home_planet_map_))
                home_planet = hp_propagated.fillna('Missing').astype(str)
                cryo_sleep = cryo_deduced.fillna('Missing').astype(str)

        group_cryo_ratio = group_ids.map(self.group_cryo_ratio_map_).fillna(0.5).astype(float)

        communal_spend = df['FoodCourt'].fillna(0.0) + df['ShoppingMall'].fillna(0.0)
        log_communal_spend = np.log1p(communal_spend)
        communal_ratio = communal_spend / (spend_sum + 1.0)

        data = {
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
            'InferredCryoFlag': inferred_cryo_flag,
            'GroupCryoRatio': group_cryo_ratio,
            'CommunalSpend': communal_spend,
            'LogCommunalSpend': log_communal_spend,
            'CommunalRatio': communal_ratio,
        }
        return pd.DataFrame(data, index=df.index)

def make_pipeline(mode):
    base_num_cols = [
        'Age', 'CabinNum', 'CabinRegion', 'CabinOccupants', 'GroupNum', 'GroupSize',
        'IsAlone', 'GroupTotalSpend', 'GroupMeanSpend', 'FamilySize', 'IsChild',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend',
        'ZeroSpend', 'LogTotalSpend', 'LuxurySpend', 'LogLuxurySpend', 'LuxuryRatio',
        'RoomServiceRatio', 'SpaRatio', 'VRDeckRatio'
    ]
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

    if mode == 'baseline':
        num_cols = list(base_num_cols)
    elif mode == 'inferred_cryo':
        num_cols = base_num_cols + ['InferredCryoFlag']
    elif mode == 'transit_context':
        num_cols = base_num_cols + ['InferredCryoFlag', 'GroupCryoRatio']
    elif mode == 'full_relational':
        num_cols = base_num_cols + ['InferredCryoFlag', 'GroupCryoRatio', 'CommunalSpend', 'LogCommunalSpend', 'CommunalRatio']
    else:
        raise ValueError(f'Unknown mode {mode}')

    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, cat_cols), ('num', num_transformer, num_cols)])

    lgb_clf = LGBMClassifier(learning_rate=0.05, num_leaves=15, n_estimators=300, random_state=42, n_jobs=4, verbose=-1)
    cat_clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, random_seed=42, thread_count=4, verbose=0, allow_writing_files=False)
    chosen_model = VotingClassifier(estimators=[('lgbm', lgb_clf), ('cat', cat_clf)], voting='soft', weights=[1, 2])

    return Pipeline([
        ('party_fe', PartyFeatureEngineer(feature_mode=mode)),
        ('prep', preprocessor),
        ('model', chosen_model)
    ])

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    features = X.skb.set_name('raw_features')

    pipeline_variants = {
        'baseline': make_pipeline('baseline'),
        'inferred_cryo': make_pipeline('inferred_cryo'),
        'transit_context': make_pipeline('transit_context'),
        'full_relational': make_pipeline('full_relational'),
    }

    chosen_pipeline = skrub.choose_from(pipeline_variants, name='feature_set')
    pred = features.skb.apply(chosen_pipeline, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}