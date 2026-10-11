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


class ContextualPartyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Fits party, cabin, and surname aggregation statistics on training rows
    and maps them at transform time along with deterministic transit deductions
    (CryoSleep deduction from positive spend) and micro-spatial deck/side interactions.
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
        luxury_spend = df['RoomService'].fillna(0.0) + df['Spa'].fillna(0.0) + df['VRDeck'].fillna(0.0)
        communal_spend = df['FoodCourt'].fillna(0.0) + df['ShoppingMall'].fillna(0.0)

        log_total_spend = np.log1p(spend_sum)
        log_luxury_spend = np.log1p(luxury_spend)
        log_communal_spend = np.log1p(communal_spend)

        luxury_ratio = luxury_spend / (spend_sum + 1.0)
        communal_ratio = communal_spend / (spend_sum + 1.0)
        luxury_communal_ratio = luxury_spend / (communal_spend + 1.0)
        room_service_ratio = df['RoomService'].fillna(0.0) / (spend_sum + 1.0)
        spa_ratio = df['Spa'].fillna(0.0) / (spend_sum + 1.0)
        vr_ratio = df['VRDeck'].fillna(0.0) / (spend_sum + 1.0)
        amenity_count = (df[amenity_cols].fillna(0.0) > 0.0).sum(axis=1).astype(float)

        cabin_str = df['Cabin'].astype(str)
        cabin_parts = cabin_str.str.split('/', expand=True)
        deck = cabin_parts[0].where(df['Cabin'].notna(), 'Missing')
        side = cabin_parts[2].where(df['Cabin'].notna(), 'Missing')
        deck_side = (deck + '_' + side).where(df['Cabin'].notna(), 'Missing')
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

        # Deterministic transit deduction: CryoSleep strictly precludes amenity spending
        cryo_raw = df['CryoSleep'].copy()
        cryo_deduced = cryo_raw.where(~(cryo_raw.isna() & (spend_sum > 0.0)), False)
        cryo_sleep_deduced = cryo_deduced.fillna('Missing').astype(str)

        destination = df['Destination'].fillna('Missing').astype(str)
        vip = df['VIP'].fillna('Missing').astype(str)

        out_df = pd.DataFrame(
            {
                'HomePlanet': home_planet,
                'CryoSleep': cryo_sleep,
                'CryoSleepDeduced': cryo_sleep_deduced,
                'Destination': destination,
                'VIP': vip,
                'Deck': deck,
                'Side': side,
                'DeckSide': deck_side,
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
                'CommunalSpend': communal_spend,
                'LogCommunalSpend': log_communal_spend,
                'LuxuryRatio': luxury_ratio,
                'CommunalRatio': communal_ratio,
                'LuxuryCommunalRatio': luxury_communal_ratio,
                'RoomServiceRatio': room_service_ratio,
                'SpaRatio': spa_ratio,
                'VRDeckRatio': vr_ratio,
                'AmenityCount': amenity_count,
            },
            index=df.index,
        )
        return out_df


def make_ensemble():
    lgb_clf = LGBMClassifier(
        learning_rate=0.05,
        num_leaves=15,
        n_estimators=300,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    cat_clf = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        random_seed=42,
        thread_count=4,
        verbose=0,
        allow_writing_files=False,
    )
    return VotingClassifier(
        estimators=[('lgbm', lgb_clf), ('cat', cat_clf)],
        voting='soft',
        weights=[1, 2],
    )


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    features = X.skb.set_name('raw_features')

    # Common transformer blocks
    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    # Variant 1: Parent baseline feature set
    cat_cols_baseline = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    num_cols_baseline = [
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
    prep_baseline = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_cols_baseline),
            ('num', num_transformer, num_cols_baseline),
        ]
    )

    # Variant 2: Deductive CryoSleep + DeckSide interaction
    cat_cols_deductive = ['HomePlanet', 'CryoSleepDeduced', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide']
    num_cols_deductive = list(num_cols_baseline)
    prep_deductive = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_cols_deductive),
            ('num', num_transformer, num_cols_deductive),
        ]
    )

    # Variant 3: Deductive CryoSleep + DeckSide + Amenity decomposition profiles
    cat_cols_full = ['HomePlanet', 'CryoSleepDeduced', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide']
    num_cols_full = list(num_cols_baseline) + [
        'CommunalSpend',
        'LogCommunalSpend',
        'CommunalRatio',
        'LuxuryCommunalRatio',
        'AmenityCount',
    ]
    prep_full = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_cols_full),
            ('num', num_transformer, num_cols_full),
        ]
    )

    pipe_baseline = Pipeline([
        ('fe', ContextualPartyFeatureEngineer()),
        ('prep', prep_baseline),
        ('model', make_ensemble()),
    ])

    pipe_deductive = Pipeline([
        ('fe', ContextualPartyFeatureEngineer()),
        ('prep', prep_deductive),
        ('model', make_ensemble()),
    ])

    pipe_full = Pipeline([
        ('fe', ContextualPartyFeatureEngineer()),
        ('prep', prep_full),
        ('model', make_ensemble()),
    ])

    chosen_pipe = skrub.choose_from(
        {
            'parent_baseline': pipe_baseline,
            'deductive_deckside': pipe_deductive,
            'full_context_fe': pipe_full,
        },
        name='feature_representation',
    )

    pred = features.skb.apply(chosen_pipe, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}