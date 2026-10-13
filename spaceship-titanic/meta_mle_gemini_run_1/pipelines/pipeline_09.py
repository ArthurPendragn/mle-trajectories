import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skrub
from xgboost import XGBClassifier


class PartyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Recovers missing attributes via physical cabin layout and traveling group cohesion,
    deduces logical constraints, and engineers intra-group relational signals.
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
        group_ids = df['PassengerId'].astype(str).str.slice(0, 4)
        group_num = pd.to_numeric(
            df['PassengerId'].astype(str).str.slice(5), errors='coerce'
        ).fillna(1.0)

        amenity_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        missing_amenities = df[amenity_cols].isna().sum(axis=1).astype(float)
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

        # Cabin decomposition
        cabin_str = df['Cabin'].astype(str)
        cabin_parts = cabin_str.str.split('/', expand=True)
        raw_deck = cabin_parts[0].where(df['Cabin'].notna(), np.nan)
        raw_side = cabin_parts[2].where(df['Cabin'].notna(), np.nan)
        cabin_num = pd.to_numeric(cabin_parts[1], errors='coerce').fillna(-1.0)
        cabin_region = (cabin_num // 300.0).astype(float)

        # Group cohesion: recover missing Deck and Side from party members
        df_deck = pd.DataFrame({'group_id': group_ids, 'deck': raw_deck}).dropna()
        group_to_deck = df_deck.groupby('group_id')['deck'].first().to_dict()
        deck_imputed = raw_deck.fillna(group_ids.map(group_to_deck)).fillna('Missing')

        df_side = pd.DataFrame({'group_id': group_ids, 'side': raw_side}).dropna()
        group_to_side = df_side.groupby('group_id')['side'].first().to_dict()
        side_imputed = raw_side.fillna(group_ids.map(group_to_side)).fillna('Missing')

        # HomePlanet recovery: group consensus + strict Deck physical layout rules
        # Decks A, B, C, T are exclusive to Europa; Deck G is exclusive to Earth
        df_planet = pd.DataFrame(
            {'group_id': group_ids, 'planet': df['HomePlanet']}
        ).dropna()
        group_to_planet = df_planet.groupby('group_id')['planet'].first().to_dict()
        planet_imputed = df['HomePlanet'].fillna(group_ids.map(group_to_planet))
        planet_imputed = planet_imputed.mask(
            planet_imputed.isna() & deck_imputed.isin(['A', 'B', 'C', 'T']), 'Europa'
        )
        planet_imputed = planet_imputed.mask(
            planet_imputed.isna() & (deck_imputed == 'G'), 'Earth'
        )
        home_planet = planet_imputed.fillna('Missing').astype(str)

        # CryoSleep logical deduction: passengers spending money cannot be in CryoSleep
        raw_cryo = df['CryoSleep'].copy()
        raw_cryo = raw_cryo.mask(raw_cryo.isna() & (spend_sum > 0), False)
        cryo_sleep = raw_cryo.fillna('Missing').astype(str)

        # Intra-group behavioral signals
        df_spend = pd.DataFrame(
            {
                'group_id': group_ids,
                'spend': spend_sum,
                'zero': zero_spend,
                'cryo': (raw_cryo == True).astype(float),
            }
        )
        group_tot_spend_local = df_spend.groupby('group_id')['spend'].transform('sum')
        group_mean_spend_local = df_spend.groupby('group_id')['spend'].transform('mean')
        group_zero_spend_ratio = df_spend.groupby('group_id')['zero'].transform('mean')
        group_cryo_ratio = df_spend.groupby('group_id')['cryo'].transform('mean')
        has_cryo_companion = (group_cryo_ratio > 0.0).astype(float)

        spend_diff_from_group_mean = spend_sum - group_mean_spend_local
        spend_ratio_in_group = spend_sum / (group_tot_spend_local + 1.0)

        # External training map fallbacks
        group_size_local = group_ids.map(group_ids.value_counts()).astype(float)
        group_size_fitted = (
            group_ids.map(self.group_size_map_).fillna(1.0).astype(float)
        )
        group_size = np.maximum(group_size_local, group_size_fitted)
        is_alone = (group_size == 1.0).astype(float)

        group_total_spend = (
            group_ids.map(self.group_sum_spend_map_)
            .fillna(group_tot_spend_local)
            .astype(float)
        )
        group_mean_spend = (
            group_ids.map(self.group_mean_spend_map_)
            .fillna(group_mean_spend_local)
            .astype(float)
        )

        cabins = df['Cabin'].fillna('Missing').astype(str)
        cabin_occupants = (
            cabins.map(self.cabin_occupants_map_).fillna(1.0).astype(float)
        )
        surnames = df['Name'].fillna('Missing').astype(str).str.split().str[-1]
        family_size = surnames.map(self.surname_freq_map_).fillna(1.0).astype(float)
        is_child = (df['Age'].fillna(999.0) <= 12.0).astype(float)

        destination = df['Destination'].fillna('Missing').astype(str)
        vip = df['VIP'].fillna('Missing').astype(str)

        out_df = pd.DataFrame(
            {
                'HomePlanet': home_planet,
                'CryoSleep': cryo_sleep,
                'Destination': destination,
                'VIP': vip,
                'Deck': deck_imputed,
                'Side': side_imputed,
                'Age': df['Age'],
                'CabinNum': cabin_num,
                'CabinRegion': cabin_region,
                'CabinOccupants': cabin_occupants,
                'GroupNum': group_num,
                'GroupSize': group_size,
                'IsAlone': is_alone,
                'GroupTotalSpend': group_total_spend,
                'GroupMeanSpend': group_mean_spend,
                'GroupCryoRatio': group_cryo_ratio,
                'GroupZeroSpendRatio': group_zero_spend_ratio,
                'HasCryoCompanion': has_cryo_companion,
                'SpendDiffFromGroupMean': spend_diff_from_group_mean,
                'SpendRatioInGroup': spend_ratio_in_group,
                'MissingAmenityCount': missing_amenities,
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


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    features = X.skb.set_name('raw_features')

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
        'GroupCryoRatio',
        'GroupZeroSpendRatio',
        'HasCryoCompanion',
        'SpendDiffFromGroupMean',
        'SpendRatioInGroup',
        'MissingAmenityCount',
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

    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    num_transformer = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_cols),
            ('num', num_transformer, num_cols),
        ]
    )

    xgb_model = XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
        n_jobs=4,
        eval_metric='logloss',
    )
    lgb_model = LGBMClassifier(
        learning_rate=0.05,
        num_leaves=15,
        n_estimators=300,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=4,
    )

    voting_candidates = {
        'xgb_lgb_equal': VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgbm', lgb_model)],
            voting='soft',
        ),
        'xgb_lgb_weighted_xgb': VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgbm', lgb_model)],
            voting='soft',
            weights=[1.5, 1.0],
        ),
        'xgb_lgb_rf_weighted': VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgbm', lgb_model), ('rf', rf_model)],
            voting='soft',
            weights=[2.0, 2.0, 1.0],
        ),
    }

    chosen_model = skrub.choose_from(voting_candidates, name='ensemble_variant')

    pipeline = Pipeline(
        [
            ('party_fe', PartyFeatureEngineer()),
            ('prep', preprocessor),
            ('model', chosen_model),
        ]
    )

    pred = features.skb.apply(pipeline, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}