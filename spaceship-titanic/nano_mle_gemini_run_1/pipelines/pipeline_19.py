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


class EarthCalibratedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Fits party and cabin context statistics on training rows (group spend,
    group sizes, cabin occupancy, surname frequencies) and contextual HomePlanet
    lookups, then extracts origin-specific transit interactions and localized
    cabin cluster features at transform time.
    """

    def __init__(
        self,
        use_cryo_deduction=True,
        use_hp_propagation=True,
        use_earth_interactions=True,
        use_deck_g_hotspot=True,
        use_deck_side=True,
    ):
        self.use_cryo_deduction = use_cryo_deduction
        self.use_hp_propagation = use_hp_propagation
        self.use_earth_interactions = use_earth_interactions
        self.use_deck_g_hotspot = use_deck_g_hotspot
        self.use_deck_side = use_deck_side

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

        # Contextual HomePlanet propagation dictionaries
        hp_known = df.loc[df['HomePlanet'].notna() & (df['HomePlanet'] != 'Missing')]
        self.group_known_hp_ = hp_known.groupby(group_ids.loc[hp_known.index])['HomePlanet'].first().to_dict()
        surname_known = hp_known.loc[(surnames.loc[hp_known.index] != 'Missing') & (surnames.loc[hp_known.index] != '')]
        self.surname_known_hp_ = surname_known.groupby(surnames.loc[surname_known.index])['HomePlanet'].first().to_dict()

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

        # Deterministic CryoSleep deduction: spend > 0 implies CryoSleep is False
        cryo_series = df['CryoSleep'].copy()
        if self.use_cryo_deduction:
            cryo_deduced_false = (spend_sum > 0.0) & (cryo_series.isna() | (cryo_series == 'Missing'))
            cryo_series = cryo_series.mask(cryo_deduced_false, False)
        cryo_sleep = cryo_series.fillna('Missing').astype(str)

        # Contextual HomePlanet propagation within group / surname
        hp_series = df['HomePlanet'].copy()
        if self.use_hp_propagation:
            known_in_df = df.loc[df['HomePlanet'].notna() & (df['HomePlanet'] != 'Missing')]
            within_group_hp = known_in_df.groupby(group_ids.loc[known_in_df.index])['HomePlanet'].first().to_dict()
            imputed_hp = group_ids.map(within_group_hp)
            imputed_hp = imputed_hp.fillna(group_ids.map(self.group_known_hp_))
            imputed_hp = imputed_hp.fillna(surnames.map(self.surname_known_hp_))
            hp_series = hp_series.fillna(imputed_hp)
        home_planet = hp_series.fillna('Missing').astype(str)

        destination = df['Destination'].fillna('Missing').astype(str)
        vip = df['VIP'].fillna('Missing').astype(str)

        out_dict = {
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
        }

        if self.use_deck_side:
            out_dict['DeckSide'] = (deck + '/' + side).astype(str)

        if self.use_earth_interactions:
            out_dict['IsEarth'] = (home_planet == 'Earth').astype(float)
            out_dict['IsEarthCryo'] = ((home_planet == 'Earth') & (cryo_sleep == 'True')).astype(float)
            out_dict['IsEarthChild'] = ((home_planet == 'Earth') & (df['Age'].fillna(999.0) <= 12.0)).astype(float)
            out_dict['IsEarthZeroSpend'] = (
                (home_planet == 'Earth') & (spend_sum == 0.0) & (cryo_sleep != 'True')
            ).astype(float)
            out_dict['IsEarthPSO'] = ((home_planet == 'Earth') & (destination == 'PSO J318.5-22')).astype(float)

        if self.use_deck_g_hotspot:
            out_dict['DeckGStarboardHotspot'] = (
                (deck == 'G') & (side == 'S') & (cabin_num >= 750.0) & (cabin_num <= 1200.0)
            ).astype(float)

        return pd.DataFrame(out_dict, index=df.index)


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    features = X.skb.set_name('raw_features')

    # Base estimators and soft voting ensemble from parent candidate
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
    chosen_model = VotingClassifier(
        estimators=[('lgbm', lgb_clf), ('cat', cat_clf)],
        voting='soft',
        weights=[1, 2],
    )

    base_num_cols = [
        'Age', 'CabinNum', 'CabinRegion', 'CabinOccupants', 'GroupNum', 'GroupSize',
        'IsAlone', 'GroupTotalSpend', 'GroupMeanSpend', 'FamilySize', 'IsChild',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend',
        'ZeroSpend', 'LogTotalSpend', 'LuxurySpend', 'LogLuxurySpend', 'LuxuryRatio',
        'RoomServiceRatio', 'SpaRatio', 'VRDeckRatio',
    ]

    # Pipeline Variant 0: parent baseline features
    fe_baseline = EarthCalibratedFeatureEngineer(
        use_cryo_deduction=False,
        use_hp_propagation=False,
        use_earth_interactions=False,
        use_deck_g_hotspot=False,
        use_deck_side=False,
    )
    prep_baseline = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']),
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), base_num_cols),
        ]
    )
    pipe_baseline = Pipeline([('fe', fe_baseline), ('prep', prep_baseline), ('model', chosen_model)])

    # Pipeline Variant 1: Earth calibrated interactions + DeckSide + deterministic Cryo & HP propagation
    fe_earth = EarthCalibratedFeatureEngineer(
        use_cryo_deduction=True,
        use_hp_propagation=True,
        use_earth_interactions=True,
        use_deck_g_hotspot=False,
        use_deck_side=True,
    )
    prep_earth = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide']),
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), base_num_cols + ['IsEarth', 'IsEarthCryo', 'IsEarthChild', 'IsEarthZeroSpend', 'IsEarthPSO']),
        ]
    )
    pipe_earth = Pipeline([('fe', fe_earth), ('prep', prep_earth), ('model', chosen_model)])

    # Pipeline Variant 2: Earth calibrated interactions + DeckSide + Deck G Starboard hotspot cluster
    fe_hotspot = EarthCalibratedFeatureEngineer(
        use_cryo_deduction=True,
        use_hp_propagation=True,
        use_earth_interactions=True,
        use_deck_g_hotspot=True,
        use_deck_side=True,
    )
    prep_hotspot = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide']),
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), base_num_cols + ['IsEarth', 'IsEarthCryo', 'IsEarthChild', 'IsEarthZeroSpend', 'IsEarthPSO', 'DeckGStarboardHotspot']),
        ]
    )
    pipe_hotspot = Pipeline([('fe', fe_hotspot), ('prep', prep_hotspot), ('model', chosen_model)])

    # Pipeline Variant 3: Contextual propagation and spatial features only (DeckSide + DeckGStarboardHotspot without manual Earth indicators)
    fe_spatial = EarthCalibratedFeatureEngineer(
        use_cryo_deduction=True,
        use_hp_propagation=True,
        use_earth_interactions=False,
        use_deck_g_hotspot=True,
        use_deck_side=True,
    )
    prep_spatial = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide']),
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), base_num_cols + ['DeckGStarboardHotspot']),
        ]
    )
    pipe_spatial = Pipeline([('fe', fe_spatial), ('prep', prep_spatial), ('model', chosen_model)])

    chosen_pipeline = skrub.choose_from(
        {
            'parent_baseline': pipe_baseline,
            'earth_calibrated': pipe_earth,
            'earth_calibrated_hotspot': pipe_hotspot,
            'deduction_and_spatial_only': pipe_spatial,
        },
        name='feature_representation',
    )

    pred = features.skb.apply(chosen_pipeline, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}