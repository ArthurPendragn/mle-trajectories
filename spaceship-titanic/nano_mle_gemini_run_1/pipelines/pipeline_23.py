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
import skrub


class PartyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Fits party, cabin, and relational statistics on training rows without leakage.
    Incorporates deterministic CryoSleep deduction and Deck G Starboard spatial indicators.
    """

    def __init__(self, mode='deck_g_spatial'):
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

        group_size = group_ids.map(self.group_size_map_).fillna(1.0).astype(float)
        is_alone = (group_size == 1.0).astype(float)
        group_total_spend = (
            group_ids.map(self.group_sum_spend_map_).fillna(spend_sum).astype(float)
        )
        group_mean_spend = (
            group_ids.map(self.group_mean_spend_map_).fillna(spend_sum).astype(float)
        )
        cabin_occupants = (
            cabins.map(self.cabin_occupants_map_).fillna(1.0).astype(float)
        )
        family_size = surnames.map(self.surname_freq_map_).fillna(1.0).astype(float)
        is_child = (df['Age'].fillna(999.0) <= 12.0).astype(float)

        home_planet = df['HomePlanet'].fillna('Missing').astype(str)
        destination = df['Destination'].fillna('Missing').astype(str)
        vip = df['VIP'].fillna('Missing').astype(str)

        if self.mode == 'baseline':
            cryo_sleep = df['CryoSleep'].fillna('Missing').astype(str)
        else:
            cryo_series = df['CryoSleep'].copy()
            cryo_series = cryo_series.mask(
                cryo_series.isna() & (spend_sum > 0.0), False
            )
            cryo_sleep = cryo_series.fillna('Missing').astype(str)

        out_data = {
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

        if self.mode != 'baseline':
            out_data['DeckSide'] = deck + '/' + side
            out_data['HomePlanet_Cryo'] = home_planet + '_' + cryo_sleep
            out_data['IsEarthChild'] = (
                (home_planet == 'Earth') & (df['Age'].fillna(999.0) <= 12.0)
            ).astype(float)
            is_g_s_hotspot = (
                (deck == 'G')
                & (side == 'S')
                & (cabin_num >= 750.0)
                & (cabin_num <= 1200.0)
            )
            out_data['DeckGStarboardHotspot'] = is_g_s_hotspot.astype(float)

        return pd.DataFrame(out_data, index=df.index)


class PowerVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Weighted soft-voting and non-linear power-transformed probability consensus classifier.
    When power == 1.0, produces standard weighted linear soft voting.
    When power > 1.0, power-transforms probabilities before weighted combination to sharpen consensus.
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
            p_powers = [
                np.power(np.clip(pi, 1e-12, 1.0), self.power) for pi in probas
            ]
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

    base_cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    base_num_cols = [
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

    cat_cols = base_cat_cols + ['DeckSide']
    num_cols = base_num_cols + ['DeckGStarboardHotspot']

    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    num_transformer = Pipeline(
        [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[('cat', cat_transformer, cat_cols), ('num', num_transformer, num_cols)]
    )

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
    hgb_clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        random_state=42,
    )

    model_2way = [('lgbm', lgb_clf), ('cat', cat_clf)]
    model_3way = [('lgbm', lgb_clf), ('cat', cat_clf), ('hgb', hgb_clf)]

    consensus_options = {
        'parent_lgb_cat_linear': PowerVotingClassifier(
            estimators=model_2way, weights=[1, 2], power=1.0
        ),
        'lgb_cat_hgb_linear': PowerVotingClassifier(
            estimators=model_3way, weights=[2, 3, 1], power=1.0
        ),
        'lgb_cat_power_p2': PowerVotingClassifier(
            estimators=model_2way, weights=[1, 2], power=2.0
        ),
        'lgb_cat_power_p3': PowerVotingClassifier(
            estimators=model_2way, weights=[1, 2], power=3.0
        ),
        'lgb_cat_hgb_power_p2': PowerVotingClassifier(
            estimators=model_3way, weights=[2, 3, 1], power=2.0
        ),
    }

    chosen_model = skrub.choose_from(consensus_options, name='consensus_variant')

    chosen_pipeline = Pipeline(
        [
            ('party_fe', PartyFeatureEngineer(mode='deck_g_spatial')),
            ('prep', preprocessor),
            ('model', chosen_model),
        ]
    )

    pred = features.skb.apply(chosen_pipeline, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}