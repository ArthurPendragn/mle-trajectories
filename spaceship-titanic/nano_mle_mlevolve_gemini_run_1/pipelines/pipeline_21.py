import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier


class FeatureSubsetSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.cols is not None:
            return X[self.cols]
        return X


def make_spline_pipeline(n_knots, c_val, cont_cols, bin_cols, cat_cols):
    cont_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('spline', SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)),
        ('scaler', StandardScaler())
    ])
    bin_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('cont', cont_transformer, cont_cols),
        ('bin', bin_transformer, bin_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=c_val, max_iter=1000, random_state=42))
    ])


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    train_manifest = skrub.as_data_op(TRAIN_PATH).skb.apply_func(
        pd.read_csv, usecols=['PassengerId', 'HomePlanet', 'Cabin', 'Name']
    )
    test_manifest = skrub.as_data_op(TEST_PATH).skb.apply_func(
        pd.read_csv, usecols=['PassengerId', 'HomePlanet', 'Cabin', 'Name']
    )
    all_manifest = train_manifest.skb.concat([test_manifest], axis=0)

    all_pids = all_manifest['PassengerId']
    all_groups = all_pids.str.split('_').str[0]
    group_counts = all_groups.skb.apply_func(pd.Series.value_counts).to_frame(name='GroupSize')

    group_id = X['PassengerId'].str.split('_').str[0]
    group_num = X['PassengerId'].str.split('_').str[1].astype('float64')
    group_size = X[['PassengerId']].assign(group_id=group_id).join(group_counts, on='group_id')['GroupSize'].astype('float64')
    is_alone = (group_size == 1.0).astype('float64')

    all_names = all_manifest['Name'].dropna()
    all_surnames = all_names.str.split(' ').str[-1]
    surname_counts = all_surnames.skb.apply_func(pd.Series.value_counts).to_frame(name='FamilySize')
    passenger_surname = X['Name'].str.split(' ').str[-1]
    family_size = X[['PassengerId']].assign(surname=passenger_surname).join(surname_counts, on='surname')['FamilySize'].fillna(1.0).astype('float64')
    is_family_alone = (family_size == 1.0).astype('float64')

    manifest_deck = all_manifest['Cabin'].str.split('/').str[0]
    manifest_side = all_manifest['Cabin'].str.split('/').str[2]

    group_planet_df = (
        all_manifest[['HomePlanet']]
        .assign(group_id=all_groups)
        .dropna(subset=['HomePlanet'])
        .drop_duplicates(subset=['group_id'])
        .set_index('group_id')[['HomePlanet']]
        .rename(columns={'HomePlanet': 'GroupHomePlanet'})
    )
    companion_home_planet = X[['PassengerId']].assign(group_id=group_id).join(group_planet_df, on='group_id')['GroupHomePlanet'].astype(object)

    group_deck_df = (
        all_manifest[['PassengerId']]
        .assign(group_id=all_groups, deck=manifest_deck)
        .dropna(subset=['deck'])
        .drop_duplicates(subset=['group_id'])
        .set_index('group_id')[['deck']]
        .rename(columns={'deck': 'GroupDeck'})
    )
    companion_deck = X[['PassengerId']].assign(group_id=group_id).join(group_deck_df, on='group_id')['GroupDeck'].astype(object)

    group_side_df = (
        all_manifest[['PassengerId']]
        .assign(group_id=all_groups, side=manifest_side)
        .dropna(subset=['side'])
        .drop_duplicates(subset=['group_id'])
        .set_index('group_id')[['side']]
        .rename(columns={'side': 'GroupSide'})
    )
    companion_side = X[['PassengerId']].assign(group_id=group_id).join(group_side_df, on='group_id')['GroupSide'].astype(object)

    deck = X['Cabin'].str.split('/').str[0].astype(object)
    cabin_num = X['Cabin'].str.split('/').str[1].astype('float64')
    side = X['Cabin'].str.split('/').str[2].astype(object)

    deck = deck.fillna(companion_deck)
    side = side.fillna(companion_side)
    deck_side = deck.str.cat(side, sep='_').astype(object)
    deck = deck.where(deck.notna(), np.nan)
    side = side.where(side.notna(), np.nan)
    deck_side = deck_side.where(deck_side.notna(), np.nan)

    is_europa_deck = deck.isin(['A', 'B', 'C', 'T']).fillna(False)
    is_earth_deck = (deck == 'G').fillna(False)

    home_planet = X['HomePlanet'].astype(object)
    home_planet = home_planet.fillna(companion_home_planet)
    home_planet = home_planet.mask(home_planet.isna() & is_europa_deck, 'Europa')
    home_planet = home_planet.mask(home_planet.isna() & is_earth_deck, 'Earth')
    home_planet = home_planet.where(home_planet.notna(), np.nan).astype(object)

    room_service = X['RoomService'].fillna(0.0)
    food_court = X['FoodCourt'].fillna(0.0)
    shopping_mall = X['ShoppingMall'].fillna(0.0)
    spa = X['Spa'].fillna(0.0)
    vr_deck = X['VRDeck'].fillna(0.0)
    service_spend = room_service + spa + vr_deck
    leisure_spend = food_court + shopping_mall
    total_spend = service_spend + leisure_spend

    is_zero_spend = (total_spend == 0.0).astype('float64')
    has_room_service = (room_service > 0.0).astype('float64')
    has_food_court = (food_court > 0.0).astype('float64')
    has_shopping_mall = (shopping_mall > 0.0).astype('float64')
    has_spa = (spa > 0.0).astype('float64')
    has_vr_deck = (vr_deck > 0.0).astype('float64')

    log_total_spend = total_spend.skb.apply_func(np.log1p)
    log_room_service = room_service.skb.apply_func(np.log1p)
    log_food_court = food_court.skb.apply_func(np.log1p)
    log_shopping_mall = shopping_mall.skb.apply_func(np.log1p)
    log_spa = spa.skb.apply_func(np.log1p)
    log_vr_deck = vr_deck.skb.apply_func(np.log1p)
    log_service_spend = service_spend.skb.apply_func(np.log1p)
    log_leisure_spend = leisure_spend.skb.apply_func(np.log1p)
    log_service_leisure_diff = log_service_spend - log_leisure_spend

    cryo_sleep_raw = X['CryoSleep'].astype(object)
    should_impute_cryo = cryo_sleep_raw.isna() & (total_spend > 0.0)
    cryo_sleep = cryo_sleep_raw.mask(should_impute_cryo, False)
    cryo_sleep = cryo_sleep.where(cryo_sleep.notna(), np.nan).astype(object)

    age_safe = X['Age'].fillna(28.0) + 1.0
    spend_per_age = (total_spend / age_safe).skb.apply_func(np.log1p)
    service_ratio = service_spend / (total_spend + 1.0)
    leisure_ratio = leisure_spend / (total_spend + 1.0)

    base_cols = [
        'Age', 'CabinNum', 'GroupNum', 'GroupSize', 'FamilySize',
        'Deck', 'Side', 'DeckSide', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP',
        'IsAlone', 'IsFamilyAlone', 'IsZeroSpend',
        'HasRoomService', 'HasFoodCourt', 'HasShoppingMall', 'HasSpa', 'HasVRDeck',
        'LogTotalSpend', 'LogRoomService', 'LogFoodCourt', 'LogShoppingMall', 'LogSpa', 'LogVRDeck', 'LogServiceLeisureDiff'
    ]

    cont_cols_base = ['Age', 'CabinNum', 'GroupNum', 'GroupSize', 'FamilySize', 'LogTotalSpend', 'LogRoomService', 'LogFoodCourt', 'LogShoppingMall', 'LogSpa', 'LogVRDeck', 'LogServiceLeisureDiff']
    cont_cols_full = cont_cols_base + ['SpendPerAge', 'ServiceRatio', 'LeisureRatio']
    bin_cols = ['IsZeroSpend', 'IsAlone', 'IsFamilyAlone', 'HasRoomService', 'HasFoodCourt', 'HasShoppingMall', 'HasSpa', 'HasVRDeck']
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide']

    features = X[['Age']].assign(
        Deck=deck,
        CabinNum=cabin_num,
        Side=side,
        DeckSide=deck_side,
        GroupNum=group_num,
        GroupSize=group_size,
        FamilySize=family_size,
        IsAlone=is_alone,
        IsFamilyAlone=is_family_alone,
        HomePlanet=home_planet,
        CryoSleep=cryo_sleep,
        Destination=X['Destination'].astype(object),
        VIP=X['VIP'].astype(object),
        IsZeroSpend=is_zero_spend,
        HasRoomService=has_room_service,
        HasFoodCourt=has_food_court,
        HasShoppingMall=has_shopping_mall,
        HasSpa=has_spa,
        HasVRDeck=has_vr_deck,
        LogTotalSpend=log_total_spend,
        LogRoomService=log_room_service,
        LogFoodCourt=log_food_court,
        LogShoppingMall=log_shopping_mall,
        LogSpa=log_spa,
        LogVRDeck=log_vr_deck,
        LogServiceLeisureDiff=log_service_leisure_diff,
        SpendPerAge=spend_per_age,
        ServiceRatio=service_ratio,
        LeisureRatio=leisure_ratio
    ).skb.set_name('features')

    pipe_gam_base = make_spline_pipeline(8, 1.0, cont_cols_base, bin_cols, cat_cols)
    pipe_lgb_base = make_pipeline(
        FeatureSubsetSelector(cols=base_cols),
        skrub.TableVectorizer(),
        LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=8)
    )
    pipe_xgb_base = make_pipeline(
        FeatureSubsetSelector(cols=base_cols),
        skrub.TableVectorizer(),
        XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, eval_metric='logloss', tree_method='hist', n_jobs=8)
    )
    model_base = StackingClassifier(
        estimators=[('gam', pipe_gam_base), ('lgb', pipe_lgb_base), ('xgb', pipe_xgb_base)],
        final_estimator=LogisticRegression(C=0.1, random_state=42),
        cv=5
    )

    pipe_gam_full = make_spline_pipeline(8, 1.0, cont_cols_full, bin_cols, cat_cols)
    pipe_lgb_full = make_pipeline(
        skrub.TableVectorizer(),
        LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=8)
    )
    pipe_xgb_full = make_pipeline(
        skrub.TableVectorizer(),
        XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, eval_metric='logloss', tree_method='hist', n_jobs=8)
    )
    model_full = StackingClassifier(
        estimators=[('gam', pipe_gam_full), ('lgb', pipe_lgb_full), ('xgb', pipe_xgb_full)],
        final_estimator=LogisticRegression(C=0.1, random_state=42),
        cv=5
    )

    model = skrub.choose_from({'group_and_family': model_base, 'full_with_ratios': model_full}, name='feature_set')
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}