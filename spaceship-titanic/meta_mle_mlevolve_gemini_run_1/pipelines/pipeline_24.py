import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier


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
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', cont_transformer, cont_cols),
            ('bin', bin_transformer, bin_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=c_val, max_iter=1000, random_state=42))
    ])


def make_ensemble(cont_cols, bin_cols, cat_cols):
    active_cols = cont_cols + bin_cols + cat_cols
    pipe_gam = make_spline_pipeline(8, 1.0, cont_cols, bin_cols, cat_cols)
    pipe_lgb = Pipeline([
        ('sel', ColumnTransformer([('pass', 'passthrough', active_cols)])),
        ('vec', skrub.TableVectorizer()),
        ('clf', LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=8))
    ])
    pipe_xgb = Pipeline([
        ('sel', ColumnTransformer([('pass', 'passthrough', active_cols)])),
        ('vec', skrub.TableVectorizer()),
        ('clf', XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, eval_metric='logloss', tree_method='hist', n_jobs=8))
    ])
    return VotingClassifier(
        estimators=[('gam', pipe_gam), ('lgb', pipe_lgb), ('xgb', pipe_xgb)],
        voting='soft',
        weights=[1, 2, 2]
    )


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    train_pids = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv, usecols=['PassengerId'])
    test_pids = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv, usecols=['PassengerId'])
    all_pids = train_pids.skb.concat([test_pids], axis=0)
    all_groups = all_pids['PassengerId'].str.split('_').str[0]
    group_counts = all_groups.skb.apply_func(pd.Series.value_counts).to_frame(name='GroupSize')

    train_names = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv, usecols=['PassengerId', 'Name'])
    test_names = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv, usecols=['PassengerId', 'Name'])
    all_names = train_names.skb.concat([test_names], axis=0)
    all_surnames = all_names['Name'].str.split().str[-1]
    surname_counts = all_surnames.skb.apply_func(pd.Series.value_counts).to_frame(name='FamilySize')

    group_id = X['PassengerId'].str.split('_').str[0]
    group_num = X['PassengerId'].str.split('_').str[1].astype('float64')
    group_size = X[['PassengerId']].assign(group_id=group_id).join(group_counts, on='group_id')['GroupSize'].astype('float64')
    is_alone = (group_size == 1.0).astype('float64')

    surname = X['Name'].str.split().str[-1]
    family_size = X[['PassengerId']].assign(surname=surname).join(surname_counts, on='surname')['FamilySize'].astype('float64').fillna(1.0)
    is_family_alone = (family_size == 1.0).astype('float64')

    deck = X['Cabin'].str.split('/').str[0].astype(object)
    cabin_num = X['Cabin'].str.split('/').str[1].astype('float64')
    side = X['Cabin'].str.split('/').str[2].astype(object)
    deck_side = deck.str.cat(side, sep='_').astype(object)
    deck = deck.where(deck.notna(), np.nan)
    side = side.where(side.notna(), np.nan)
    deck_side = deck_side.where(deck_side.notna(), np.nan)

    is_europa_deck = deck.isin(['A', 'B', 'C', 'T']).fillna(False)
    is_earth_deck = (deck == 'G').fillna(False)
    home_planet = X['HomePlanet'].astype(object)
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

    spend_denom = total_spend + 1.0
    pct_service_spend = service_spend / spend_denom
    pct_room_service = room_service / spend_denom
    pct_food_court = food_court / spend_denom
    pct_shopping_mall = shopping_mall / spend_denom
    pct_spa = spa / spend_denom
    pct_vr_deck = vr_deck / spend_denom

    log_total_spend = total_spend.skb.apply_func(np.log1p)
    log_room_service = room_service.skb.apply_func(np.log1p)
    log_food_court = food_court.skb.apply_func(np.log1p)
    log_shopping_mall = shopping_mall.skb.apply_func(np.log1p)
    log_spa = spa.skb.apply_func(np.log1p)
    log_vr_deck = vr_deck.skb.apply_func(np.log1p)
    log_service_spend = service_spend.skb.apply_func(np.log1p)
    log_leisure_spend = leisure_spend.skb.apply_func(np.log1p)
    log_service_leisure_diff = log_service_spend - log_leisure_spend

    is_child = ((X['Age'] >= 0.0) & (X['Age'] <= 12.0)).astype('float64').fillna(0.0)

    cryo_sleep_raw = X['CryoSleep'].astype(object)
    should_impute_cryo = cryo_sleep_raw.isna() & (total_spend > 0.0)
    cryo_sleep = cryo_sleep_raw.mask(should_impute_cryo, False)
    cryo_sleep = cryo_sleep.where(cryo_sleep.notna(), np.nan).astype(object)

    features = X[['Age']].assign(
        Deck=deck,
        CabinNum=cabin_num,
        Side=side,
        DeckSide=deck_side,
        GroupNum=group_num,
        GroupSize=group_size,
        IsAlone=is_alone,
        FamilySize=family_size,
        IsFamilyAlone=is_family_alone,
        IsChild=is_child,
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
        PctServiceSpend=pct_service_spend,
        PctRoomService=pct_room_service,
        PctFoodCourt=pct_food_court,
        PctShoppingMall=pct_shopping_mall,
        PctSpa=pct_spa,
        PctVRDeck=pct_vr_deck
    ).skb.set_name('features')

    cont_cols_base = ['Age', 'CabinNum', 'GroupNum', 'GroupSize', 'LogTotalSpend', 'LogRoomService', 'LogFoodCourt', 'LogShoppingMall', 'LogSpa', 'LogVRDeck', 'LogServiceLeisureDiff']
    bin_cols_base = ['IsZeroSpend', 'IsAlone', 'HasRoomService', 'HasFoodCourt', 'HasShoppingMall', 'HasSpa', 'HasVRDeck']
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide']

    pct_cols = ['PctServiceSpend', 'PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa', 'PctVRDeck']
    family_cont_cols = ['FamilySize']
    family_bin_cols = ['IsFamilyAlone']
    demo_bin_cols = ['IsChild']

    cont_spend = cont_cols_base + pct_cols
    bin_spend = bin_cols_base

    cont_social = cont_cols_base + family_cont_cols
    bin_social = bin_cols_base + family_bin_cols + demo_bin_cols

    cont_full = cont_cols_base + pct_cols + family_cont_cols
    bin_full = bin_cols_base + family_bin_cols + demo_bin_cols

    model = skrub.choose_from({
        'full_domain': make_ensemble(cont_full, bin_full, cat_cols),
        'spend_shares_only': make_ensemble(cont_spend, bin_spend, cat_cols),
        'social_demographic': make_ensemble(cont_social, bin_social, cat_cols)
    }, name='feature_expansion')

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}