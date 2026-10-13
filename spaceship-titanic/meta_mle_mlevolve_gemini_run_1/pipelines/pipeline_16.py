import numpy as np
import skrub
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    deck = X['Cabin'].str.split('/').str[0].astype(object)
    cabin_num = X['Cabin'].str.split('/').str[1].astype('float64')
    side = X['Cabin'].str.split('/').str[2].astype(object)
    deck_side = deck.str.cat(side, sep='_').astype(object)
    deck = deck.where(deck.notna(), np.nan)
    side = side.where(side.notna(), np.nan)
    deck_side = deck_side.where(deck_side.notna(), np.nan)

    group_num = X['PassengerId'].str.split('_').str[1].astype('float64')

    room_service = X['RoomService'].fillna(0.0)
    food_court = X['FoodCourt'].fillna(0.0)
    shopping_mall = X['ShoppingMall'].fillna(0.0)
    spa = X['Spa'].fillna(0.0)
    vr_deck = X['VRDeck'].fillna(0.0)
    total_spend = room_service + food_court + shopping_mall + spa + vr_deck

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
        HomePlanet=X['HomePlanet'].astype(object),
        CryoSleep=cryo_sleep,
        Destination=X['Destination'].astype(object),
        VIP=X['VIP'].astype(object),
        TotalSpend=total_spend,
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
    ).skb.set_name('features')

    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide']
    num_cols = [
        'Age',
        'CabinNum',
        'GroupNum',
        'TotalSpend',
        'IsZeroSpend',
        'HasRoomService',
        'HasFoodCourt',
        'HasShoppingMall',
        'HasSpa',
        'HasVRDeck',
        'LogTotalSpend',
        'LogRoomService',
        'LogFoodCourt',
        'LogShoppingMall',
        'LogSpa',
        'LogVRDeck',
    ]

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
        ('scaler', StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
        ]
    )

    clf_c05 = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    ])
    clf_c10 = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    clf_c20 = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
    ])

    model = skrub.choose_from({'c05': clf_c05, 'c10': clf_c10, 'c20': clf_c20}, name='reg_c')

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}