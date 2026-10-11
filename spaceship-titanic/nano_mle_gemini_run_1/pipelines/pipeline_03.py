import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import skrub
from xgboost import XGBClassifier


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    amenity_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    amenities = X[amenity_cols].fillna(0.0)
    spend_sum = amenities.sum(axis=1)
    zero_spend = (spend_sum == 0.0).astype(float)
    log_total_spend = spend_sum.skb.apply_func(np.log1p)

    spend_denom = spend_sum.replace(0.0, 1.0)
    room_share = amenities['RoomService'] / spend_denom
    food_share = amenities['FoodCourt'] / spend_denom
    shop_share = amenities['ShoppingMall'] / spend_denom
    spa_share = amenities['Spa'] / spend_denom
    vr_share = amenities['VRDeck'] / spend_denom

    deck = X['Cabin'].str.slice(0, 1).fillna('Missing')
    side = X['Cabin'].str.slice(-1).fillna('Missing')
    cabin_num = X['Cabin'].str.split('/').str[1].astype(float)

    group_id = X['PassengerId'].str.slice(0, 4)
    group_num = X['PassengerId'].str.slice(5).astype(float)
    group_counts = X[['PassengerId']].assign(GroupId=group_id).groupby('GroupId', as_index=False).count().rename(columns={'PassengerId': 'GroupSize'})
    X_merged = X.assign(GroupId=group_id).merge(group_counts, on='GroupId', how='left')
    group_size = X_merged['GroupSize'].astype(float)
    is_alone = (group_size == 1.0).astype(float)

    home_planet = X['HomePlanet'].fillna('Missing').astype(str)
    cryo_sleep = X['CryoSleep'].fillna('Missing').astype(str)
    destination = X['Destination'].fillna('Missing').astype(str)
    vip = X['VIP'].fillna('Missing').astype(str)

    X_engineered = X_merged.assign(
        HomePlanet=home_planet,
        CryoSleep=cryo_sleep,
        Destination=destination,
        VIP=vip,
        Deck=deck,
        Side=side,
        CabinNum=cabin_num,
        GroupNum=group_num,
        GroupSize=group_size,
        IsAlone=is_alone,
        TotalSpend=spend_sum,
        ZeroSpend=zero_spend,
        LogTotalSpend=log_total_spend,
        RoomServiceShare=room_share,
        FoodCourtShare=food_share,
        ShoppingMallShare=shop_share,
        SpaShare=spa_share,
        VRDeckShare=vr_share,
    )
    features = X_engineered.drop(columns=['PassengerId', 'Cabin', 'Name', 'GroupId']).skb.set_name('engineered_features')

    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    num_cols = [
        'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'CabinNum', 'GroupNum', 'GroupSize', 'IsAlone', 'TotalSpend', 'ZeroSpend',
        'LogTotalSpend', 'RoomServiceShare', 'FoodCourtShare', 'ShoppingMallShare',
        'SpaShare', 'VRDeckShare',
    ]

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols),
    ])

    chosen_pipeline = skrub.choose_from({
        'xgb_depth4': Pipeline([
            ('prep', preprocessor),
            ('model', XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=8, eval_metric='logloss')),
        ]),
        'xgb_depth5': Pipeline([
            ('prep', preprocessor),
            ('model', XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=8, eval_metric='logloss')),
        ]),
        'lgbm_leaves15': Pipeline([
            ('prep', preprocessor),
            ('model', LGBMClassifier(num_leaves=15, learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=8, verbose=-1)),
        ]),
        'lgbm_leaves31': Pipeline([
            ('prep', preprocessor),
            ('model', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=8, verbose=-1)),
        ]),
    }, name='model_variant')

    pred = features.skb.apply(chosen_pipeline, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}