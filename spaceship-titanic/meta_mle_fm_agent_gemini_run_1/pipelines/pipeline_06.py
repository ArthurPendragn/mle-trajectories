import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skrub

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    cabin = X['Cabin']
    deck = cabin.str.slice(0, 1)
    side = cabin.str.slice(-1, None)
    cabin_num = cabin.str.split('/').str[1].skb.apply_func(pd.to_numeric, errors='coerce')

    passenger_id = X['PassengerId']
    group_id = passenger_id.str.slice(0, 4).skb.apply_func(pd.to_numeric, errors='coerce')
    group_pos = passenger_id.str.slice(5, 7).skb.apply_func(pd.to_numeric, errors='coerce')

    room_service = X['RoomService'].fillna(0.0)
    food_court = X['FoodCourt'].fillna(0.0)
    shopping_mall = X['ShoppingMall'].fillna(0.0)
    spa = X['Spa'].fillna(0.0)
    vr_deck = X['VRDeck'].fillna(0.0)
    total_spend = room_service + food_court + shopping_mall + spa + vr_deck

    has_spent = (total_spend > 0.0).map({True: 'True', False: 'False'})
    spend_count = (
        (room_service > 0.0).astype('float64')
        + (food_court > 0.0).astype('float64')
        + (shopping_mall > 0.0).astype('float64')
        + (spa > 0.0).astype('float64')
        + (vr_deck > 0.0).astype('float64')
    )

    cryo_sleep = X['CryoSleep'].map({True: 'True', False: 'False', 'True': 'True', 'False': 'False'})
    vip = X['VIP'].map({True: 'True', False: 'False', 'True': 'True', 'False': 'False'})

    base_features = X[['HomePlanet', 'Destination', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
    features_none = base_features.assign(
        CryoSleep=cryo_sleep,
        VIP=vip,
        CabinDeck=deck,
        CabinNum=cabin_num,
        CabinSide=side,
        TotalSpend=total_spend,
        HasSpent=has_spent,
        SpendCount=spend_count,
        GroupId=group_id,
        GroupPos=group_pos,
    )

    features_log1p = features_none.assign(
        RoomService=X['RoomService'].skb.apply_func(np.log1p),
        FoodCourt=X['FoodCourt'].skb.apply_func(np.log1p),
        ShoppingMall=X['ShoppingMall'].skb.apply_func(np.log1p),
        Spa=X['Spa'].skb.apply_func(np.log1p),
        VRDeck=X['VRDeck'].skb.apply_func(np.log1p),
        TotalSpend=total_spend.skb.apply_func(np.log1p),
    )

    spend_choice = skrub.choose_from(
        {'none': features_none, 'log1p': features_log1p},
        name='spend_transform',
    )
    features = skrub.as_data_op(spend_choice)

    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'SpendCount', 'CabinNum', 'GroupId', 'GroupPos']
    cat_cols = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'CabinDeck', 'CabinSide', 'HasSpent']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    preprocessed_features = features.skb.apply(preprocessor).skb.set_name('features')

    model = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    pred = preprocessed_features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}