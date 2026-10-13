import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skrub

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    cabin = X['Cabin']
    deck = cabin.str.slice(0, 1).fillna('Missing')
    side = cabin.str.slice(-1, None).fillna('Missing')
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
    log_room_service = room_service.skb.apply_func(np.log1p)
    log_food_court = food_court.skb.apply_func(np.log1p)
    log_shopping_mall = shopping_mall.skb.apply_func(np.log1p)
    log_spa = spa.skb.apply_func(np.log1p)
    log_vr_deck = vr_deck.skb.apply_func(np.log1p)
    log_total_spend = total_spend.skb.apply_func(np.log1p)
    has_spent = (total_spend > 0.0).astype('float64')
    spend_count = (room_service > 0.0).astype('float64') + (food_court > 0.0).astype('float64') + (shopping_mall > 0.0).astype('float64') + (spa > 0.0).astype('float64') + (vr_deck > 0.0).astype('float64')
    cryo_sleep = X['CryoSleep'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    vip = X['VIP'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    home_planet = X['HomePlanet'].fillna('Missing')
    destination = X['Destination'].fillna('Missing')
    base_features = X[['Age']]
    features = base_features.assign(
        RoomService=log_room_service,
        FoodCourt=log_food_court,
        ShoppingMall=log_shopping_mall,
        Spa=log_spa,
        VRDeck=log_vr_deck,
        TotalSpend=log_total_spend,
        HasSpent=has_spent,
        SpendCount=spend_count,
        CabinNum=cabin_num,
        GroupId=group_id,
        GroupPos=group_pos,
        CryoSleep=cryo_sleep,
        VIP=vip,
        HomePlanet=home_planet,
        Destination=destination,
        CabinDeck=deck,
        CabinSide=side
    )
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'HasSpent', 'SpendCount', 'CabinNum', 'GroupId', 'GroupPos', 'CryoSleep', 'VIP']
    cat_cols = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide']
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])
    encoded_features = features.skb.apply(preprocessor).skb.set_name('features')
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', alpha=0.001, batch_size=128, learning_rate_init=0.001, max_iter=250, early_stopping=True, n_iter_no_change=15, random_state=42)
    pred = encoded_features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}