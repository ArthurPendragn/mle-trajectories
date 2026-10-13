import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
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
    has_spent = (total_spend > 0.0).astype('float64')
    spend_count = (
        (room_service > 0.0).astype('float64')
        + (food_court > 0.0).astype('float64')
        + (shopping_mall > 0.0).astype('float64')
        + (spa > 0.0).astype('float64')
        + (vr_deck > 0.0).astype('float64')
    )

    cryo_sleep = X['CryoSleep'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    vip = X['VIP'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    home_planet = X['HomePlanet'].fillna('Missing')
    destination = X['Destination'].fillna('Missing')

    base_features = X[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
    features = base_features.assign(
        TotalSpend=total_spend,
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
        CabinSide=side,
    )

    num_cols = [
        'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'TotalSpend', 'HasSpent', 'SpendCount', 'CabinNum', 'GroupId', 'GroupPos',
        'CryoSleep', 'VIP'
    ]
    cat_cols = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide']

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
    ])
    encoded_features = features.skb.apply(preprocessor).skb.set_name('features')

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.001,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=250,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=42,
    )
    cb = CatBoostClassifier(
        iterations=500,
        learning_rate=0.04,
        depth=4,
        random_seed=42,
        verbose=0,
        thread_count=4,
    )

    model = skrub.choose_from(
        {
            'cb2_mlp1': VotingClassifier(
                estimators=[('catboost', cb), ('mlp', mlp)],
                voting='soft',
                weights=[2, 1],
            ),
            'cb3_mlp1': VotingClassifier(
                estimators=[('catboost', cb), ('mlp', mlp)],
                voting='soft',
                weights=[3, 1],
            ),
            'cb1_mlp1': VotingClassifier(
                estimators=[('catboost', cb), ('mlp', mlp)],
                voting='soft',
                weights=[1, 1],
            ),
        },
        name='ensemble_weights',
    )

    pred = encoded_features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}