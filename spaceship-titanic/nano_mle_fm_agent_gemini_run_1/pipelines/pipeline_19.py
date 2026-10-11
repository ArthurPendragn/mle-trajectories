import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

    base_features = X[['HomePlanet', 'Destination', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
    features = base_features.assign(
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

    deck_side = deck + '_' + side
    is_zero_spend = (total_spend == 0.0).fillna(False)
    is_awake = (cryo_sleep == 0.0).fillna(False)
    zero_spend_awake = (is_zero_spend & is_awake).astype('float64')
    log_total_spend = total_spend.skb.apply_func(np.log1p)

    features_spatial = features.assign(CabinDeckSide=deck_side)
    features_behavioral = features.assign(
        ZeroSpendAwake=zero_spend_awake,
        LogTotalSpend=log_total_spend,
    )
    features_composite = features.assign(
        CabinDeckSide=deck_side,
        ZeroSpendAwake=zero_spend_awake,
        LogTotalSpend=log_total_spend,
    )

    features_choice = skrub.as_data_op(
        skrub.choose_from(
            {
                'spatial_deck_side': features_spatial,
                'behavioral_spend': features_behavioral,
                'composite_interaction': features_composite,
            },
            name='feature_interaction',
        )
    )

    encoded_features = features_choice.skb.apply(skrub.TableVectorizer(cardinality_threshold=40)).skb.set_name('features')

    model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=4)
    pred = encoded_features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}