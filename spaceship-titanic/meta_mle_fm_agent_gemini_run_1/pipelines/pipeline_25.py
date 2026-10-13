import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
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
    spend_count = (room_service > 0.0).astype('float64') + (food_court > 0.0).astype('float64') + (shopping_mall > 0.0).astype('float64') + (spa > 0.0).astype('float64') + (vr_deck > 0.0).astype('float64')
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
        GroupPos=group_pos
    )
    deck_side = deck + '_' + side
    luxury_spend = room_service + spa + vr_deck
    communal_spend = food_court + shopping_mall
    luxury_ratio = luxury_spend / (total_spend + 1.0)

    features_spatial = features.assign(CabinDeckSide=deck_side)
    features_amenity = features.assign(
        CabinDeckSide=deck_side,
        LuxurySpend=luxury_spend,
        CommunalSpend=communal_spend
    )
    features_ratio = features.assign(
        CabinDeckSide=deck_side,
        LuxurySpend=luxury_spend,
        CommunalSpend=communal_spend,
        LuxuryRatio=luxury_ratio
    )

    feature_choice = skrub.choose_from({
        'spatial_deck_side': features_spatial,
        'spatial_amenity_decomp': features_amenity,
        'spatial_amenity_ratio': features_ratio
    }, name='feature_engineering')

    encoded_features = skrub.as_data_op(feature_choice).skb.apply(skrub.TableVectorizer(cardinality_threshold=40)).skb.set_name('features')

    cat_est = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=4)
    lgb_est = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=4, verbose=-1)
    model = VotingClassifier(estimators=[('cat', cat_est), ('lgb', lgb_est)], voting='soft', weights=[3, 1])

    pred = encoded_features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}