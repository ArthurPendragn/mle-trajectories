import numpy as np
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
    deck_side = deck + '_' + side
    cabin_num = cabin.str.split('/').str[1].skb.apply_func(pd.to_numeric, errors='coerce')

    passenger_id = X['PassengerId']
    group_id = passenger_id.str.slice(0, 4).skb.apply_func(pd.to_numeric, errors='coerce')
    group_pos = passenger_id.str.slice(5, 7).skb.apply_func(pd.to_numeric, errors='coerce')

    amenity_missing = (
        X['RoomService'].isna().astype('float64') +
        X['FoodCourt'].isna().astype('float64') +
        X['ShoppingMall'].isna().astype('float64') +
        X['Spa'].isna().astype('float64') +
        X['VRDeck'].isna().astype('float64')
    )

    room_service = X['RoomService'].fillna(0.0)
    food_court = X['FoodCourt'].fillna(0.0)
    shopping_mall = X['ShoppingMall'].fillna(0.0)
    spa = X['Spa'].fillna(0.0)
    vr_deck = X['VRDeck'].fillna(0.0)

    total_spend = room_service + food_court + shopping_mall + spa + vr_deck
    has_spent = (total_spend > 0.0).astype('float64')
    spend_count = (
        (room_service > 0.0).astype('float64') +
        (food_court > 0.0).astype('float64') +
        (shopping_mall > 0.0).astype('float64') +
        (spa > 0.0).astype('float64') +
        (vr_deck > 0.0).astype('float64')
    )

    raw_cryo = X['CryoSleep'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    cryo_sleep = raw_cryo.mask(total_spend > 0.0, 0.0)
    cryo_missing = X['CryoSleep'].isna().astype('float64')

    vip = X['VIP'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})

    luxury_spend = room_service + spa + vr_deck
    communal_spend = food_court + shopping_mall
    luxury_share = luxury_spend / (total_spend + 1.0)
    communal_share = communal_spend / (total_spend + 1.0)

    is_child = (X['Age'] <= 12.0).fillna(False).astype('float64')
    log_total_spend = total_spend.skb.apply_func(np.log1p)
    spend_per_age = total_spend / (X['Age'].fillna(28.8) + 1.0)

    base_features = X[['HomePlanet', 'Destination', 'Age']].assign(
        RoomService=room_service,
        FoodCourt=food_court,
        ShoppingMall=shopping_mall,
        Spa=spa,
        VRDeck=vr_deck,
        AmenityMissingCount=amenity_missing,
        CryoSleep=cryo_sleep,
        VIP=vip,
        CabinDeck=deck,
        CabinNum=cabin_num,
        CabinSide=side,
        CabinDeckSide=deck_side,
        TotalSpend=total_spend,
        HasSpent=has_spent,
        SpendCount=spend_count,
        GroupId=group_id,
        GroupPos=group_pos,
    )

    features_decomp = base_features.assign(LuxurySpend=luxury_spend, CommunalSpend=communal_spend)
    features_base_clean = features_decomp.assign(LuxuryShare=luxury_share, IsChild=is_child)
    features_communal_log = features_base_clean.assign(CommunalShare=communal_share, LogTotalSpend=log_total_spend)
    features_spend_rate = features_communal_log.assign(SpendPerAge=spend_per_age, CryoMissing=cryo_missing)

    feature_candidates = {
        'clean_amenities_base': features_base_clean,
        'clean_amenities_communal_and_log': features_communal_log,
        'clean_amenities_spend_rate': features_spend_rate,
    }
    feature_choice = skrub.choose_from(feature_candidates, name='feature_engineering')
    features_op = skrub.as_data_op(feature_choice)
    encoded_features = features_op.skb.apply(skrub.TableVectorizer(cardinality_threshold=40)).skb.set_name('features')

    cat = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=4)
    lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=4, verbose=-1)
    model = VotingClassifier(estimators=[('cat', cat), ('lgb', lgb)], voting='soft', weights=[3, 1])

    pred = encoded_features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}