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

    raw_cryo = X['CryoSleep'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    cryo_sleep = raw_cryo.mask(total_spend > 0.0, 0.0)

    home_planet = X['HomePlanet']
    is_deck_europa = deck.isin(['A', 'B', 'C', 'T']).fillna(False)
    is_deck_earth = (deck == 'G').fillna(False)
    hp_na = home_planet.isna().fillna(False)
    home_planet_clean = home_planet.mask(hp_na & is_deck_europa, 'Europa').mask(hp_na & is_deck_earth, 'Earth')

    raw_vip = X['VIP'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    is_earth = (home_planet_clean == 'Earth').fillna(False)
    vip_na = raw_vip.isna().fillna(False)
    vip_clean = raw_vip.mask(vip_na & is_earth, 0.0)

    luxury_spend = room_service + spa + vr_deck
    communal_spend = food_court + shopping_mall
    luxury_share = luxury_spend / (total_spend + 1.0)
    communal_share = communal_spend / (total_spend + 1.0)
    is_child = (X['Age'] <= 12.0).fillna(False).astype('float64')

    route = home_planet_clean + '_' + X['Destination']
    planet_deck = home_planet_clean + '_' + deck

    features_base = X[['Destination', 'Age']].assign(
        HomePlanet=home_planet_clean,
        CryoSleep=cryo_sleep,
        VIP=vip_clean,
        RoomService=room_service,
        FoodCourt=food_court,
        ShoppingMall=shopping_mall,
        Spa=spa,
        VRDeck=vr_deck,
        CabinDeck=deck,
        CabinNum=cabin_num,
        CabinSide=side,
        CabinDeckSide=deck_side,
        TotalSpend=total_spend,
        HasSpent=has_spent,
        SpendCount=spend_count,
        GroupId=group_id,
        GroupPos=group_pos,
        LuxurySpend=luxury_spend,
        CommunalSpend=communal_spend,
        LuxuryShare=luxury_share,
        IsChild=is_child,
    )

    features_route = features_base.assign(Route=route)
    features_route_communal = features_route.assign(CommunalShare=communal_share)
    features_route_planet_deck = features_route.assign(PlanetDeck=planet_deck)

    feature_choice = skrub.choose_from(
        {
            'clean_route': features_route,
            'clean_route_communal': features_route_communal,
            'clean_route_planet_deck': features_route_planet_deck,
        },
        name='feature_engineering',
    )
    features_op = skrub.as_data_op(feature_choice)
    encoded_features = features_op.skb.apply(
        skrub.TableVectorizer(cardinality_threshold=40)
    ).skb.set_name('features')

    cat = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=4)
    lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=4, verbose=-1)
    model = VotingClassifier(estimators=[('cat', cat), ('lgb', lgb)], voting='soft', weights=[3, 1])

    pred = encoded_features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}