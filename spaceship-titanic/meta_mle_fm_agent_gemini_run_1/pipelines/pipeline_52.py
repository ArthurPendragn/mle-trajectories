import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import skrub


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    cabin = X['Cabin']
    deck = cabin.str.slice(0, 1)
    side = cabin.str.slice(-1, None)
    deck_side = deck + '_' + side
    cabin_num = (
        cabin.str.split('/')
        .str[1]
        .skb.apply_func(pd.to_numeric, errors='coerce')
    )
    passenger_id = X['PassengerId']
    group_id = (
        passenger_id.str.slice(0, 4).skb.apply_func(
            pd.to_numeric, errors='coerce'
        )
    )
    group_pos = (
        passenger_id.str.slice(5, 7).skb.apply_func(
            pd.to_numeric, errors='coerce'
        )
    )
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
    raw_cryo = X['CryoSleep'].map(
        {True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0}
    )
    cryo_sleep = raw_cryo.mask(total_spend > 0.0, 0.0)
    vip = X['VIP'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    luxury_spend = room_service + spa + vr_deck
    communal_spend = food_court + shopping_mall
    luxury_share = luxury_spend / (total_spend + 1.0)
    communal_share = communal_spend / (total_spend + 1.0)
    spend_orientation = (communal_spend - luxury_spend) / (total_spend + 1.0)
    has_luxury = (luxury_spend > 0.0).astype('float64')
    has_communal = (communal_spend > 0.0).astype('float64')
    luxury_only = ((luxury_spend > 0.0) & (communal_spend == 0.0)).astype(
        'float64'
    )
    communal_only = ((communal_spend > 0.0) & (luxury_spend == 0.0)).astype(
        'float64'
    )
    is_child = (X['Age'] <= 12.0).fillna(False).astype('float64')
    base_features = X[[
        'HomePlanet',
        'Destination',
        'Age',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck',
    ]]
    features_base = base_features.assign(
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
    features_decomp = features_base.assign(
        LuxurySpend=luxury_spend, CommunalSpend=communal_spend
    )
    features_child = features_decomp.assign(
        LuxuryShare=luxury_share, IsChild=is_child
    )
    features_location = features_child.assign(
        HasLuxurySpend=has_luxury,
        HasCommunalSpend=has_communal,
        LuxuryOnly=luxury_only,
        CommunalOnly=communal_only,
    )
    features_orient = features_location.assign(
        CommunalShare=communal_share, SpendOrientation=spend_orientation
    )
    encoded_features = features_orient.skb.apply(
        skrub.TableVectorizer(cardinality_threshold=40)
    ).skb.set_name('features')

    cat_vote = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
        thread_count=4,
    )
    lgb_vote = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    xgb_vote = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        eval_metric='logloss',
    )

    cat_stack = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
        thread_count=4,
    )
    lgb_stack = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    xgb_stack = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        eval_metric='logloss',
    )

    voting_model = VotingClassifier(
        estimators=[('cat', cat_vote), ('lgb', lgb_vote), ('xgb', xgb_vote)],
        voting='soft',
        weights=[6, 1, 1],
    )
    stacking_model = StackingClassifier(
        estimators=[
            ('cat', cat_stack),
            ('lgb', lgb_stack),
            ('xgb', xgb_stack),
        ],
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5,
        n_jobs=1,
    )

    chosen_model = skrub.choose_from(
        {'voting_w611': voting_model, 'stacking_lr': stacking_model},
        name='model_architecture',
    )
    pred = encoded_features.skb.apply(chosen_model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}