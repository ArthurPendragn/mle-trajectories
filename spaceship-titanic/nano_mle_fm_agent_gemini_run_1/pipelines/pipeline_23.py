import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
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
    features_spatial = features.assign(CabinDeckSide=deck_side)

    encoded_features = features_spatial.skb.apply(
        skrub.TableVectorizer(cardinality_threshold=40)
    ).skb.set_name('features')

    cat_300 = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=4)
    lgb_300 = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=4, verbose=-1)
    xgb_300 = XGBClassifier(n_estimators=300, max_depth=5, colsample_bytree=0.8, learning_rate=0.05, random_state=42, n_jobs=4, eval_metric='logloss')

    cat_500 = CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, random_seed=42, verbose=0, thread_count=4)
    lgb_500 = LGBMClassifier(n_estimators=500, learning_rate=0.03, random_state=42, n_jobs=4, verbose=-1)
    xgb_500 = XGBClassifier(n_estimators=500, max_depth=5, colsample_bytree=0.8, learning_rate=0.03, random_state=42, n_jobs=4, eval_metric='logloss')

    ensemble_choices = {
        'cat_lgb_fine_500': VotingClassifier(
            estimators=[('cat', cat_500), ('lgb', lgb_500)],
            voting='soft',
            weights=[3, 1],
        ),
        'cat_lgb_xgb_triad': VotingClassifier(
            estimators=[('cat', cat_300), ('lgb', lgb_300), ('xgb', xgb_300)],
            voting='soft',
            weights=[3, 1, 1],
        ),
        'cat_lgb_xgb_fine_500': VotingClassifier(
            estimators=[('cat', cat_500), ('lgb', lgb_500), ('xgb', xgb_500)],
            voting='soft',
            weights=[3, 1, 1],
        ),
    }

    model_choice = skrub.choose_from(ensemble_choices, name='model_ensemble')
    pred = encoded_features.skb.apply(model_choice, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}