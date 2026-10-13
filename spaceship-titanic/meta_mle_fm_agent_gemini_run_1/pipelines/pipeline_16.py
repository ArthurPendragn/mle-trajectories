import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
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
    features = base_features.assign(CryoSleep=cryo_sleep, VIP=vip, CabinDeck=deck, CabinNum=cabin_num, CabinSide=side, TotalSpend=total_spend, HasSpent=has_spent, SpendCount=spend_count, GroupId=group_id, GroupPos=group_pos)
    encoded_features = features.skb.apply(skrub.TableVectorizer(cardinality_threshold=40)).skb.set_name('features')

    cat_clf_1 = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=4)
    lgb_clf_1 = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=4, verbose=-1)

    cat_clf_2 = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=4)
    lgb_clf_2 = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=4, verbose=-1)

    cat_clf_3 = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=4)
    lgb_clf_3 = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=4, verbose=-1)
    hist_clf_3 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42)

    model_choice = skrub.choose_from(
        {
            'cat_lgb_weighted': VotingClassifier(
                estimators=[('cat', cat_clf_1), ('lgb', lgb_clf_1)],
                voting='soft',
                weights=[2, 1],
            ),
            'cat_lgb_equal': VotingClassifier(
                estimators=[('cat', cat_clf_2), ('lgb', lgb_clf_2)],
                voting='soft',
                weights=[1, 1],
            ),
            'cat_lgb_hist': VotingClassifier(
                estimators=[('cat', cat_clf_3), ('lgb', lgb_clf_3), ('hist', hist_clf_3)],
                voting='soft',
                weights=[2, 1, 1],
            ),
        },
        name='ensemble_scheme',
    )
    pred = encoded_features.skb.apply(model_choice, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}