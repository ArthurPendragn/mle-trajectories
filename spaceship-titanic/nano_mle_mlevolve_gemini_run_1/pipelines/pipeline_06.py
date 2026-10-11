import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    cabin_split = X['Cabin'].str.split('/', expand=True)
    deck = cabin_split[0].astype('string')
    cabin_num = cabin_split[1].astype('float64')
    side = cabin_split[2].astype('string')
    group_num = X['PassengerId'].str[5:].astype('float64')

    room_service_filled = X['RoomService'].fillna(0.0)
    food_court_filled = X['FoodCourt'].fillna(0.0)
    shopping_mall_filled = X['ShoppingMall'].fillna(0.0)
    spa_filled = X['Spa'].fillna(0.0)
    vr_deck_filled = X['VRDeck'].fillna(0.0)
    total_spend = room_service_filled + food_court_filled + shopping_mall_filled + spa_filled + vr_deck_filled
    has_spend = (total_spend > 0.0).astype('float64')

    log_room_service = room_service_filled.skb.apply_func(np.log1p)
    log_food_court = food_court_filled.skb.apply_func(np.log1p)
    log_shopping_mall = shopping_mall_filled.skb.apply_func(np.log1p)
    log_spa = spa_filled.skb.apply_func(np.log1p)
    log_vr_deck = vr_deck_filled.skb.apply_func(np.log1p)
    log_total_spend = total_spend.skb.apply_func(np.log1p)

    cryo_sleep = X['CryoSleep'].astype('string')
    vip = X['VIP'].astype('string')

    features = X.assign(
        RoomService=log_room_service,
        FoodCourt=log_food_court,
        ShoppingMall=log_shopping_mall,
        Spa=log_spa,
        VRDeck=log_vr_deck,
        TotalSpend=log_total_spend,
        HasSpend=has_spend,
        Deck=deck,
        CabinNum=cabin_num,
        Side=side,
        GroupNum=group_num,
        CryoSleep=cryo_sleep,
        VIP=vip,
    ).drop(columns=['PassengerId', 'Cabin', 'Name']).skb.set_name('features')

    pipe_lgb = make_pipeline(
        skrub.TableVectorizer(),
        LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=8),
    )
    pipe_xgb = make_pipeline(
        skrub.TableVectorizer(),
        XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, eval_metric='logloss', tree_method='hist', n_jobs=8),
    )
    pipe_hgb = make_pipeline(
        skrub.TableVectorizer(),
        HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=31, random_state=42),
    )

    models = {
        'dual_lgb_xgb': VotingClassifier(
            estimators=[('lgb', pipe_lgb), ('xgb', pipe_xgb)],
            voting='soft',
        ),
        'triple_lgb_xgb_hgb': VotingClassifier(
            estimators=[('lgb', pipe_lgb), ('xgb', pipe_xgb), ('hgb', pipe_hgb)],
            voting='soft',
        ),
    }

    model_choice = skrub.choose_from(models, name='ensemble_type')
    pred = features.skb.apply(model_choice, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': scoring}