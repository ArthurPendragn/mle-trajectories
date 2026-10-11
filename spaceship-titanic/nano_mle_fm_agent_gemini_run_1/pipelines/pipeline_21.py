from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import VotingClassifier
import skrub

class GroupFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        group_str = X['PassengerId'].astype(str).str.slice(0, 4)
        self.counts_ = group_str.value_counts().to_dict()
        return self

    def transform(self, X):
        X_out = X.copy()
        group_str = X_out['PassengerId'].astype(str).str.slice(0, 4)
        size = group_str.map(self.counts_).fillna(1.0).astype('float64')
        X_out['GroupSize'] = size
        X_out['IsAlone'] = (size == 1.0).astype('float64')
        return X_out.drop(columns=['PassengerId'], errors='ignore')

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
    group_str = passenger_id.str.slice(0, 4)
    group_id = group_str.skb.apply_func(pd.to_numeric, errors='coerce')
    group_pos = passenger_id.str.slice(5, 7).skb.apply_func(pd.to_numeric, errors='coerce')

    room_service = X['RoomService'].fillna(0.0)
    food_court = X['FoodCourt'].fillna(0.0)
    shopping_mall = X['ShoppingMall'].fillna(0.0)
    spa = X['Spa'].fillna(0.0)
    vr_deck = X['VRDeck'].fillna(0.0)

    total_spend = room_service + food_court + shopping_mall + spa + vr_deck
    luxury_spend = room_service + spa + vr_deck
    communal_spend = food_court + shopping_mall
    luxury_ratio = luxury_spend / (total_spend + 1.0)
    has_spent = (total_spend > 0.0).astype('float64')
    spend_count = (
        (room_service > 0.0).astype('float64')
        + (food_court > 0.0).astype('float64')
        + (shopping_mall > 0.0).astype('float64')
        + (spa > 0.0).astype('float64')
        + (vr_deck > 0.0).astype('float64')
    )

    cryo_sleep = X['CryoSleep'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    cryo_sleep_imputed = cryo_sleep.mask(cryo_sleep.isna() & (total_spend > 0.0), 0.0)
    vip = X['VIP'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})

    base_features = X[['HomePlanet', 'Destination', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
    features_full_raw = base_features.assign(
        PassengerId=passenger_id,
        CryoSleep=cryo_sleep_imputed,
        VIP=vip,
        CabinDeck=deck,
        CabinNum=cabin_num,
        CabinSide=side,
        DeckSide=deck_side,
        TotalSpend=total_spend,
        LuxurySpend=luxury_spend,
        CommunalSpend=communal_spend,
        LuxuryRatio=luxury_ratio,
        HasSpent=has_spent,
        SpendCount=spend_count,
        GroupId=group_id,
        GroupPos=group_pos,
    )
    features_full = features_full_raw.skb.apply(GroupFeatureTransformer())
    encoded_features = features_full.skb.apply(skrub.TableVectorizer(cardinality_threshold=40)).skb.set_name('features')

    weights = skrub.choose_from(
        {'cat3_lgb1': [3, 1], 'cat2_lgb1': [2, 1], 'cat4_lgb1': [4, 1]},
        name='ensemble_weights',
    )
    model = VotingClassifier(
        estimators=[
            ('cat', CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=4)),
            ('lgb', LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=4, verbose=-1)),
        ],
        voting='soft',
        weights=weights,
    )
    pred = encoded_features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}