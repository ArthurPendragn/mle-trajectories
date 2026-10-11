from catboost import CatBoostClassifier
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import skrub


class PartyFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.group_counts_ = df['GroupId'].value_counts().to_dict()
        surname = df['Surname'].dropna()
        self.surname_counts_ = surname.value_counts().to_dict()
        return self

    def transform(self, X):
        df = (
            X
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(
                X, columns=getattr(self, 'feature_names_in_', None)
            )
        )
        group_size = (
            df['GroupId'].map(self.group_counts_).fillna(1.0).astype('float64')
        )
        family_size = (
            df['Surname']
            .map(self.surname_counts_)
            .fillna(1.0)
            .astype('float64')
        )
        is_alone = (group_size == 1.0).astype('float64')
        out = df.drop(columns=['Surname'], errors='ignore').copy()
        out['GroupSize'] = group_size
        out['FamilySize'] = family_size
        out['IsAlone'] = is_alone
        return out


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    cabin = X['Cabin']
    deck = cabin.str.slice(0, 1)
    side = cabin.str.slice(-1, None)
    cabin_num = cabin.str.split('/').str[1].skb.apply_func(
        pd.to_numeric, errors='coerce'
    )

    passenger_id = X['PassengerId']
    group_id = passenger_id.str.slice(0, 4).skb.apply_func(
        pd.to_numeric, errors='coerce'
    )
    group_pos = passenger_id.str.slice(5, 7).skb.apply_func(
        pd.to_numeric, errors='coerce'
    )
    surname = X['Name'].str.split().str[-1]

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

    cryo_sleep = X['CryoSleep'].map(
        {True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0}
    )
    vip = X['VIP'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})

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
        Surname=surname,
    )

    party_features = features.skb.apply(PartyFeatureExtractor())
    encoded_features = party_features.skb.apply(
        skrub.TableVectorizer(cardinality_threshold=40)
    ).skb.set_name('features')

    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
        thread_count=4,
    )
    pred = encoded_features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}