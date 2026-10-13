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
    cabin_num = (
        cabin.str.split('/')
        .str[1]
        .skb.apply_func(pd.to_numeric, errors='coerce')
    )

    passenger_id = X['PassengerId']
    group_str = passenger_id.str.slice(0, 4)
    group_pos = (
        passenger_id.str.slice(5, 7).skb.apply_func(
            pd.to_numeric, errors='coerce'
        )
    )
    group_df = X[['PassengerId']].assign(Group=group_str)
    group_counts = (
        group_df.groupby('Group')[['PassengerId']]
        .agg('count')
        .rename(columns={'PassengerId': 'GroupSize'})
    )
    merged_group = group_df.join(group_counts, on='Group')
    group_size = merged_group['GroupSize'].astype('float64')
    is_alone = (group_size == 1.0).astype('float64')

    name = X['Name']
    surname = name.str.split(' ').str[-1]
    surname_df = X[['PassengerId']].assign(Surname=surname)
    surname_counts = (
        surname_df.groupby('Surname')[['PassengerId']]
        .agg('count')
        .rename(columns={'PassengerId': 'FamilySize'})
    )
    merged_family = surname_df.join(surname_counts, on='Surname')
    family_size = (
        merged_family['FamilySize'].fillna(1.0).astype('float64')
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
    is_child = (X['Age'] <= 12.0).fillna(False).astype('float64')
    spend_per_age = total_spend / (X['Age'] + 1.0)

    base_features = X[['HomePlanet', 'Destination', 'Age']].assign(
        RoomService=room_service,
        FoodCourt=food_court,
        ShoppingMall=shopping_mall,
        Spa=spa,
        VRDeck=vr_deck,
    )

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
        GroupPos=group_pos,
        GroupSize=group_size,
        IsAlone=is_alone,
        LuxurySpend=luxury_spend,
        CommunalSpend=communal_spend,
        LuxuryShare=luxury_share,
        IsChild=is_child,
    )

    features_communal = features_base.assign(CommunalShare=communal_share)
    features_full = features_communal.assign(
        FamilySize=family_size, SpendPerAge=spend_per_age
    )

    candidate_features = {
        'group_and_clean_amenities': features_base,
        'group_communal_and_clean': features_communal,
        'full_manifest_social_spatial': features_full,
    }
    feature_choice = skrub.choose_from(
        candidate_features, name='feature_engineering'
    )
    features_op = skrub.as_data_op(feature_choice)
    encoded_features = features_op.skb.apply(
        skrub.TableVectorizer(cardinality_threshold=40)
    ).skb.set_name('features')

    cat_500 = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=0,
        thread_count=4,
    )
    lgb_500 = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    model = VotingClassifier(
        estimators=[('cat', cat_500), ('lgb', lgb_500)],
        voting='soft',
        weights=[3, 1],
    )

    pred = encoded_features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}