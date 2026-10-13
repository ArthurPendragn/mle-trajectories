"""CatBoost gradient boosting pipeline with enriched route, deck-planet interaction, and expenditure ratio features.
Expands upon the top-performing CatBoost model by introducing travel route (HomePlanet to Destination), deck-planet interactions, expenditure relative proportions, and senior age indicators.
Evaluates depth-8 CatBoost configurations alongside a multi-depth soft-voting ensemble to capture both fine-grained interactions and diverse tree depths.
This improves model generalization on passenger survival patterns across distinct cabins and passenger segments.
"""
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder
import skrub


def build():
    eval_setup = build_evaluation()
    X = eval_setup['X']
    y = eval_setup['y']

    deck = X['Cabin'].str.slice(0, 1)
    side = X['Cabin'].str.slice(-1)
    deck_side = deck + '_' + side
    cabin_num = X['Cabin'].str.split('/').str[1].astype('float64')
    cabin_region = (cabin_num // 300).astype('float64')

    deck_planet = deck + '_' + X['HomePlanet'].astype('string')
    route = X['HomePlanet'].astype('string') + '_to_' + X['Destination'].astype('string')

    rs = X['RoomService'].fillna(0.0)
    fc = X['FoodCourt'].fillna(0.0)
    sm = X['ShoppingMall'].fillna(0.0)
    spa = X['Spa'].fillna(0.0)
    vr = X['VRDeck'].fillna(0.0)

    amenity_spend = rs + fc + sm + spa + vr
    service_spend = rs + spa + vr
    basic_spend = fc + sm
    spend_count = (
        (rs > 0.0).astype('float64')
        + (fc > 0.0).astype('float64')
        + (sm > 0.0).astype('float64')
        + (spa > 0.0).astype('float64')
        + (vr > 0.0).astype('float64')
    )
    has_spent = (amenity_spend > 0.0).astype('float64')

    service_ratio = service_spend / (amenity_spend + 1.0)
    basic_ratio = basic_spend / (amenity_spend + 1.0)
    spa_vr_ratio = (spa + vr) / (amenity_spend + 1.0)
    food_mall_ratio = (fc + sm) / (amenity_spend + 1.0)

    is_child = (X['Age'] < 13.0).astype('float64')
    is_senior = (X['Age'] >= 60.0).astype('float64')

    clean_X = X.assign(
        Deck=deck,
        Side=side,
        DeckSide=deck_side,
        DeckPlanet=deck_planet,
        Route=route,
        CabinNum=cabin_num,
        CabinRegion=cabin_region,
        GroupId=X['PassengerId'].str.slice(0, 4).astype('float64'),
        GroupNum=X['PassengerId'].str.slice(5, 7).astype('float64'),
        TotalSpend=amenity_spend,
        ServiceSpend=service_spend,
        BasicSpend=basic_spend,
        SpendCount=spend_count,
        HasSpent=has_spent,
        ServiceRatio=service_ratio,
        BasicRatio=basic_ratio,
        SpaVrRatio=spa_vr_ratio,
        FoodMallRatio=food_mall_ratio,
        IsChild=is_child,
        IsSenior=is_senior,
    ).drop(columns=['PassengerId', 'Cabin', 'Name'], errors='ignore').skb.set_name('feature_engineering')

    table_vec = skrub.TableVectorizer(
        cardinality_threshold=40,
        low_cardinality=OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
    )
    features = clean_X.skb.apply(table_vec).skb.set_name('vectorizer')

    cb_base = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, verbose=0, random_seed=42, thread_count=8)
    cb_iter1200 = CatBoostClassifier(iterations=1200, learning_rate=0.018, depth=8, verbose=0, random_seed=42, thread_count=8)
    cb_ensemble = VotingClassifier(
        estimators=[
            ('cb8', CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, verbose=0, random_seed=42, thread_count=8)),
            ('cb7', CatBoostClassifier(iterations=1000, learning_rate=0.025, depth=7, verbose=0, random_seed=2024, thread_count=8)),
        ],
        voting='soft',
    )

    models = {
        'cb_d8_lr002_iter1000': cb_base,
        'cb_d8_lr0018_iter1200': cb_iter1200,
        'cb_ensemble_d7_d8': cb_ensemble,
    }
    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': eval_setup['scoring'], 'row_keys': eval_setup['row_keys']}