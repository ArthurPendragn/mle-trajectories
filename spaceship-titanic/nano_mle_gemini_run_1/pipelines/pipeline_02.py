from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import skrub
from xgboost import XGBClassifier


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    amenity_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    spend_sum = X[amenity_cols].fillna(0.0).sum(axis=1)
    zero_spend = (spend_sum == 0.0).astype(float)

    deck = X['Cabin'].str.slice(0, 1).fillna('Missing')
    side = X['Cabin'].str.slice(-1).fillna('Missing')
    group_num = X['PassengerId'].str.slice(5).astype(float)

    home_planet = X['HomePlanet'].fillna('Missing').astype(str)
    cryo_sleep = X['CryoSleep'].fillna('Missing').astype(str)
    destination = X['Destination'].fillna('Missing').astype(str)
    vip = X['VIP'].fillna('Missing').astype(str)

    X_engineered = X.assign(
        HomePlanet=home_planet,
        CryoSleep=cryo_sleep,
        Destination=destination,
        VIP=vip,
        Deck=deck,
        Side=side,
        GroupNum=group_num,
        TotalSpend=spend_sum,
        ZeroSpend=zero_spend,
    )
    features = X_engineered.drop(
        columns=['PassengerId', 'Cabin', 'Name']
    ).skb.set_name('engineered_features')

    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    num_cols = [
        'Age',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck',
        'GroupNum',
        'TotalSpend',
        'ZeroSpend',
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'cat',
                OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                cat_cols,
            ),
            ('num', 'passthrough', num_cols),
        ]
    )

    pipelines = {
        'catboost_depth4': Pipeline([
            ('prep', preprocessor),
            (
                'model',
                CatBoostClassifier(
                    depth=4,
                    learning_rate=0.05,
                    iterations=300,
                    random_seed=42,
                    thread_count=8,
                    verbose=0,
                ),
            ),
        ]),
        'catboost_depth6': Pipeline([
            ('prep', preprocessor),
            (
                'model',
                CatBoostClassifier(
                    depth=6,
                    learning_rate=0.05,
                    iterations=300,
                    random_seed=42,
                    thread_count=8,
                    verbose=0,
                ),
            ),
        ]),
        'xgboost_depth4': Pipeline([
            ('prep', preprocessor),
            (
                'model',
                XGBClassifier(
                    max_depth=4,
                    learning_rate=0.05,
                    n_estimators=300,
                    random_state=42,
                    n_jobs=8,
                    eval_metric='logloss',
                ),
            ),
        ]),
        'xgboost_depth6': Pipeline([
            ('prep', preprocessor),
            (
                'model',
                XGBClassifier(
                    max_depth=6,
                    learning_rate=0.05,
                    n_estimators=300,
                    random_state=42,
                    n_jobs=8,
                    eval_metric='logloss',
                ),
            ),
        ]),
    }

    chosen_pipeline = skrub.choose_from(pipelines, name='model_variant')
    pred = features.skb.apply(chosen_pipeline, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}