from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

    cat_transformer = OneHotEncoder(
        sparse_output=False, handle_unknown='ignore'
    )
    num_transformer = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_cols),
            ('num', num_transformer, num_cols),
        ]
    )

    pipelines = {
        'xgb_lgb_ensemble': Pipeline(
            [
                ('prep', preprocessor),
                (
                    'model',
                    VotingClassifier(
                        estimators=[
                            (
                                'xgb',
                                XGBClassifier(
                                    max_depth=4,
                                    learning_rate=0.05,
                                    n_estimators=300,
                                    random_state=42,
                                    n_jobs=4,
                                    eval_metric='logloss',
                                ),
                            ),
                            (
                                'lgbm',
                                LGBMClassifier(
                                    learning_rate=0.05,
                                    num_leaves=15,
                                    n_estimators=300,
                                    random_state=42,
                                    n_jobs=4,
                                    verbose=-1,
                                ),
                            ),
                        ],
                        voting='soft',
                    ),
                ),
            ]
        ),
        'xgb_lgb_mlp_ensemble': Pipeline(
            [
                ('prep', preprocessor),
                (
                    'model',
                    VotingClassifier(
                        estimators=[
                            (
                                'xgb',
                                XGBClassifier(
                                    max_depth=4,
                                    learning_rate=0.05,
                                    n_estimators=300,
                                    random_state=42,
                                    n_jobs=4,
                                    eval_metric='logloss',
                                ),
                            ),
                            (
                                'lgbm',
                                LGBMClassifier(
                                    learning_rate=0.05,
                                    num_leaves=15,
                                    n_estimators=300,
                                    random_state=42,
                                    n_jobs=4,
                                    verbose=-1,
                                ),
                            ),
                            (
                                'mlp',
                                MLPClassifier(
                                    hidden_layer_sizes=(64, 32),
                                    max_iter=300,
                                    early_stopping=True,
                                    random_state=42,
                                ),
                            ),
                        ],
                        voting='soft',
                    ),
                ),
            ]
        ),
        'xgb_lgb_lr_ensemble': Pipeline(
            [
                ('prep', preprocessor),
                (
                    'model',
                    VotingClassifier(
                        estimators=[
                            (
                                'xgb',
                                XGBClassifier(
                                    max_depth=4,
                                    learning_rate=0.05,
                                    n_estimators=300,
                                    random_state=42,
                                    n_jobs=4,
                                    eval_metric='logloss',
                                ),
                            ),
                            (
                                'lgbm',
                                LGBMClassifier(
                                    learning_rate=0.05,
                                    num_leaves=15,
                                    n_estimators=300,
                                    random_state=42,
                                    n_jobs=4,
                                    verbose=-1,
                                ),
                            ),
                            (
                                'lr',
                                LogisticRegression(
                                    max_iter=500, random_state=42
                                ),
                            ),
                        ],
                        voting='soft',
                    ),
                ),
            ]
        ),
        'standalone_mlp': Pipeline(
            [
                ('prep', preprocessor),
                (
                    'model',
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=300,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    chosen_pipeline = skrub.choose_from(pipelines, name='model_variant')
    pred = features.skb.apply(chosen_pipeline, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}