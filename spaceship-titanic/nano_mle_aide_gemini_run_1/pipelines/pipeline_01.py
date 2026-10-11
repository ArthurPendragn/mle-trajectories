"""Baseline gradient boosting pipeline for Spaceship Titanic passenger transport prediction.
Features are engineered by extracting deck and side from the Cabin column and computing total luxury amenity spending.
High-cardinality identifier columns (PassengerId, Name, Cabin) are removed, and categoricals are one-hot encoded via TableVectorizer.
Two HistGradientBoostingClassifier configurations are evaluated to establish a competitive tabular baseline."""

import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def build():
    eval_setup = build_evaluation()
    X = eval_setup['X']
    y = eval_setup['y']

    total_spend = (
        X['RoomService'].fillna(0.0)
        + X['FoodCourt'].fillna(0.0)
        + X['ShoppingMall'].fillna(0.0)
        + X['Spa'].fillna(0.0)
        + X['VRDeck'].fillna(0.0)
    )
    deck = X['Cabin'].str[0]
    side = X['Cabin'].str[-1]

    features = (
        X.assign(TotalSpend=total_spend, Deck=deck, Side=side)
        .drop(columns=['PassengerId', 'Name', 'Cabin'])
        .skb.set_name('features')
    )

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    model = skrub.choose_from(
        {
            'hgb_default': make_pipeline(
                skrub.TableVectorizer(low_cardinality=ohe),
                HistGradientBoostingClassifier(random_state=42),
            ),
            'hgb_lr005_maxiter200': make_pipeline(
                skrub.TableVectorizer(low_cardinality=ohe),
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    min_samples_leaf=30,
                    random_state=42,
                ),
            ),
        },
        name='model_variant',
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': eval_setup['scoring'],
        'row_keys': eval_setup['row_keys'],
    }