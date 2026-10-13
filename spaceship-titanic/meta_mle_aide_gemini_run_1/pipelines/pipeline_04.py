"""CatBoost gradient boosting pipeline for Spaceship Titanic passenger transport prediction.
Features are engineered by extracting cabin deck and side from the Cabin identifier, computing total luxury amenity expenditure across all five facilities, and adding a zero-spending binary indicator.
High-cardinality identifiers (PassengerId, Name, Cabin) are removed, and low-cardinality categoricals are one-hot encoded via TableVectorizer.
Multiple CatBoostClassifier configurations varying tree depth, learning rate, and iteration count are evaluated to test an oblivious tree gradient boosting family distinct from previous drafts.
"""

from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder
import skrub


def build():
    eval_setup = build_evaluation()
    X = eval_setup['X']
    y = eval_setup['y']

    amenity_spend = (
        X['RoomService'].fillna(0.0)
        + X['FoodCourt'].fillna(0.0)
        + X['ShoppingMall'].fillna(0.0)
        + X['Spa'].fillna(0.0)
        + X['VRDeck'].fillna(0.0)
    )

    clean_X = (
        X.assign(
            Deck=X['Cabin'].str.slice(0, 1),
            Side=X['Cabin'].str.slice(-1),
            TotalSpend=amenity_spend,
            HasSpent=(amenity_spend > 0.0).astype('float64'),
        )
        .drop(columns=['PassengerId', 'Cabin', 'Name'], errors='ignore')
        .skb.set_name('feature_engineering')
    )

    table_vec = skrub.TableVectorizer(
        cardinality_threshold=40,
        low_cardinality=OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
    )
    features = clean_X.skb.apply(table_vec).skb.set_name('vectorizer')

    model = skrub.choose_from(
        {
            'cb_depth6_lr004': CatBoostClassifier(
                iterations=600,
                learning_rate=0.04,
                depth=6,
                verbose=0,
                random_seed=42,
                thread_count=8,
            ),
            'cb_depth7_lr003': CatBoostClassifier(
                iterations=700,
                learning_rate=0.03,
                depth=7,
                verbose=0,
                random_seed=42,
                thread_count=8,
            ),
            'cb_depth5_lr006': CatBoostClassifier(
                iterations=500,
                learning_rate=0.06,
                depth=5,
                verbose=0,
                random_seed=42,
                thread_count=8,
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