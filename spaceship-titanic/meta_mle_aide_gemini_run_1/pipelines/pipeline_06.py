"""CatBoost gradient boosting pipeline with enriched cabin location and group structure features.
In addition to cabin deck and side, the numerical cabin number is extracted to capture spatial positions along the ship decks, and passenger group identifier and group sequence number are parsed from PassengerId.
Categorical features are one-hot encoded using TableVectorizer while continuous luxury amenity expenditures and spatial features are retained.
Multiple CatBoostClassifier configurations with depths 7 and 8 and adjusted iteration counts are compared to evaluate the expanded feature representation.
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
            CabinNum=X['Cabin'].str.split('/').str[1].astype('float64'),
            GroupId=X['PassengerId'].str.slice(0, 4).astype('float64'),
            GroupNum=X['PassengerId'].str.slice(5, 7).astype('float64'),
            TotalSpend=amenity_spend,
            HasSpent=(amenity_spend > 0.0).astype('float64'),
        )
        .drop(columns=['PassengerId', 'Cabin', 'Name'], errors='ignore')
        .skb.set_name('feature_engineering')
    )

    table_vec = skrub.TableVectorizer(
        cardinality_threshold=40,
        low_cardinality=OneHotEncoder(
            sparse_output=False, handle_unknown='ignore'
        ),
    )
    features = clean_X.skb.apply(table_vec).skb.set_name('vectorizer')

    models = {
        'cb_d7_lr003_iter800': CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=7,
            verbose=0,
            random_seed=42,
            thread_count=8,
        ),
        'cb_d7_lr0025_iter1000': CatBoostClassifier(
            iterations=1000,
            learning_rate=0.025,
            depth=7,
            verbose=0,
            random_seed=42,
            thread_count=8,
        ),
        'cb_d8_lr0025_iter800': CatBoostClassifier(
            iterations=800,
            learning_rate=0.025,
            depth=8,
            verbose=0,
            random_seed=42,
            thread_count=8,
        ),
    }
    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': eval_setup['scoring'],
        'row_keys': eval_setup['row_keys'],
    }