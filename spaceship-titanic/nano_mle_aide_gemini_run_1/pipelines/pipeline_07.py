"""CatBoost gradient boosting pipeline with expanded passenger domain features and amenity spending breakdowns.
Builds upon the high-performing CatBoost model by engineering deck-side interaction, cabin deck regions, child indicators, and granular amenity spending counts and service totals.
Evaluates CatBoost classifiers at depth 8 with adjusted regularization and iteration schedules to capture multi-way spatial and behavioral patterns without overfitting.
"""

from catboost import CatBoostClassifier
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
    is_child = (X['Age'] < 13.0).astype('float64')

    clean_X = (
        X.assign(
            Deck=deck,
            Side=side,
            DeckSide=deck_side,
            CabinNum=cabin_num,
            CabinRegion=cabin_region,
            GroupId=X['PassengerId'].str.slice(0, 4).astype('float64'),
            GroupNum=X['PassengerId'].str.slice(5, 7).astype('float64'),
            TotalSpend=amenity_spend,
            ServiceSpend=service_spend,
            BasicSpend=basic_spend,
            SpendCount=spend_count,
            HasSpent=has_spent,
            IsChild=is_child,
        )
        .drop(columns=['PassengerId', 'Cabin', 'Name'], errors='ignore')
        .skb.set_name('feature_engineering')
    )

    table_vec = skrub.TableVectorizer(
        cardinality_threshold=40,
        low_cardinality=OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
    )
    features = clean_X.skb.apply(table_vec).skb.set_name('vectorizer')

    models = {
        'cb_d8_lr0025_iter800': CatBoostClassifier(
            iterations=800,
            learning_rate=0.025,
            depth=8,
            verbose=0,
            random_seed=42,
            thread_count=8,
        ),
        'cb_d8_lr002_iter1000': CatBoostClassifier(
            iterations=1000,
            learning_rate=0.02,
            depth=8,
            verbose=0,
            random_seed=42,
            thread_count=8,
        ),
        'cb_d8_lr0025_l2reg5_iter800': CatBoostClassifier(
            iterations=800,
            learning_rate=0.025,
            depth=8,
            l2_leaf_reg=5.0,
            verbose=0,
            random_seed=42,
            thread_count=8,
        ),
    }
    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': eval_setup['scoring'], 'row_keys': eval_setup['row_keys']}