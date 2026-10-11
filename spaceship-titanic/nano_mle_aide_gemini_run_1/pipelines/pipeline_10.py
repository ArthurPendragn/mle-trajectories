"""CatBoost multi-seed ensemble and schedule refinement on top passenger domain features.
Builds directly on the top-performing depth-8 CatBoost model by testing schedule and regularization adjustments alongside a multi-seed soft-voting ensemble.
Averaging predictions across differently seeded CatBoost models smooths out variance from oblivious tree permutations at depth 8.
Compares a seed ensemble against extended iteration schedules and relaxed leaf regularization to improve generalization.
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

    cb_base = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_s123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_s456 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        verbose=0,
        random_seed=456,
        thread_count=8,
    )
    seed_ensemble = VotingClassifier(
        estimators=[('cb_42', cb_base), ('cb_123', cb_s123), ('cb_456', cb_s456)],
        voting='soft',
    )

    models = {
        'cb_d8_lr0018_iter1200': CatBoostClassifier(
            iterations=1200,
            learning_rate=0.018,
            depth=8,
            verbose=0,
            random_seed=42,
            thread_count=8,
        ),
        'cb_d8_l2reg2_iter1000': CatBoostClassifier(
            iterations=1000,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=2.0,
            verbose=0,
            random_seed=42,
            thread_count=8,
        ),
        'cb_d8_seed_ensemble': seed_ensemble,
    }

    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': eval_setup['scoring'],
        'row_keys': eval_setup['row_keys'],
    }