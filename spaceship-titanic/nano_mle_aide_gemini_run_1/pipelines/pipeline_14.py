"""Differential expenditure contrast and schedule refinement on multi-regularization CatBoost ensemble.
Builds directly on the top-performing multi-regularization CatBoost ensemble (parent score 0.819855).
Amenities strongly diverge in their transport association: FoodCourt and ShoppingMall spending indicate higher survival, whereas RoomService, Spa, and VRDeck spending correlate heavily with non-transport.
We engineer SpendDiff (basic_spend - service_spend) to provide oblivious trees with an explicit linear contrast across spending classes that cannot be captured by orthogonal axis-aligned splits.
To evaluate optimal ensemble synergy on this enriched representation, we compare the standard 4-model multi-regularization ensemble against an extended schedule (1200 iterations at learning rate 0.018) and a regularization-weighted soft vote prioritizing lower-variance penalized trees.
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
    spend_diff = basic_spend - service_spend
    service_ratio = service_spend / (amenity_spend + 1.0)
    basic_ratio = basic_spend / (amenity_spend + 1.0)
    spend_count = (
        (rs > 0.0).astype('float64')
        + (fc > 0.0).astype('float64')
        + (sm > 0.0).astype('float64')
        + (spa > 0.0).astype('float64')
        + (vr > 0.0).astype('float64')
    )
    has_spent = (amenity_spend > 0.0).astype('float64')

    is_child = (X['Age'] < 13.0).astype('float64')
    is_baby = (X['Age'] <= 2.0).astype('float64')

    cryo_str = X['CryoSleep'].astype('string')
    cryo_fill = (amenity_spend > 0.0).map({True: 'False'})
    cryo_imputed = cryo_str.fillna(cryo_fill)

    clean_X = (
        X.assign(
            CryoSleep=cryo_imputed,
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
            SpendDiff=spend_diff,
            ServiceRatio=service_ratio,
            BasicRatio=basic_ratio,
            SpendCount=spend_count,
            HasSpent=has_spent,
            IsChild=is_child,
            IsBaby=is_baby,
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

    cb_std_s42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_std_s123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_high_s42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_high_s123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )

    ensemble_standard = VotingClassifier(
        estimators=[
            ('cb_std_42', cb_std_s42),
            ('cb_std_123', cb_std_s123),
            ('cb_high_42', cb_high_s42),
            ('cb_high_123', cb_high_s123),
        ],
        voting='soft',
    )

    cb_1200_std_s42 = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.018,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_1200_std_s123 = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.018,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_1200_high_s42 = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.018,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_1200_high_s123 = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.018,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )

    ensemble_iter1200 = VotingClassifier(
        estimators=[
            ('cb_1200_std_42', cb_1200_std_s42),
            ('cb_1200_std_123', cb_1200_std_s123),
            ('cb_1200_high_42', cb_1200_high_s42),
            ('cb_1200_high_123', cb_1200_high_s123),
        ],
        voting='soft',
    )

    ensemble_weighted = VotingClassifier(
        estimators=[
            ('cb_std_42', cb_std_s42),
            ('cb_std_123', cb_std_s123),
            ('cb_high_42', cb_high_s42),
            ('cb_high_123', cb_high_s123),
        ],
        voting='soft',
        weights=[1.0, 1.0, 1.5, 1.5],
    )

    models = {
        'cb_multi_reg_spend_diff': ensemble_standard,
        'cb_multi_reg_iter1200': ensemble_iter1200,
        'cb_multi_reg_weighted': ensemble_weighted,
    }
    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': eval_setup['scoring'],
        'row_keys': eval_setup['row_keys'],
    }