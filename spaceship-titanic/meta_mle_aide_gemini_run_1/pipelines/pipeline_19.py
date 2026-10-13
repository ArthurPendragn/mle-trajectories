"""Structural depth diversity and leaf regularization synergy for subspace-randomized CatBoost ensemble.
Builds directly on the top-performing multi-regularization CatBoost ensemble with feature subspace randomization (parent score 0.819855).
While depth-8 oblivious trees partition into 256 leaves and capture high-order feature interactions, shallower depth-7 trees (128 leaves) offer lower variance and smoother decision boundaries.
Combining depth-8 and depth-7 models under feature subspace randomization (rsm=0.85) pairs multi-way spatial/amenity interactions with structural regularization.
We evaluate three complementary depth-and-regularization ensembling architectures:
1. A balanced 4-model ensemble pairing depth 8 and depth 7 across dual regularization (l2=3.0, 5.0) and seeds.
2. A 6-model ensemble anchoring the full depth-8 backbone with depth-7 intermediate regularization smoothing.
3. An 8-model full-factorial ensemble blending complete depth-8 and depth-7 multi-regularization suites.
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

    clean_X = X.assign(
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
        ServiceRatio=service_ratio,
        BasicRatio=basic_ratio,
        SpendCount=spend_count,
        HasSpent=has_spent,
        IsChild=is_child,
        IsBaby=is_baby,
    ).drop(columns=['PassengerId', 'Cabin', 'Name'], errors='ignore').skb.set_name('feature_engineering')

    table_vec = skrub.TableVectorizer(
        cardinality_threshold=40,
        low_cardinality=OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
    )
    features = clean_X.skb.apply(table_vec).skb.set_name('vectorizer')

    # Depth 8 base learners with rsm=0.85
    cb_d8_r3_42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0, rsm=0.85, verbose=0, random_seed=42, thread_count=8)
    cb_d8_r3_123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0, rsm=0.85, verbose=0, random_seed=123, thread_count=8)
    cb_d8_r5_42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=5.0, rsm=0.85, verbose=0, random_seed=42, thread_count=8)
    cb_d8_r5_123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=5.0, rsm=0.85, verbose=0, random_seed=123, thread_count=8)

    # Depth 7 base learners with rsm=0.85
    cb_d7_r3_42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=7, l2_leaf_reg=3.0, rsm=0.85, verbose=0, random_seed=42, thread_count=8)
    cb_d7_r3_123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=7, l2_leaf_reg=3.0, rsm=0.85, verbose=0, random_seed=123, thread_count=8)
    cb_d7_r4_42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=7, l2_leaf_reg=4.0, rsm=0.85, verbose=0, random_seed=42, thread_count=8)
    cb_d7_r4_123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=7, l2_leaf_reg=4.0, rsm=0.85, verbose=0, random_seed=123, thread_count=8)
    cb_d7_r5_42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=7, l2_leaf_reg=5.0, rsm=0.85, verbose=0, random_seed=42, thread_count=8)
    cb_d7_r5_123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=7, l2_leaf_reg=5.0, rsm=0.85, verbose=0, random_seed=123, thread_count=8)

    # Variant 1: Balanced 4-model depth-regularization hybrid
    ensemble_4m = VotingClassifier(
        estimators=[
            ('cb_d8_r3_42', cb_d8_r3_42),
            ('cb_d8_r5_123', cb_d8_r5_123),
            ('cb_d7_r3_42', cb_d7_r3_42),
            ('cb_d7_r5_123', cb_d7_r5_123),
        ],
        voting='soft',
    )

    # Variant 2: 6-model depth-regularization ensemble (depth-8 backbone + depth-7 intermediate smoothing)
    ensemble_6m = VotingClassifier(
        estimators=[
            ('cb_d8_r3_42', cb_d8_r3_42),
            ('cb_d8_r3_123', cb_d8_r3_123),
            ('cb_d8_r5_42', cb_d8_r5_42),
            ('cb_d8_r5_123', cb_d8_r5_123),
            ('cb_d7_r4_42', cb_d7_r4_42),
            ('cb_d7_r4_123', cb_d7_r4_123),
        ],
        voting='soft',
    )

    # Variant 3: 8-model full factorial depth-regularization ensemble
    ensemble_8m = VotingClassifier(
        estimators=[
            ('cb_d8_r3_42', cb_d8_r3_42),
            ('cb_d8_r3_123', cb_d8_r3_123),
            ('cb_d8_r5_42', cb_d8_r5_42),
            ('cb_d8_r5_123', cb_d8_r5_123),
            ('cb_d7_r3_42', cb_d7_r3_42),
            ('cb_d7_r3_123', cb_d7_r3_123),
            ('cb_d7_r5_42', cb_d7_r5_42),
            ('cb_d7_r5_123', cb_d7_r5_123),
        ],
        voting='soft',
    )

    models = {
        'cb_depth_reg_rsm85_4m': ensemble_4m,
        'cb_depth_reg_rsm85_6m': ensemble_6m,
        'cb_depth_reg_rsm85_8m': ensemble_8m,
    }
    model = skrub.choose_from(models, name='model_variant')

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': eval_setup['scoring'], 'row_keys': eval_setup['row_keys']}