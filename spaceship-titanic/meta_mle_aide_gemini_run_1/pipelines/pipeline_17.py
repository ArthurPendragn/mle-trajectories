"""Feature subspace randomization and regularization diversity for depth-8 CatBoost ensemble.
Builds directly on the top-performing multi-regularization CatBoost ensemble (parent score 0.819855).
At depth 8, oblivious trees partition into 256 leaves and can repeatedly split on the same dominant features at upper levels across seeds.
Introducing random subspace feature sampling (rsm) decorrelates individual trees by forcing split discovery across diverse spatial and amenity signals.
We evaluate three complementary ensembling configurations combining dual L2 leaf regularization (l2=3.0 and l2=5.0) with subspace randomization: a hybrid full-and-subspace ensemble, an rsm=0.85 ensemble, and an rsm=0.80 ensemble.
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

    cb_reg3_full_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=1.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_reg3_rsm85_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=0.85,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_reg5_full_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        rsm=1.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_reg5_rsm85_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        rsm=0.85,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    ensemble_hybrid_rsm = VotingClassifier(
        estimators=[
            ('cb_r3_f42', cb_reg3_full_42),
            ('cb_r3_r123', cb_reg3_rsm85_123),
            ('cb_r5_f42', cb_reg5_full_42),
            ('cb_r5_r123', cb_reg5_rsm85_123),
        ],
        voting='soft',
    )

    cb_rsm85_reg3_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=0.85,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_rsm85_reg3_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=0.85,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_rsm85_reg5_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        rsm=0.85,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_rsm85_reg5_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        rsm=0.85,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    ensemble_pure_rsm85 = VotingClassifier(
        estimators=[
            ('cb_r3_42', cb_rsm85_reg3_42),
            ('cb_r3_123', cb_rsm85_reg3_123),
            ('cb_r5_42', cb_rsm85_reg5_42),
            ('cb_r5_123', cb_rsm85_reg5_123),
        ],
        voting='soft',
    )

    cb_rsm80_reg3_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=0.80,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_rsm80_reg3_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=0.80,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_rsm80_reg5_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        rsm=0.80,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_rsm80_reg5_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        rsm=0.80,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    ensemble_pure_rsm80 = VotingClassifier(
        estimators=[
            ('cb_r3_42', cb_rsm80_reg3_42),
            ('cb_r3_123', cb_rsm80_reg3_123),
            ('cb_r5_42', cb_rsm80_reg5_42),
            ('cb_r5_123', cb_rsm80_reg5_123),
        ],
        voting='soft',
    )

    models = {
        'cb_hybrid_rsm_reg': ensemble_hybrid_rsm,
        'cb_multi_reg_rsm85': ensemble_pure_rsm85,
        'cb_multi_reg_rsm80': ensemble_pure_rsm80,
    }
    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': eval_setup['scoring'],
        'row_keys': eval_setup['row_keys'],
    }