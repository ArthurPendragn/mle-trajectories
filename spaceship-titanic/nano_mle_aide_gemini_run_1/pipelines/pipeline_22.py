"""Subspace diversity and depth-weighted voting for structural CatBoost ensemble.
Builds directly on the top-performing 8-model depth-diversity CatBoost ensemble (parent score 0.820890).
While the parent paired depth-8 (256 leaves) and depth-7 (128 leaves) oblivious trees with uniform rsm=0.85 and equal weights, depth-8 trees have strictly higher capacity and benefit from higher voting weight.
Furthermore, diversifying feature subspace sampling rates across models (rsm=0.80 for aggressive partition decorrelation, rsm=0.90 for high-order feature preservation) injects orthogonal tree structure variance across random seeds.
We evaluate three complementary architectures: depth-weighted soft voting on the parent backbone, multi-subspace rate diversification across dual depths, and their synergistic combination."""

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

    cb_d8_r3_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=0.85,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_d8_r3_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=0.85,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_d8_r5_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        rsm=0.85,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_d8_r5_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        rsm=0.85,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )

    cb_d7_r3_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=3.0,
        rsm=0.85,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_d7_r3_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=3.0,
        rsm=0.85,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_d7_r5_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=5.0,
        rsm=0.85,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_d7_r5_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=5.0,
        rsm=0.85,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )

    ensemble_depth_weighted_8m = VotingClassifier(
        estimators=[
            ('d8_r3_42', cb_d8_r3_42),
            ('d8_r3_123', cb_d8_r3_123),
            ('d8_r5_42', cb_d8_r5_42),
            ('d8_r5_123', cb_d8_r5_123),
            ('d7_r3_42', cb_d7_r3_42),
            ('d7_r3_123', cb_d7_r3_123),
            ('d7_r5_42', cb_d7_r5_42),
            ('d7_r5_123', cb_d7_r5_123),
        ],
        voting='soft',
        weights=[1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0],
    )

    cb_d8_r3_s80_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=0.80,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_d8_r3_s90_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        rsm=0.90,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_d8_r5_s80_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        rsm=0.80,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_d8_r5_s90_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        rsm=0.90,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )

    cb_d7_r3_s80_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=3.0,
        rsm=0.80,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_d7_r3_s90_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=3.0,
        rsm=0.90,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_d7_r5_s80_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=5.0,
        rsm=0.80,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_d7_r5_s90_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=5.0,
        rsm=0.90,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )

    subspace_div_estimators = [
        ('d8_r3_s80_42', cb_d8_r3_s80_42),
        ('d8_r3_s90_123', cb_d8_r3_s90_123),
        ('d8_r5_s80_123', cb_d8_r5_s80_123),
        ('d8_r5_s90_42', cb_d8_r5_s90_42),
        ('d7_r3_s80_42', cb_d7_r3_s80_42),
        ('d7_r3_s90_123', cb_d7_r3_s90_123),
        ('d7_r5_s80_123', cb_d7_r5_s80_123),
        ('d7_r5_s90_42', cb_d7_r5_s90_42),
    ]

    ensemble_subspace_div_8m = VotingClassifier(
        estimators=subspace_div_estimators,
        voting='soft',
    )

    ensemble_subspace_weighted_8m = VotingClassifier(
        estimators=subspace_div_estimators,
        voting='soft',
        weights=[1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0],
    )

    models = {
        'cb_depth_weighted_8m': ensemble_depth_weighted_8m,
        'cb_subspace_div_8m': ensemble_subspace_div_8m,
        'cb_subspace_weighted_8m': ensemble_subspace_weighted_8m,
    }

    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': eval_setup['scoring'],
        'row_keys': eval_setup['row_keys'],
    }