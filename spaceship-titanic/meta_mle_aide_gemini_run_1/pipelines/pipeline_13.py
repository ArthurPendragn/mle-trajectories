"""Regularization spectrum refinement and seed scaling for depth-8 CatBoost ensemble.
Builds directly on the top-performing multi-regularization CatBoost ensemble on the verified feature representation.
At depth 8, oblivious trees partition into 256 leaves where leaf regularization and permutation randomness strongly influence generalization.
To systematically identify the optimal trade-off between regularization diversity and seed variance reduction, this experiment evaluates three ensemble architectures:
a tri-tier regularization ensemble spanning l2=3.0, 4.0, and 5.0 across two seeds, an expanded 6-seed ensemble at l2=3.0 and 5.0, and a comprehensive 8-seed ensemble across four distinct seeds.
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
        low_cardinality=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    )
    features = clean_X.skb.apply(table_vec).skb.set_name('vectorizer')

    cb_l2_3_s42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0, verbose=0, random_seed=42, thread_count=8)
    cb_l2_3_s123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0, verbose=0, random_seed=123, thread_count=8)
    cb_l2_3_s456 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0, verbose=0, random_seed=456, thread_count=8)
    cb_l2_3_s789 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0, verbose=0, random_seed=789, thread_count=8)

    cb_l2_4_s42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=4.0, verbose=0, random_seed=42, thread_count=8)
    cb_l2_4_s123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=4.0, verbose=0, random_seed=123, thread_count=8)

    cb_l2_5_s42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=5.0, verbose=0, random_seed=42, thread_count=8)
    cb_l2_5_s123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=5.0, verbose=0, random_seed=123, thread_count=8)
    cb_l2_5_s456 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=5.0, verbose=0, random_seed=456, thread_count=8)
    cb_l2_5_s789 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=5.0, verbose=0, random_seed=789, thread_count=8)

    ensemble_tri_reg = VotingClassifier(
        estimators=[
            ('cb_l2_3_42', cb_l2_3_s42), ('cb_l2_3_123', cb_l2_3_s123),
            ('cb_l2_4_42', cb_l2_4_s42), ('cb_l2_4_123', cb_l2_4_s123),
            ('cb_l2_5_42', cb_l2_5_s42), ('cb_l2_5_123', cb_l2_5_s123)
        ],
        voting='soft'
    )

    ensemble_6seed = VotingClassifier(
        estimators=[
            ('cb_l2_3_42', cb_l2_3_s42), ('cb_l2_3_123', cb_l2_3_s123), ('cb_l2_3_456', cb_l2_3_s456),
            ('cb_l2_5_42', cb_l2_5_s42), ('cb_l2_5_123', cb_l2_5_s123), ('cb_l2_5_456', cb_l2_5_s456)
        ],
        voting='soft'
    )

    ensemble_8seed = VotingClassifier(
        estimators=[
            ('cb_l2_3_42', cb_l2_3_s42), ('cb_l2_3_123', cb_l2_3_s123), ('cb_l2_3_456', cb_l2_3_s456), ('cb_l2_3_789', cb_l2_3_s789),
            ('cb_l2_5_42', cb_l2_5_s42), ('cb_l2_5_123', cb_l2_5_s123), ('cb_l2_5_456', cb_l2_5_s456), ('cb_l2_5_789', cb_l2_5_s789)
        ],
        voting='soft'
    )

    models = {
        'cb_multi_reg_tri': ensemble_tri_reg,
        'cb_multi_reg_6seed': ensemble_6seed,
        'cb_multi_reg_8seed': ensemble_8seed
    }
    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': eval_setup['scoring'], 'row_keys': eval_setup['row_keys']}