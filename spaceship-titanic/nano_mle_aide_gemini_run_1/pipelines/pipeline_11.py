"""Ensemble diversity and domain-aligned feature refinement on top of CatBoost.
Builds directly on the top-performing depth-8 CatBoost seed ensemble by introducing spending proportion ratios, infant indicators, and rule-based CryoSleep consistency.
To reduce variance and decorrelate predictions, tests three distinct ensembling strategies: an expanded five-seed depth-8 ensemble, a multi-depth hybrid combining depth 8 with depth 7, and a regularization diversity ensemble blending standard and higher leaf penalization.
These complementary ensembling variants explore variance reduction and depth complementarity on the refined feature representation.
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

    cb_d8_s42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, verbose=0, random_seed=42, thread_count=8)
    cb_d8_s123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, verbose=0, random_seed=123, thread_count=8)
    cb_d8_s456 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, verbose=0, random_seed=456, thread_count=8)
    cb_d8_s789 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, verbose=0, random_seed=789, thread_count=8)
    cb_d8_s2024 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, verbose=0, random_seed=2024, thread_count=8)

    ensemble_5seeds = VotingClassifier(
        estimators=[
            ('cb_42', cb_d8_s42),
            ('cb_123', cb_d8_s123),
            ('cb_456', cb_d8_s456),
            ('cb_789', cb_d8_s789),
            ('cb_2024', cb_d8_s2024),
        ],
        voting='soft',
    )

    cb_md_d8_s42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, verbose=0, random_seed=42, thread_count=8)
    cb_md_d8_s123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, verbose=0, random_seed=123, thread_count=8)
    cb_md_d8_s456 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, verbose=0, random_seed=456, thread_count=8)
    cb_md_d7_s42 = CatBoostClassifier(iterations=1000, learning_rate=0.025, depth=7, verbose=0, random_seed=42, thread_count=8)
    cb_md_d7_s123 = CatBoostClassifier(iterations=1000, learning_rate=0.025, depth=7, verbose=0, random_seed=123, thread_count=8)

    ensemble_multi_depth = VotingClassifier(
        estimators=[
            ('cb_d8_42', cb_md_d8_s42),
            ('cb_d8_123', cb_md_d8_s123),
            ('cb_d8_456', cb_md_d8_s456),
            ('cb_d7_42', cb_md_d7_s42),
            ('cb_d7_123', cb_md_d7_s123),
        ],
        voting='soft',
    )

    cb_reg_std_s42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0, verbose=0, random_seed=42, thread_count=8)
    cb_reg_std_s123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0, verbose=0, random_seed=123, thread_count=8)
    cb_reg_high_s42 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=5.0, verbose=0, random_seed=42, thread_count=8)
    cb_reg_high_s123 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=5.0, verbose=0, random_seed=123, thread_count=8)

    ensemble_multi_reg = VotingClassifier(
        estimators=[
            ('cb_std_42', cb_reg_std_s42),
            ('cb_std_123', cb_reg_std_s123),
            ('cb_high_42', cb_reg_high_s42),
            ('cb_high_123', cb_reg_high_s123),
        ],
        voting='soft',
    )

    models = {
        'cb_d8_5seed_ensemble': ensemble_5seeds,
        'cb_multi_depth_ensemble': ensemble_multi_depth,
        'cb_multi_reg_ensemble': ensemble_multi_reg,
    }
    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': eval_setup['scoring'], 'row_keys': eval_setup['row_keys']}