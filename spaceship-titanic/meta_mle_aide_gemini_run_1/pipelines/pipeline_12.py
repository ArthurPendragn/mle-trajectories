"""Ensemble regularization scaling and physical domain imputation for CatBoost.
Builds directly on the top-performing multi-regularization ensemble by incorporating exact physical domain consistency rules: imputing HomePlanet from exclusive cabin decks (A/B/C/T for Europa, G for Earth) and enforcing VIP non-eligibility for Earth passengers and minors.
To build on the regularization diversity breakthrough, compares three 6-model soft-voting ensemble architectures:
1. An expanded two-tier regularization ensemble (l2=3.0 and l2=5.0) across three distinct seeds to minimize random permutation variance.
2. A tri-tier regularization ensemble spanning l2=3.0, 5.0, and 8.0 across two seeds to provide progressive leaf-penalty smoothing.
3. A regularization-depth hybrid blending depth-8 models at l2=3.0 and l2=5.0 with shallower depth-7 models to balance multi-way interactions and tree structure diversity.
"""

from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder
import skrub


def build():
    eval_setup = build_evaluation()
    X = eval_setup['X']
    y = eval_setup['y']

    # Cabin components
    deck = X['Cabin'].str.slice(0, 1)
    side = X['Cabin'].str.slice(-1)
    deck_side = deck + '_' + side
    cabin_num = X['Cabin'].str.split('/').str[1].astype('float64')
    cabin_region = (cabin_num // 300).astype('float64')

    # Domain consistency: HomePlanet derived from exclusive deck assignments
    hp_deck_map = deck.map({
        'A': 'Europa',
        'B': 'Europa',
        'C': 'Europa',
        'T': 'Europa',
        'G': 'Earth',
    })
    home_planet = X['HomePlanet'].astype('string').fillna(hp_deck_map)

    # Spending components
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

    # Age indicators
    is_child = (X['Age'] < 13.0).astype('float64')
    is_baby = (X['Age'] <= 2.0).astype('float64')

    # Domain consistency: CryoSleep cannot be True if passenger spent amenity money
    cryo_str = X['CryoSleep'].astype('string')
    cryo_fill = (amenity_spend > 0.0).fillna(False).map({True: 'False'})
    cryo_imputed = cryo_str.fillna(cryo_fill)

    # Domain consistency: VIP cannot be True for Earth passengers or minors (<18)
    vip_str = X['VIP'].astype('string')
    is_non_vip = (home_planet == 'Earth').fillna(False) | (
        X['Age'] < 18.0
    ).fillna(False)
    vip_fill = is_non_vip.map({True: 'False'})
    vip_imputed = vip_str.fillna(vip_fill)

    clean_X = (
        X.assign(
            HomePlanet=home_planet,
            CryoSleep=cryo_imputed,
            VIP=vip_imputed,
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

    # Variant 1: Expanded two-tier regularization ensemble across 3 seeds (6 models)
    cb_std_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_std_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_std_456 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=456,
        thread_count=8,
    )
    cb_high_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_high_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_high_456 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=456,
        thread_count=8,
    )
    ensemble_reg_6seeds = VotingClassifier(
        estimators=[
            ('std_42', cb_std_42),
            ('std_123', cb_std_123),
            ('std_456', cb_std_456),
            ('high_42', cb_high_42),
            ('high_123', cb_high_123),
            ('high_456', cb_high_456),
        ],
        voting='soft',
    )

    # Variant 2: Tri-tier regularization ensemble (l2=3.0, 5.0, 8.0) across 2 seeds (6 models)
    cb_tri_l2_3_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_tri_l2_3_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_tri_l2_5_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_tri_l2_5_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_tri_l2_8_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=8.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_tri_l2_8_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=8.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    ensemble_tri_reg = VotingClassifier(
        estimators=[
            ('l2_3_42', cb_tri_l2_3_42),
            ('l2_3_123', cb_tri_l2_3_123),
            ('l2_5_42', cb_tri_l2_5_42),
            ('l2_5_123', cb_tri_l2_5_123),
            ('l2_8_42', cb_tri_l2_8_42),
            ('l2_8_123', cb_tri_l2_8_123),
        ],
        voting='soft',
    )

    # Variant 3: Regularization-depth hybrid (depth 8 with l2=3/5 + depth 7 with l2=4) (6 models)
    cb_hyb_d8_l2_3_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_hyb_d8_l2_3_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_hyb_d8_l2_5_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_hyb_d8_l2_5_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_hyb_d7_l2_4_42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.025,
        depth=7,
        l2_leaf_reg=4.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_hyb_d7_l2_4_123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.025,
        depth=7,
        l2_leaf_reg=4.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    ensemble_reg_depth_hybrid = VotingClassifier(
        estimators=[
            ('d8_l2_3_42', cb_hyb_d8_l2_3_42),
            ('d8_l2_3_123', cb_hyb_d8_l2_3_123),
            ('d8_l2_5_42', cb_hyb_d8_l2_5_42),
            ('d8_l2_5_123', cb_hyb_d8_l2_5_123),
            ('d7_l2_4_42', cb_hyb_d7_l2_4_42),
            ('d7_l2_4_123', cb_hyb_d7_l2_4_123),
        ],
        voting='soft',
    )

    models = {
        'cb_reg_blend_6seeds': ensemble_reg_6seeds,
        'cb_tri_reg_ensemble': ensemble_tri_reg,
        'cb_reg_depth_hybrid': ensemble_reg_depth_hybrid,
    }
    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': eval_setup['scoring'],
        'row_keys': eval_setup['row_keys'],
    }