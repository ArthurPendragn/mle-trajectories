"""Ordinal deck level representation and split/sample diversity for multi-regularization CatBoost ensemble.
Builds directly on the top-performing multi-regularization CatBoost ensemble (parent score 0.819855).
In oblivious trees of depth 8, orthogonal one-hot encoding of cabin decks requires multiple partition levels to isolate deck tiers.
We introduce DeckLevel, an ordinal mapping of spaceship decks (T/A/B/C/D/E/F/G), allowing a single tree level to partition upper luxury decks from lower quarters.
On this enriched representation, we evaluate three ensembling strategies: the verified dual-tier regularization baseline, split-scoring diversity via random_strength perturbation, and sample-weighting diversity via bagging_temperature adjustments.
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
    deck_level = deck.map({
        'T': 0.0,
        'A': 1.0,
        'B': 2.0,
        'C': 3.0,
        'D': 4.0,
        'E': 5.0,
        'F': 6.0,
        'G': 7.0,
    })

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
            DeckLevel=deck_level,
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

    cb_reg_std_s42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_reg_std_s123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    cb_reg_high_s42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_reg_high_s123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    ensemble_multi_reg = VotingClassifier(
        estimators=[
            ('cb_std_42', cb_reg_std_s42),
            ('cb_std_123', cb_reg_std_s123),
            ('cb_high_42', cb_reg_high_s42),
            ('cb_high_123', cb_reg_high_s123),
        ],
        voting='soft',
    )

    cb_rs_high_s42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        random_strength=2.0,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_rs_high_s123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        random_strength=2.0,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    ensemble_random_strength = VotingClassifier(
        estimators=[
            ('cb_std_42', cb_reg_std_s42),
            ('cb_std_123', cb_reg_std_s123),
            ('cb_rs_42', cb_rs_high_s42),
            ('cb_rs_123', cb_rs_high_s123),
        ],
        voting='soft',
    )

    cb_bt_high_s42 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        bagging_temperature=0.7,
        verbose=0,
        random_seed=42,
        thread_count=8,
    )
    cb_bt_high_s123 = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        bagging_temperature=0.7,
        verbose=0,
        random_seed=123,
        thread_count=8,
    )
    ensemble_bagging_temp = VotingClassifier(
        estimators=[
            ('cb_std_42', cb_reg_std_s42),
            ('cb_std_123', cb_reg_std_s123),
            ('cb_bt_42', cb_bt_high_s42),
            ('cb_bt_123', cb_bt_high_s123),
        ],
        voting='soft',
    )

    models = {
        'cb_multi_reg': ensemble_multi_reg,
        'cb_reg_random_strength': ensemble_random_strength,
        'cb_reg_bagging_temp': ensemble_bagging_temp,
    }
    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': eval_setup['scoring'],
        'row_keys': eval_setup['row_keys'],
    }