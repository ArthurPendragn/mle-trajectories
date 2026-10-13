"""Schedule refinement and shrinkage tuning for subspace-diversified CatBoost ensemble.
Builds directly on the top-performing 8-model subspace-diversified CatBoost ensemble (parent score 0.821465).
The parent established that pairing depth-8 (256 leaves) and depth-7 (128 leaves) oblivious trees under dual feature subspace sampling rates (rsm=0.80 and 0.90) with depth-weighted soft voting yields the highest accuracy.
At depth 8 and 7, oblivious trees construct complex multi-way partitions where smaller step sizes with extended iteration schedules can reduce gradient variance and refine split discovery.
We evaluate three schedule and shrinkage configurations on the winning 8-model subspace architecture:
the 1000-iteration baseline at lr=0.02, an extended 1100-iteration schedule at lr=0.018, and a calibrated 1200-iteration schedule at lr=0.017.
"""
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder
import skrub


def make_subspace_ensemble(iterations, lr):
    estimators = [
        ('d8_r3_s80_42', CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=8, l2_leaf_reg=3.0, rsm=0.80, verbose=0, random_seed=42, thread_count=8)),
        ('d8_r3_s90_123', CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=8, l2_leaf_reg=3.0, rsm=0.90, verbose=0, random_seed=123, thread_count=8)),
        ('d8_r5_s80_123', CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=8, l2_leaf_reg=5.0, rsm=0.80, verbose=0, random_seed=123, thread_count=8)),
        ('d8_r5_s90_42', CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=8, l2_leaf_reg=5.0, rsm=0.90, verbose=0, random_seed=42, thread_count=8)),
        ('d7_r3_s80_42', CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=7, l2_leaf_reg=3.0, rsm=0.80, verbose=0, random_seed=42, thread_count=8)),
        ('d7_r3_s90_123', CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=7, l2_leaf_reg=3.0, rsm=0.90, verbose=0, random_seed=123, thread_count=8)),
        ('d7_r5_s80_123', CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=7, l2_leaf_reg=5.0, rsm=0.80, verbose=0, random_seed=123, thread_count=8)),
        ('d7_r5_s90_42', CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=7, l2_leaf_reg=5.0, rsm=0.90, verbose=0, random_seed=42, thread_count=8)),
    ]
    weights = [1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0]
    return VotingClassifier(estimators=estimators, voting='soft', weights=weights)


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

    models = {
        'cb_subspace_iter1000_lr002': make_subspace_ensemble(1000, 0.02),
        'cb_subspace_iter1100_lr0018': make_subspace_ensemble(1100, 0.018),
        'cb_subspace_iter1200_lr0017': make_subspace_ensemble(1200, 0.017),
    }
    model = skrub.choose_from(models, name='schedule_variant')

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': eval_setup['scoring'], 'row_keys': eval_setup['row_keys']}