"""Cross-family gradient boosting ensemble combining oblivious, depth-wise, and leaf-wise tree architectures.
The CatBoost-only model family plateaued across depth and regularization variations around 0.82089 accuracy.
Oblivious trees split on identical features across branches; by contrast, XGBoost builds asymmetric depth-wise trees with Hessian shrinkage and row subsampling, and LightGBM grows leaf-wise best-first partitions.
This expansion anchors on the top-performing 8-model depth-7/8 CatBoost backbone and blends it into multi-family soft-voting ensembles.
We evaluate three complementary cross-family architectures: an 8-CatBoost + 2-XGBoost ensemble, an 8-CatBoost + 4-XGBoost multi-depth ensemble, and a 12-model tri-family ensemble incorporating CatBoost, XGBoost, and LightGBM.
"""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder
import skrub
from xgboost import XGBClassifier


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

    cb_estimators = [
        ('cb_d8_r3_42', cb_d8_r3_42),
        ('cb_d8_r3_123', cb_d8_r3_123),
        ('cb_d8_r5_42', cb_d8_r5_42),
        ('cb_d8_r5_123', cb_d8_r5_123),
        ('cb_d7_r3_42', cb_d7_r3_42),
        ('cb_d7_r3_123', cb_d7_r3_123),
        ('cb_d7_r5_42', cb_d7_r5_42),
        ('cb_d7_r5_123', cb_d7_r5_123),
    ]

    xgb_d6_r3_42 = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        eval_metric='logloss',
        random_state=42,
        n_jobs=8,
    )
    xgb_d6_r5_123 = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        eval_metric='logloss',
        random_state=123,
        n_jobs=8,
    )
    xgb_d5_r3_42 = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        eval_metric='logloss',
        random_state=42,
        n_jobs=8,
    )
    xgb_d5_r5_123 = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        eval_metric='logloss',
        random_state=123,
        n_jobs=8,
    )

    lgb_l31_r3_42 = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=8,
        verbose=-1,
    )
    lgb_l31_r5_123 = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        random_state=123,
        n_jobs=8,
        verbose=-1,
    )

    ensemble_cb_xgb_10m = VotingClassifier(
        estimators=cb_estimators
        + [('xgb_d6_r3_42', xgb_d6_r3_42), ('xgb_d6_r5_123', xgb_d6_r5_123)],
        voting='soft',
    )

    ensemble_cb_xgb_12m = VotingClassifier(
        estimators=cb_estimators
        + [
            ('xgb_d6_r3_42', xgb_d6_r3_42),
            ('xgb_d6_r5_123', xgb_d6_r5_123),
            ('xgb_d5_r3_42', xgb_d5_r3_42),
            ('xgb_d5_r5_123', xgb_d5_r5_123),
        ],
        voting='soft',
    )

    ensemble_tri_family_12m = VotingClassifier(
        estimators=cb_estimators
        + [
            ('xgb_d6_r3_42', xgb_d6_r3_42),
            ('xgb_d5_r5_123', xgb_d5_r5_123),
            ('lgb_l31_r3_42', lgb_l31_r3_42),
            ('lgb_l31_r5_123', lgb_l31_r5_123),
        ],
        voting='soft',
    )

    models = {
        'cb_xgb_10m': ensemble_cb_xgb_10m,
        'cb_xgb_12m': ensemble_cb_xgb_12m,
        'cb_xgb_lgb_tri_family_12m': ensemble_tri_family_12m,
    }

    model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': eval_setup['scoring'],
        'row_keys': eval_setup['row_keys'],
    }