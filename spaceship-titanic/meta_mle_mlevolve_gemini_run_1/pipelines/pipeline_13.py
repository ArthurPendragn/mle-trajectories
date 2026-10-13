import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    cabin_split = X['Cabin'].str.split('/', expand=True)
    deck = cabin_split[0].astype('string')
    cabin_num = cabin_split[1].astype('float64')
    side = cabin_split[2].astype('string')
    deck_side = (deck + '_' + side).astype('string')
    cabin_region = (cabin_num // 200).astype('float64')

    group_id = X['PassengerId'].str[:4].astype('string')
    group_id_num = group_id.astype('float64')
    group_num = X['PassengerId'].str[5:].astype('float64')
    is_group_lead = (group_num == 1.0).astype('float64')

    room_service_filled = X['RoomService'].fillna(0.0)
    food_court_filled = X['FoodCourt'].fillna(0.0)
    shopping_mall_filled = X['ShoppingMall'].fillna(0.0)
    spa_filled = X['Spa'].fillna(0.0)
    vr_deck_filled = X['VRDeck'].fillna(0.0)

    service_spend = room_service_filled + spa_filled + vr_deck_filled
    leisure_spend = food_court_filled + shopping_mall_filled
    total_spend = service_spend + leisure_spend
    has_spend = (total_spend > 0.0).astype('float64')

    spent_room_service = (room_service_filled > 0.0).astype('float64')
    spent_food_court = (food_court_filled > 0.0).astype('float64')
    spent_shopping_mall = (shopping_mall_filled > 0.0).astype('float64')
    spent_spa = (spa_filled > 0.0).astype('float64')
    spent_vr_deck = (vr_deck_filled > 0.0).astype('float64')
    num_amenities_spent = spent_room_service + spent_food_court + spent_shopping_mall + spent_spa + spent_vr_deck

    log_room_service = room_service_filled.skb.apply_func(np.log1p)
    log_food_court = food_court_filled.skb.apply_func(np.log1p)
    log_shopping_mall = shopping_mall_filled.skb.apply_func(np.log1p)
    log_spa = spa_filled.skb.apply_func(np.log1p)
    log_vr_deck = vr_deck_filled.skb.apply_func(np.log1p)
    log_service_spend = service_spend.skb.apply_func(np.log1p)
    log_leisure_spend = leisure_spend.skb.apply_func(np.log1p)
    log_total_spend = total_spend.skb.apply_func(np.log1p)

    spend_denom = total_spend + 1.0
    pct_room_service = room_service_filled / spend_denom
    pct_food_court = food_court_filled / spend_denom
    pct_shopping_mall = shopping_mall_filled / spend_denom
    pct_spa = spa_filled / spend_denom
    pct_vr_deck = vr_deck_filled / spend_denom
    pct_service = service_spend / spend_denom
    pct_leisure = leisure_spend / spend_denom

    spend_hhi = pct_room_service**2 + pct_food_court**2 + pct_shopping_mall**2 + pct_spa**2 + pct_vr_deck**2
    log_service_leisure_diff = log_service_spend - log_leisure_spend
    avg_spend_per_amenity = total_spend / (num_amenities_spent + 1.0)

    age_filled = X['Age'].fillna(-1.0)
    is_child = ((age_filled >= 0.0) & (age_filled <= 12.0)).astype('float64')
    is_infant = ((age_filled >= 0.0) & (age_filled <= 4.0)).astype('float64')
    spend_per_age = log_total_spend / (age_filled.clip(lower=1.0) + 1.0)

    home_planet = X['HomePlanet'].astype('string')
    is_europa_deck = deck.isin(['A', 'B', 'C', 'T']).fillna(False)
    is_earth_deck = (deck == 'G').fillna(False)
    home_planet_imputed = home_planet.mask(home_planet.isna() & is_europa_deck, 'Europa')
    home_planet_imputed = home_planet_imputed.mask(home_planet_imputed.isna() & is_earth_deck, 'Earth')

    destination = X['Destination'].astype('string')
    route = (home_planet_imputed + '_' + destination).astype('string')

    cryo_sleep_str = X['CryoSleep'].astype('string')
    should_impute_cryo = cryo_sleep_str.isna() & (total_spend > 0.0)
    cryo_sleep = cryo_sleep_str.mask(should_impute_cryo, 'False')
    vip = X['VIP'].astype('string')

    features = X.assign(
        RoomService=log_room_service,
        FoodCourt=log_food_court,
        ShoppingMall=log_shopping_mall,
        Spa=log_spa,
        VRDeck=log_vr_deck,
        ServiceSpend=log_service_spend,
        LeisureSpend=log_leisure_spend,
        TotalSpend=log_total_spend,
        HasSpend=has_spend,
        PctRoomService=pct_room_service,
        PctFoodCourt=pct_food_court,
        PctShoppingMall=pct_shopping_mall,
        PctSpa=pct_spa,
        PctVRDeck=pct_vr_deck,
        PctService=pct_service,
        PctLeisure=pct_leisure,
        NumAmenitiesSpent=num_amenities_spent,
        LogServiceLeisureDiff=log_service_leisure_diff,
        AvgSpendPerAmenity=avg_spend_per_amenity,
        SpendPerAge=spend_per_age,
        SpendHHI=spend_hhi,
        IsChild=is_child,
        IsInfant=is_infant,
        HomePlanet=home_planet_imputed,
        Destination=destination,
        Route=route,
        Deck=deck,
        CabinNum=cabin_num,
        CabinRegion=cabin_region,
        Side=side,
        DeckSide=deck_side,
        GroupIdNum=group_id_num,
        GroupNum=group_num,
        IsGroupLead=is_group_lead,
        CryoSleep=cryo_sleep,
        VIP=vip,
    ).drop(columns=['PassengerId', 'Cabin', 'Name']).skb.set_name('features')

    pipe_lgb = make_pipeline(
        skrub.TableVectorizer(),
        LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=8),
    )
    pipe_xgb = make_pipeline(
        skrub.TableVectorizer(),
        XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, eval_metric='logloss', tree_method='hist', n_jobs=8),
    )
    pipe_rf = make_pipeline(
        skrub.TableVectorizer(),
        SimpleImputer(strategy='median'),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=8),
    )
    pipe_et = make_pipeline(
        skrub.TableVectorizer(),
        SimpleImputer(strategy='median'),
        ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=8),
    )

    ensemble_rf = VotingClassifier(
        estimators=[('lgb', pipe_lgb), ('xgb', pipe_xgb), ('rf', pipe_rf)],
        voting='soft',
    )
    ensemble_et = VotingClassifier(
        estimators=[('lgb', pipe_lgb), ('xgb', pipe_xgb), ('et', pipe_et)],
        voting='soft',
    )

    model = skrub.choose_from(
        {
            'triple_lgb_xgb_rf': ensemble_rf,
            'triple_lgb_xgb_et': ensemble_et,
        },
        name='bagging_diversity',
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': scoring}