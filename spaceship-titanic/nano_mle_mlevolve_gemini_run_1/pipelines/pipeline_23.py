import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

def make_spline_pipeline(n_knots, c_val, cont_cols, bin_cols, cat_cols):
    cont_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('spline', SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)), ('scaler', StandardScaler())])
    bin_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('cont', cont_transformer, cont_cols), ('bin', bin_transformer, bin_cols), ('cat', cat_transformer, cat_cols)])
    return Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(C=c_val, max_iter=1000, random_state=42))])

def make_mlp_pipeline(cont_cols, bin_cols, cat_cols):
    cont_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    bin_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('cont', cont_transformer, cont_cols), ('bin', bin_transformer, bin_cols), ('cat', cat_transformer, cat_cols)])
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', alpha=0.01, early_stopping=True, max_iter=500, random_state=42)
    return Pipeline([('preprocessor', preprocessor), ('classifier', mlp)])

def make_lgb_pipeline():
    return make_pipeline(skrub.TableVectorizer(), LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=8))

def make_xgb_pipeline():
    return make_pipeline(skrub.TableVectorizer(), XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, eval_metric='logloss', tree_method='hist', n_jobs=8))

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    train_manifest = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv, usecols=['PassengerId', 'HomePlanet', 'Cabin', 'Name'])
    test_manifest = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv, usecols=['PassengerId', 'HomePlanet', 'Cabin', 'Name'])
    all_manifest = train_manifest.skb.concat([test_manifest], axis=0)
    all_pids = all_manifest['PassengerId']
    all_groups = all_pids.str.split('_').str[0]
    group_counts = all_groups.skb.apply_func(pd.Series.value_counts).to_frame(name='GroupSize')
    group_id = X['PassengerId'].str.split('_').str[0]
    group_num = X['PassengerId'].str.split('_').str[1].astype('float64')
    group_size = X[['PassengerId']].assign(group_id=group_id).join(group_counts, on='group_id')['GroupSize'].astype('float64')
    is_alone = (group_size == 1.0).astype('float64')
    all_names = all_manifest['Name'].dropna()
    all_surnames = all_names.str.split(' ').str[-1]
    surname_counts = all_surnames.skb.apply_func(pd.Series.value_counts).to_frame(name='FamilySize')
    passenger_surname = X['Name'].str.split(' ').str[-1]
    family_size = X[['PassengerId']].assign(surname=passenger_surname).join(surname_counts, on='surname')['FamilySize'].fillna(1.0).astype('float64')
    is_family_alone = (family_size == 1.0).astype('float64')
    manifest_deck = all_manifest['Cabin'].str.split('/').str[0]
    manifest_side = all_manifest['Cabin'].str.split('/').str[2]
    group_planet_df = all_manifest[['HomePlanet']].assign(group_id=all_groups).dropna(subset=['HomePlanet']).drop_duplicates(subset=['group_id']).set_index('group_id')[['HomePlanet']].rename(columns={'HomePlanet': 'GroupHomePlanet'})
    companion_home_planet = X[['PassengerId']].assign(group_id=group_id).join(group_planet_df, on='group_id')['GroupHomePlanet'].astype(object)
    group_deck_df = all_manifest[['PassengerId']].assign(group_id=all_groups, deck=manifest_deck).dropna(subset=['deck']).drop_duplicates(subset=['group_id']).set_index('group_id')[['deck']].rename(columns={'deck': 'GroupDeck'})
    companion_deck = X[['PassengerId']].assign(group_id=group_id).join(group_deck_df, on='group_id')['GroupDeck'].astype(object)
    group_side_df = all_manifest[['PassengerId']].assign(group_id=all_groups, side=manifest_side).dropna(subset=['side']).drop_duplicates(subset=['group_id']).set_index('group_id')[['side']].rename(columns={'side': 'GroupSide'})
    companion_side = X[['PassengerId']].assign(group_id=group_id).join(group_side_df, on='group_id')['GroupSide'].astype(object)
    deck = X['Cabin'].str.split('/').str[0].astype(object)
    cabin_num = X['Cabin'].str.split('/').str[1].astype('float64')
    side = X['Cabin'].str.split('/').str[2].astype(object)
    deck = deck.fillna(companion_deck)
    side = side.fillna(companion_side)
    deck_side = deck.str.cat(side, sep='_').astype(object)
    deck = deck.where(deck.notna(), np.nan)
    side = side.where(side.notna(), np.nan)
    deck_side = deck_side.where(deck_side.notna(), np.nan)
    is_europa_deck = deck.isin(['A', 'B', 'C', 'T']).fillna(False)
    is_earth_deck = (deck == 'G').fillna(False)
    home_planet = X['HomePlanet'].astype(object)
    home_planet = home_planet.fillna(companion_home_planet)
    home_planet = home_planet.mask(home_planet.isna() & is_europa_deck, 'Europa')
    home_planet = home_planet.mask(home_planet.isna() & is_earth_deck, 'Earth')
    home_planet = home_planet.where(home_planet.notna(), np.nan).astype(object)
    room_service = X['RoomService'].fillna(0.0)
    food_court = X['FoodCourt'].fillna(0.0)
    shopping_mall = X['ShoppingMall'].fillna(0.0)
    spa = X['Spa'].fillna(0.0)
    vr_deck = X['VRDeck'].fillna(0.0)
    service_spend = room_service + spa + vr_deck
    leisure_spend = food_court + shopping_mall
    total_spend = service_spend + leisure_spend
    is_zero_spend = (total_spend == 0.0).astype('float64')
    has_room_service = (room_service > 0.0).astype('float64')
    has_food_court = (food_court > 0.0).astype('float64')
    has_shopping_mall = (shopping_mall > 0.0).astype('float64')
    has_spa = (spa > 0.0).astype('float64')
    has_vr_deck = (vr_deck > 0.0).astype('float64')
    log_total_spend = total_spend.skb.apply_func(np.log1p)
    log_room_service = room_service.skb.apply_func(np.log1p)
    log_food_court = food_court.skb.apply_func(np.log1p)
    log_shopping_mall = shopping_mall.skb.apply_func(np.log1p)
    log_spa = spa.skb.apply_func(np.log1p)
    log_vr_deck = vr_deck.skb.apply_func(np.log1p)
    log_service_spend = service_spend.skb.apply_func(np.log1p)
    log_leisure_spend = leisure_spend.skb.apply_func(np.log1p)
    log_service_leisure_diff = log_service_spend - log_leisure_spend
    cryo_sleep_raw = X['CryoSleep'].astype(object)
    should_impute_cryo = cryo_sleep_raw.isna() & (total_spend > 0.0)
    cryo_sleep = cryo_sleep_raw.mask(should_impute_cryo, False)
    cryo_sleep = cryo_sleep.where(cryo_sleep.notna(), np.nan).astype(object)
    features = X[['Age']].assign(Deck=deck, CabinNum=cabin_num, Side=side, DeckSide=deck_side, GroupNum=group_num, GroupSize=group_size, FamilySize=family_size, IsAlone=is_alone, IsFamilyAlone=is_family_alone, HomePlanet=home_planet, CryoSleep=cryo_sleep, Destination=X['Destination'].astype(object), VIP=X['VIP'].astype(object), IsZeroSpend=is_zero_spend, HasRoomService=has_room_service, HasFoodCourt=has_food_court, HasShoppingMall=has_shopping_mall, HasSpa=has_spa, HasVRDeck=has_vr_deck, LogTotalSpend=log_total_spend, LogRoomService=log_room_service, LogFoodCourt=log_food_court, LogShoppingMall=log_shopping_mall, LogSpa=log_spa, LogVRDeck=log_vr_deck, LogServiceLeisureDiff=log_service_leisure_diff).skb.set_name('features')
    cont_cols = ['Age', 'CabinNum', 'GroupNum', 'GroupSize', 'FamilySize', 'LogTotalSpend', 'LogRoomService', 'LogFoodCourt', 'LogShoppingMall', 'LogSpa', 'LogVRDeck', 'LogServiceLeisureDiff']
    bin_cols = ['IsZeroSpend', 'IsAlone', 'IsFamilyAlone', 'HasRoomService', 'HasFoodCourt', 'HasShoppingMall', 'HasSpa', 'HasVRDeck']
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide']
    pipe_gam = make_spline_pipeline(8, 1.0, cont_cols, bin_cols, cat_cols)
    pipe_mlp = make_mlp_pipeline(cont_cols, bin_cols, cat_cols)
    pipe_lgb = make_lgb_pipeline()
    pipe_xgb = make_xgb_pipeline()
    model = skrub.choose_from({
        'quad_blend': VotingClassifier(
            estimators=[('gam', pipe_gam), ('mlp', pipe_mlp), ('lgb', pipe_lgb), ('xgb', pipe_xgb)],
            voting='soft',
            weights=[1, 1, 2, 2],
        ),
        'tripartite': VotingClassifier(
            estimators=[('gam', pipe_gam), ('mlp', pipe_mlp), ('lgb', pipe_lgb)],
            voting='soft',
            weights=[1, 1, 2],
        ),
        'equal_quad': VotingClassifier(
            estimators=[('gam', pipe_gam), ('mlp', pipe_mlp), ('lgb', pipe_lgb), ('xgb', pipe_xgb)],
            voting='soft',
            weights=[1, 1, 1, 1],
        ),
    }, name='ensemble_architecture')
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}