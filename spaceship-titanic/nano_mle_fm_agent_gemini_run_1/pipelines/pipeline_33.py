import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import skrub

TEST_PATH = '/home/estrauss-ldap/datasets/space-ship-titanic/input/test.csv'


def build():
  setup = build_evaluation()
  X = setup['X']
  y = setup['y']

  train_manifest = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
  test_manifest = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)
  manifest = train_manifest[['PassengerId', 'Name']].skb.concat(
      [test_manifest[['PassengerId', 'Name']]], axis=0
  )

  manifest_passenger_id = manifest['PassengerId']
  manifest_group_id = manifest_passenger_id.str.slice(0, 4)
  manifest_name = manifest['Name']
  manifest_surname = manifest_name.str.split().str[-1]

  manifest_with_keys = manifest.assign(
      GroupIdStr=manifest_group_id, Surname=manifest_surname
  )

  group_counts = (
      manifest_with_keys[['GroupIdStr']]
      .groupby('GroupIdStr')
      .size()
      .reset_index(name='GroupSize')
  )
  family_counts = (
      manifest_with_keys[['Surname']]
      .groupby('Surname')
      .size()
      .reset_index(name='FamilySize')
  )

  cabin = X['Cabin']
  deck = cabin.str.slice(0, 1)
  side = cabin.str.slice(-1, None)
  deck_side = deck + '_' + side
  cabin_num = (
      cabin.str.split('/')
      .str[1]
      .skb.apply_func(pd.to_numeric, errors='coerce')
  )

  passenger_id = X['PassengerId']
  group_id_str = passenger_id.str.slice(0, 4)
  group_id = group_id_str.skb.apply_func(pd.to_numeric, errors='coerce')
  group_pos = (
      passenger_id.str.slice(5, 7)
      .skb.apply_func(pd.to_numeric, errors='coerce')
  )
  surname_str = X['Name'].str.split().str[-1]

  room_service = X['RoomService'].fillna(0.0)
  food_court = X['FoodCourt'].fillna(0.0)
  shopping_mall = X['ShoppingMall'].fillna(0.0)
  spa = X['Spa'].fillna(0.0)
  vr_deck = X['VRDeck'].fillna(0.0)
  total_spend = room_service + food_court + shopping_mall + spa + vr_deck
  has_spent = (total_spend > 0.0).astype('float64')
  spend_count = (
      (room_service > 0.0).astype('float64')
      + (food_court > 0.0).astype('float64')
      + (shopping_mall > 0.0).astype('float64')
      + (spa > 0.0).astype('float64')
      + (vr_deck > 0.0).astype('float64')
  )

  raw_cryo = X['CryoSleep'].map(
      {True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0}
  )
  cryo_sleep = raw_cryo.mask(total_spend > 0.0, 0.0)
  vip = X['VIP'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})

  luxury_spend = room_service + spa + vr_deck
  communal_spend = food_court + shopping_mall
  luxury_share = luxury_spend / (total_spend + 1.0)
  communal_share = communal_spend / (total_spend + 1.0)
  is_child = (X['Age'] <= 12.0).fillna(False).astype('float64')

  base_features = X[[
      'HomePlanet',
      'Destination',
      'Age',
      'RoomService',
      'FoodCourt',
      'ShoppingMall',
      'Spa',
      'VRDeck',
  ]]
  features_base = base_features.assign(
      CryoSleep=cryo_sleep,
      VIP=vip,
      CabinDeck=deck,
      CabinNum=cabin_num,
      CabinSide=side,
      CabinDeckSide=deck_side,
      TotalSpend=total_spend,
      HasSpent=has_spent,
      SpendCount=spend_count,
      GroupId=group_id,
      GroupPos=group_pos,
      GroupIdStr=group_id_str,
      Surname=surname_str,
  )

  features_with_group = features_base.merge(
      group_counts, on='GroupIdStr', how='left'
  )
  features_with_family = features_with_group.merge(
      family_counts, on='Surname', how='left'
  )

  group_size = features_with_family['GroupSize'].fillna(1.0).astype('float64')
  family_size = features_with_family['FamilySize'].fillna(1.0).astype('float64')
  is_alone = (
      (group_size == 1.0) & (features_with_family['GroupPos'] == 1.0)
  ).astype('float64')

  features_clean = features_with_family.drop(
      columns=['GroupIdStr', 'Surname', 'GroupSize', 'FamilySize']
  )
  features_decomp = features_clean.assign(
      LuxurySpend=luxury_spend, CommunalSpend=communal_spend
  )
  features_child = features_decomp.assign(
      LuxuryShare=luxury_share, IsChild=is_child
  )

  features_group = features_child.assign(
      GroupSize=group_size, IsAlone=is_alone
  )
  features_group_comm = features_group.assign(CommunalShare=communal_share)
  features_full = features_group_comm.assign(FamilySize=family_size)

  feature_candidates = {
      'group_size_and_alone': features_group,
      'group_and_communal_share': features_group_comm,
      'group_family_communal': features_full,
  }
  chosen_features = skrub.choose_from(
      feature_candidates, name='feature_engineering'
  )
  features_op = skrub.as_data_op(chosen_features)
  encoded_features = features_op.skb.apply(
      skrub.TableVectorizer(cardinality_threshold=40)
  ).skb.set_name('features')

  cat_500 = CatBoostClassifier(
      iterations=500,
      learning_rate=0.03,
      depth=6,
      random_seed=42,
      verbose=0,
      thread_count=4,
  )
  lgb_500 = LGBMClassifier(
      n_estimators=500,
      learning_rate=0.03,
      random_state=42,
      n_jobs=4,
      verbose=-1,
  )
  model = VotingClassifier(
      estimators=[('cat', cat_500), ('lgb', lgb_500)],
      voting='soft',
      weights=[3, 1],
  )
  pred = encoded_features.skb.apply(model, y=y).skb.set_name('model')

  return {'pred': pred, 'scoring': setup['scoring']}