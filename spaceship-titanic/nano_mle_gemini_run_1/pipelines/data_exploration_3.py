import pandas as pd
import skrub

TRAIN_PATH = '/home/estrauss-ldap/datasets/space-ship-titanic/input/train.csv'
TEST_PATH = '/home/estrauss-ldap/datasets/space-ship-titanic/input/test.csv'

def build():
    train_raw = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    test_raw = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    amenity_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    train_cabin_parts = train_raw['Cabin'].str.split('/', expand=True)
    train_cabin_num = train_cabin_parts[1].skb.apply_func(pd.to_numeric, errors='coerce')

    train_extracted = train_raw.assign(
        GroupId=train_raw['PassengerId'].str.slice(0, 4),
        Deck=train_cabin_parts[0],
        CabinNum=train_cabin_num,
        Side=train_cabin_parts[2],
        Surname=train_raw['Name'].str.split().str[-1],
        TotalSpend=train_raw[amenity_cols].fillna(0.0).sum(axis=1),
        Transported_float=train_raw['Transported'].astype(float),
    )

    test_cabin_parts = test_raw['Cabin'].str.split('/', expand=True)
    test_cabin_num = test_cabin_parts[1].skb.apply_func(pd.to_numeric, errors='coerce')

    test_extracted = test_raw.assign(
        GroupId=test_raw['PassengerId'].str.slice(0, 4),
        Deck=test_cabin_parts[0],
        CabinNum=test_cabin_num,
        Side=test_cabin_parts[2],
        Surname=test_raw['Name'].str.split().str[-1],
        TotalSpend=test_raw[amenity_cols].fillna(0.0).sum(axis=1),
    )

    # 1. Group Homogeneity for HomePlanet, Destination, Deck, Side
    train_group_summary = train_extracted.groupby('GroupId').agg(
        group_size=('PassengerId', 'count'),
        hp_nunique=('HomePlanet', 'nunique'),
        hp_count=('HomePlanet', 'count'),
        dest_nunique=('Destination', 'nunique'),
        dest_count=('Destination', 'count'),
        deck_nunique=('Deck', 'nunique'),
        deck_count=('Deck', 'count'),
        side_nunique=('Side', 'nunique'),
        side_count=('Side', 'count'),
    )

    train_multi_groups = train_group_summary[train_group_summary['group_size'] >= 2]
    hp_nunique_dist = train_multi_groups[train_multi_groups['hp_count'] >= 2]['hp_nunique'].value_counts()
    dest_nunique_dist = train_multi_groups[train_multi_groups['dest_count'] >= 2]['dest_nunique'].value_counts()
    deck_nunique_dist = train_multi_groups[train_multi_groups['deck_count'] >= 2]['deck_nunique'].value_counts()
    side_nunique_dist = train_multi_groups[train_multi_groups['side_count'] >= 2]['side_nunique'].value_counts()

    # Imputation recoverability via group co-travelers in train and test
    train_merged = train_extracted.merge(
        train_group_summary[['group_size', 'hp_count', 'dest_count', 'deck_count', 'side_count']],
        on='GroupId',
        how='left',
    )
    train_imputation_counts = train_merged.assign(
        hp_missing=train_merged['HomePlanet'].isna(),
        hp_recoverable=train_merged['HomePlanet'].isna() & (train_merged['hp_count'] > 0),
        cabin_missing=train_merged['Cabin'].isna(),
        cabin_recoverable=train_merged['Cabin'].isna() & (train_merged['deck_count'] > 0),
        dest_missing=train_merged['Destination'].isna(),
        dest_recoverable=train_merged['Destination'].isna() & (train_merged['dest_count'] > 0),
    )[['hp_missing', 'hp_recoverable', 'cabin_missing', 'cabin_recoverable', 'dest_missing', 'dest_recoverable']].sum()

    test_group_summary = test_extracted.groupby('GroupId').agg(
        group_size=('PassengerId', 'count'),
        hp_nunique=('HomePlanet', 'nunique'),
        hp_count=('HomePlanet', 'count'),
        dest_nunique=('Destination', 'nunique'),
        dest_count=('Destination', 'count'),
        deck_nunique=('Deck', 'nunique'),
        deck_count=('Deck', 'count'),
        side_nunique=('Side', 'nunique'),
        side_count=('Side', 'count'),
    )
    test_merged = test_extracted.merge(
        test_group_summary[['group_size', 'hp_count', 'dest_count', 'deck_count', 'side_count']],
        on='GroupId',
        how='left',
    )
    test_imputation_counts = test_merged.assign(
        hp_missing=test_merged['HomePlanet'].isna(),
        hp_recoverable=test_merged['HomePlanet'].isna() & (test_merged['hp_count'] > 0),
        cabin_missing=test_merged['Cabin'].isna(),
        cabin_recoverable=test_merged['Cabin'].isna() & (test_merged['deck_count'] > 0),
        dest_missing=test_merged['Destination'].isna(),
        dest_recoverable=test_merged['Destination'].isna() & (test_merged['dest_count'] > 0),
    )[['hp_missing', 'hp_recoverable', 'cabin_missing', 'cabin_recoverable', 'dest_missing', 'dest_recoverable']].sum()

    # 2. Homogeneity of HomePlanet by surname
    surname_valid = train_extracted.dropna(subset=['Surname', 'HomePlanet'])
    surname_summary = surname_valid.groupby('Surname').agg(
        surname_count=('PassengerId', 'count'),
        hp_nunique=('HomePlanet', 'nunique'),
    )
    surname_multi = surname_summary[surname_summary['surname_count'] >= 2]
    surname_hp_nunique_dist = surname_multi['hp_nunique'].value_counts()
    surname_multi_summary = surname_multi['hp_nunique'].describe()

    # 3. CryoSleep spending constraints in train and test
    train_cryo_flags = train_extracted.assign(
        cryo_true=(train_extracted['CryoSleep'] == True),
        cryo_true_spend_gt_0=(train_extracted['CryoSleep'] == True) & (train_extracted['TotalSpend'] > 0),
        cryo_na=train_extracted['CryoSleep'].isna(),
        cryo_na_spend_gt_0=train_extracted['CryoSleep'].isna() & (train_extracted['TotalSpend'] > 0),
        cryo_na_spend_eq_0=train_extracted['CryoSleep'].isna() & (train_extracted['TotalSpend'] == 0),
    )[['cryo_true', 'cryo_true_spend_gt_0', 'cryo_na', 'cryo_na_spend_gt_0', 'cryo_na_spend_eq_0']].sum()

    test_cryo_flags = test_extracted.assign(
        cryo_true=(test_extracted['CryoSleep'] == True),
        cryo_true_spend_gt_0=(test_extracted['CryoSleep'] == True) & (test_extracted['TotalSpend'] > 0),
        cryo_na=test_extracted['CryoSleep'].isna(),
        cryo_na_spend_gt_0=test_extracted['CryoSleep'].isna() & (test_extracted['TotalSpend'] > 0),
        cryo_na_spend_eq_0=test_extracted['CryoSleep'].isna() & (test_extracted['TotalSpend'] == 0),
    )[['cryo_true', 'cryo_true_spend_gt_0', 'cryo_na', 'cryo_na_spend_gt_0', 'cryo_na_spend_eq_0']].sum()

    cryo_true_spend_stats = train_extracted[train_extracted['CryoSleep'] == True]['TotalSpend'].describe()

    # 4. Transported base rates by Cabin Deck, Side, Deck*Side, and CabinNum correlation
    deck_rates = train_extracted.groupby('Deck')['Transported_float'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    side_rates = train_extracted.groupby('Side')['Transported_float'].agg(['count', 'mean']).sort_values('mean', ascending=False)

    deck_side_extracted = train_extracted.assign(
        DeckSide=train_extracted['Deck'] + '/' + train_extracted['Side']
    )
    deck_side_rates = deck_side_extracted.groupby('DeckSide')['Transported_float'].agg(['count', 'mean']).sort_values('mean', ascending=False)

    cabin_clean = train_extracted.dropna(subset=['CabinNum', 'Deck', 'Transported_float'])
    deck_num_corr = cabin_clean.groupby('Deck')[['CabinNum', 'Transported_float']].corr().xs('CabinNum', level=1)[['Transported_float']].rename(columns={'Transported_float': 'corr_CabinNum_Transported'})
    deck_cabin_stats = cabin_clean.groupby('Deck').agg(
        passenger_count=('CabinNum', 'count'),
        min_cabin_num=('CabinNum', 'min'),
        median_cabin_num=('CabinNum', 'median'),
        max_cabin_num=('CabinNum', 'max'),
        transported_rate=('Transported_float', 'mean'),
    )
    deck_cabin_summary = deck_cabin_stats.merge(deck_num_corr, left_index=True, right_index=True)
    overall_cabin_num_corr = cabin_clean[['CabinNum', 'Transported_float']].corr()

    return {
        'group_hp_nunique_dist': hp_nunique_dist,
        'group_dest_nunique_dist': dest_nunique_dist,
        'group_deck_nunique_dist': deck_nunique_dist,
        'group_side_nunique_dist': side_nunique_dist,
        'train_imputation_counts': train_imputation_counts,
        'test_imputation_counts': test_imputation_counts,
        'surname_hp_nunique_dist': surname_hp_nunique_dist,
        'surname_multi_summary': surname_multi_summary,
        'train_cryo_spend_rules': train_cryo_flags,
        'test_cryo_spend_rules': test_cryo_flags,
        'cryo_true_spend_stats': cryo_true_spend_stats,
        'deck_rates': deck_rates,
        'side_rates': side_rates,
        'deck_side_rates': deck_side_rates,
        'deck_cabin_summary': deck_cabin_summary,
        'overall_cabin_num_corr': overall_cabin_num_corr,
    }