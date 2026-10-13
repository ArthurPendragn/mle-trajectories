import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

TRAIN_PATH = '/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv'
CLASS_COUNTS = {1: 1468136, 2: 2262087, 3: 195712, 4: 377, 5: 1, 6: 11426, 7: 62261}

def locked_setup_entry():
    data = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    data = data.sort_values('Id').reset_index(drop=True)
    rng = skrub.as_data_op(42).skb.apply_func(np.random.default_rng)
    pieces = []
    for label, count in CLASS_COUNTS.items():
        sample_size = max(1, int(np.floor(0.1 * count + 0.5)))
        positions = rng.choice(count, size=sample_size, replace=False)
        class_rows = data[data['Cover_Type'] == label]
        pieces.append(class_rows.iloc[positions])
    rows = pieces[0].skb.concat(pieces[1:], axis=0)
    rows = rows.sort_values('Id').reset_index(drop=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X = rows.drop(columns=['Id', 'Cover_Type']).skb.mark_as_X(cv=cv, split_kwargs={})
    y = rows['Cover_Type'].skb.mark_as_y()
    return {'X': X, 'y': y, 'scoring': 'accuracy', 'row_keys': rows['Id'], 'audit': {'class_counts': rows['Cover_Type'].value_counts().sort_index(), 'sample_shape': rows.shape, 'id_bounds': rows['Id'].agg(['min', 'max'])}}

def build_evaluation():
    return locked_setup_entry()

def build():
    setup = build_evaluation()
    X = setup['X']
    h = X['Horizontal_Distance_To_Hydrology'].astype('float64')
    v = X['Vertical_Distance_To_Hydrology'].astype('float64')
    r = X['Horizontal_Distance_To_Roadways'].astype('float64')
    f = X['Horizontal_Distance_To_Fire_Points'].astype('float64')
    e = X['Elevation'].astype('float64')
    geometry = X.assign(
        Hydrology_Euclidean_Distance=h.skb.apply_func(np.hypot, v),
        Hydrology_Elevation=e - v,
        Hydrology_Minus_Roadways=h - r,
        Hydrology_Minus_Fire_Points=h - f,
        Roadways_Minus_Fire_Points=r - f,
    )
    wilderness_count = X[[f'Wilderness_Area{i}' for i in range(1, 5)]].sum(axis=1)
    soil_count = X[[f'Soil_Type{i}' for i in range(1, 41)]].sum(axis=1)
    features = skrub.as_data_op(skrub.choose_from(
        [
            geometry,
            geometry.assign(Wilderness_Activation_Count=wilderness_count),
            geometry.assign(Soil_Activation_Count=soil_count),
            geometry.assign(
                Wilderness_Activation_Count=wilderness_count,
                Soil_Activation_Count=soil_count,
            ),
        ],
        name='activation_count_ablation',
    ))
    model = LGBMClassifier(
        objective='multiclass',
        n_estimators=350,
        learning_rate=0.07,
        num_leaves=63,
        min_child_samples=300,
        max_bin=511,
        colsample_bytree=1.0,
        subsample=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
        n_jobs=4,
    )
    pred = features.skb.apply(model, y=setup['y'])
    return {'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}