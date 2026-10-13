import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import skrub

TRAIN_PATH = '/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv'


def locked_setup_helper():
    raw = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X = raw.drop(columns=['Id', 'Cover_Type']).skb.mark_as_X(cv=cv, split_kwargs={})
    y = raw['Cover_Type'].skb.mark_as_y()
    row_keys = raw['Id']
    return {
        'X': X,
        'y': y,
        'scoring': 'accuracy',
        'row_keys': row_keys,
        'audit': {'target_counts': raw['Cover_Type'].value_counts().to_frame().reset_index()},
    }


def locked_setup_entry():
    return locked_setup_helper()


def build_evaluation():
    return locked_setup_entry()


def build():
    evaluation = build_evaluation()
    X = evaluation['X'].drop(columns=['Soil_Type7', 'Soil_Type15'])
    y = evaluation['y']
    model = lgb.LGBMClassifier(
        objective='multiclass',
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    pred = X.skb.apply(model, y=y)
    return {
        'pred': pred,
        'scoring': evaluation['scoring'],
        'row_keys': evaluation['row_keys'],
    }