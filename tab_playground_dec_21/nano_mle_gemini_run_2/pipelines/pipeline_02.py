import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
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


class SubsampleLGBMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        subsample_fraction=1.0,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ):
        self.subsample_fraction = subsample_fraction
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        n_samples = len(X)
        if self.subsample_fraction < 1.0:
            rng = np.random.RandomState(self.random_state)
            perm = rng.permutation(n_samples)
            k = int(np.round(n_samples * self.subsample_fraction))
            indices = perm[:k]
            if hasattr(X, "iloc"):
                X_sub = X.iloc[indices]
            else:
                X_sub = X[indices]
            if hasattr(y, "iloc"):
                y_sub = y.iloc[indices]
            else:
                y_sub = y[indices]
        else:
            X_sub = X
            y_sub = y

        self.model_ = lgb.LGBMClassifier(
            objective='multiclass',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        self.model_.fit(X_sub, y_sub)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def build():
    evaluation = build_evaluation()
    X = evaluation['X'].drop(columns=['Soil_Type7', 'Soil_Type15'])
    y = evaluation['y']

    model = skrub.choose_from(
        {
            'subsample_0.2': SubsampleLGBMClassifier(
                subsample_fraction=0.2,
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=63,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            'subsample_0.5': SubsampleLGBMClassifier(
                subsample_fraction=0.5,
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=63,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            'subsample_1.0': SubsampleLGBMClassifier(
                subsample_fraction=1.0,
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=63,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        },
        name='subsample_fraction',
    )

    pred = X.skb.apply(model, y=y)
    return {'pred': pred, 'scoring': evaluation['scoring'], 'row_keys': evaluation['row_keys']}