import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
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

class FractionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, training_fraction=1.0, n_estimators=700, min_child_samples=30, max_bin=255):
        self.training_fraction = training_fraction
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.max_bin = max_bin

    def fit(self, X, y):
        labels = np.asarray(y)
        rng = np.random.default_rng(42)
        selected = []
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            permutation = rng.permutation(positions)
            count = max(1, round(self.training_fraction * len(positions)))
            selected.append(permutation[:count])
        indices = np.sort(np.concatenate(selected))
        self.training_row_count_ = len(indices)
        self.training_class_counts_ = dict(zip(*np.unique(labels[indices], return_counts=True)))
        self.model_ = LGBMClassifier(
            objective='multiclass',
            n_estimators=self.n_estimators,
            learning_rate=0.07,
            num_leaves=63,
            min_child_samples=self.min_child_samples,
            max_bin=self.max_bin,
            colsample_bytree=1.0,
            subsample=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            n_jobs=4
        )
        subset = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        self.model_.fit(subset, labels[indices])
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

def build():
    setup = build_evaluation()
    histogram_bins = skrub.choose_from([255, 511, 1023], name='histogram_bins')
    pred = setup['X'].skb.apply(
        FractionClassifier(
            training_fraction=1.0,
            n_estimators=350,
            min_child_samples=300,
            max_bin=histogram_bins
        ),
        y=setup['y']
    )
    return {'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}