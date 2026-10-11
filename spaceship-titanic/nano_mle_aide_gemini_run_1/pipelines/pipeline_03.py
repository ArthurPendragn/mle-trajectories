"""
Deep learning tabular baseline for Spaceship Titanic passenger transport prediction.
Features are engineered by extracting cabin deck and side and computing total luxury amenity spend and zero-spend indicator.
Categoricals and numerics are preprocessed using TableVectorizer, median imputation, and standard scaling.
A PyTorch multi-layer perceptron with batch normalization and dropout is trained on GPU across multiple capacity configurations.
This explores a neural network model family distinct from previous gradient boosting and linear model drafts.
"""

import numpy as np
import skrub
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class TabularNN(nn.Module):
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class TabularMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_dim=128,
        dropout_rate=0.2,
        lr=1e-3,
        epochs=30,
        batch_size=128,
        random_state=42,
    ):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        y_arr = np.searchsorted(self.classes_, np.asarray(y))

        X_arr = np.asarray(X, dtype=np.float32)
        y_tensor = torch.tensor(y_arr, dtype=torch.long)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ = device

        in_features = X_arr.shape[1]
        self.net_ = TabularNN(
            in_features, hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate
        ).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        self.net_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = self.net_(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        self.net_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=self.batch_size * 2,
            shuffle=False,
        )

        probas = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device_)
                logits = self.net_(batch_x)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                probas.append(prob)
        return np.vstack(probas)

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        return self.classes_[pred_indices]


def make_mlp_pipeline(hidden_dim, dropout_rate, lr=1e-3, epochs=30):
    return Pipeline([
        (
            'vectorizer',
            skrub.TableVectorizer(
                cardinality_threshold=40,
                low_cardinality=OneHotEncoder(
                    sparse_output=False, handle_unknown='ignore'
                ),
            ),
        ),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        (
            'mlp',
            TabularMLPClassifier(
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                lr=lr,
                epochs=epochs,
                batch_size=128,
                random_state=42,
            ),
        ),
    ])


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    total_spend = (
        X['RoomService'].fillna(0.0)
        + X['FoodCourt'].fillna(0.0)
        + X['ShoppingMall'].fillna(0.0)
        + X['Spa'].fillna(0.0)
        + X['VRDeck'].fillna(0.0)
    )
    has_spend = (total_spend > 0.0).astype(float)
    deck = X['Cabin'].str[0]
    side = X['Cabin'].str[-1]

    drop_cols = ['PassengerId', 'Cabin', 'Name']
    features = (
        X.assign(
            Deck=deck,
            Side=side,
            TotalSpend=total_spend,
            HasSpend=has_spend,
        )
        .drop(columns=drop_cols)
        .skb.set_name('features')
    )

    models = {
        'mlp_h128_d02': make_mlp_pipeline(
            hidden_dim=128, dropout_rate=0.2, lr=1e-3, epochs=30
        ),
        'mlp_h256_d03': make_mlp_pipeline(
            hidden_dim=256, dropout_rate=0.3, lr=1e-3, epochs=30
        ),
    }
    chosen_model = skrub.choose_from(models, name='model_variant')
    pred = features.skb.apply(chosen_model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': setup['scoring'],
        'row_keys': setup['row_keys'],
    }