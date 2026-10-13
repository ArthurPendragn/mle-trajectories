import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class _MLPModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=128,
        epochs=35,
        random_state=42,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_ = np.unique(y)

        X_arr = np.asarray(X, dtype=np.float32)
        y_bool = np.asarray(y) == self.classes_[1]
        y_arr = y_bool.astype(np.float32).reshape(-1, 1)

        in_dim = X_arr.shape[1]
        self.net_ = _MLPModule(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.net_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimizer.zero_grad()
                out = self.net_(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        self.net_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_arr).to(self.device_)
            logits = self.net_(X_tensor)
            p1 = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
            p0 = 1.0 - p1
            return np.hstack([p0, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    cabin_split = X['Cabin'].str.split('/', expand=True)
    deck = cabin_split[0].astype('string')
    cabin_num = cabin_split[1].astype('float64')
    side = cabin_split[2].astype('string')
    group_num = X['PassengerId'].str[5:].astype('float64')

    room_service = X['RoomService'].fillna(0.0)
    food_court = X['FoodCourt'].fillna(0.0)
    shopping_mall = X['ShoppingMall'].fillna(0.0)
    spa = X['Spa'].fillna(0.0)
    vr_deck = X['VRDeck'].fillna(0.0)

    total_spend = room_service + food_court + shopping_mall + spa + vr_deck
    has_spend = (total_spend > 0.0).astype('float64')
    cryo_sleep = X['CryoSleep'].astype('string')
    vip = X['VIP'].astype('string')

    features = (
        X.assign(
            Deck=deck,
            CabinNum=cabin_num,
            Side=side,
            GroupNum=group_num,
            TotalSpend=total_spend,
            HasSpend=has_spend,
            CryoSleep=cryo_sleep,
            VIP=vip,
        )
        .drop(columns=['PassengerId', 'Cabin', 'Name'])
        .skb.set_name('features')
    )

    mlp_variant = skrub.choose_from(
        {
            'mlp_h128_l3': TorchMLPClassifier(
                hidden_dim=128, num_layers=3, dropout=0.2, lr=1e-3, epochs=35
            ),
            'mlp_h256_l2': TorchMLPClassifier(
                hidden_dim=256, num_layers=2, dropout=0.15, lr=1e-3, epochs=35
            ),
        },
        name='mlp_config',
    )

    model = make_pipeline(
        skrub.TableVectorizer(),
        SimpleImputer(strategy='median'),
        StandardScaler(),
        mlp_variant,
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': scoring}