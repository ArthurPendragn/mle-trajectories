import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

class _ResNetBlock(nn.Module):

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.linear1(out)
        out = self.drop1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.linear2(out)
        out = self.drop2(out)
        return residual + out

class _ResNetModule(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_blocks, dropout):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([_ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head_norm = nn.BatchNorm1d(hidden_dim)
        self.head_relu = nn.ReLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_norm(h)
        h = self.head_relu(h)
        return self.head(h)

class TorchTabularResNetClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, hidden_dim=128, num_blocks=2, dropout=0.2, lr=0.001, weight_decay=0.0001, batch_size=128, epochs=35, random_state=42):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
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
        self.net_ = _ResNetModule(in_dim=in_dim, hidden_dim=self.hidden_dim, num_blocks=self.num_blocks, dropout=self.dropout).to(self.device_)
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-05)
        criterion = nn.BCEWithLogitsLoss()
        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            self.net_.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimizer.zero_grad()
                out = self.net_(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
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

    room_service_filled = X['RoomService'].fillna(0.0)
    food_court_filled = X['FoodCourt'].fillna(0.0)
    shopping_mall_filled = X['ShoppingMall'].fillna(0.0)
    spa_filled = X['Spa'].fillna(0.0)
    vr_deck_filled = X['VRDeck'].fillna(0.0)
    total_spend = room_service_filled + food_court_filled + shopping_mall_filled + spa_filled + vr_deck_filled
    has_spend = (total_spend > 0.0).astype('float64')

    log_room_service = room_service_filled.skb.apply_func(np.log1p)
    log_food_court = food_court_filled.skb.apply_func(np.log1p)
    log_shopping_mall = shopping_mall_filled.skb.apply_func(np.log1p)
    log_spa = spa_filled.skb.apply_func(np.log1p)
    log_vr_deck = vr_deck_filled.skb.apply_func(np.log1p)
    log_total_spend = total_spend.skb.apply_func(np.log1p)

    cryo_sleep = X['CryoSleep'].astype('string')
    vip = X['VIP'].astype('string')

    features = X.assign(
        RoomService=log_room_service,
        FoodCourt=log_food_court,
        ShoppingMall=log_shopping_mall,
        Spa=log_spa,
        VRDeck=log_vr_deck,
        TotalSpend=log_total_spend,
        HasSpend=has_spend,
        Deck=deck,
        CabinNum=cabin_num,
        Side=side,
        GroupNum=group_num,
        CryoSleep=cryo_sleep,
        VIP=vip
    ).drop(columns=['PassengerId', 'Cabin', 'Name']).skb.set_name('features')

    pipe_lgb = make_pipeline(
        skrub.TableVectorizer(),
        LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=8)
    )

    pipe_resnet = make_pipeline(
        skrub.TableVectorizer(),
        SimpleImputer(strategy='median'),
        StandardScaler(),
        TorchTabularResNetClassifier(hidden_dim=128, num_blocks=2, dropout=0.2, lr=0.001, epochs=35)
    )

    models = {
        'equal_1_1': VotingClassifier(
            estimators=[('lgb', pipe_lgb), ('resnet', pipe_resnet)],
            voting='soft',
            weights=[1, 1]
        ),
        'gbdt_heavy_2_1': VotingClassifier(
            estimators=[('lgb', pipe_lgb), ('resnet', pipe_resnet)],
            voting='soft',
            weights=[2, 1]
        )
    }

    model_choice = skrub.choose_from(models, name='ensemble_weights')
    pred = features.skb.apply(model_choice, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': scoring}