import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer

class LinearHostnameRanker(BaseEstimator):
    def __init__(self, weight_decay=0.00001, epochs=12, batch_size=512,
                 learning_rate=0.001, seed=42):
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.set_num_threads(16)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        X = X.tocsr().astype(np.float32)
        targets = np.asarray(y, dtype=np.float32)
        targets = targets / np.maximum(targets.sum(axis=1, keepdims=True), 1)
        self.net_ = nn.Linear(X.shape[1], targets.shape[1]).to(self.device_)
        optimizer = torch.optim.Adam(
            self.net_.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(torch.arange(X.shape[0])),
            batch_size=self.batch_size, shuffle=True,
            generator=generator, num_workers=0
        )
        self.net_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for indices, in loader:
                rows = indices.numpy()
                xb = torch.from_numpy(X[rows].toarray()).to(self.device_)
                yb = torch.from_numpy(targets[rows]).to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.net_(xb)
                loss = -(yb * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(rows)
            print('epoch', epoch + 1, 'loss', total_loss / X.shape[0])
        return self

    def predict(self, X):
        X = X.tocsr().astype(np.float32)
        self.net_.eval()
        output = []
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                xb = torch.from_numpy(
                    X[start:start + self.batch_size].toarray()
                ).to(self.device_)
                output.append(
                    self.net_(xb).topk(10, dim=1).indices.cpu().numpy()
                )
        return np.concatenate(output, axis=0)

def build():
    setup = build_evaluation()
    hosts = setup['X']['domain'].fillna('').str.lower().skb.set_name(
        'hostname_normalization'
    )
    features = hosts.skb.apply(
        HashingVectorizer(
            analyzer='char', ngram_range=(2, 5), n_features=16384,
            alternate_sign=False, norm='l2', lowercase=False,
            dtype=np.float32
        ),
        no_wrap=True
    ).skb.set_name('hostname_character_hash')
    model = skrub.choose_from(
        {
            'weak_l2': LinearHostnameRanker(weight_decay=0.000001),
            'moderate_l2': LinearHostnameRanker(weight_decay=0.0001)
        },
        name='linear_regularization'
    )
    pred = features.skb.apply(model, y=setup['y']).skb.set_name('model')
    return {
        'pred': pred,
        'scoring': setup['scoring'],
        'row_keys': setup['row_keys']
    }