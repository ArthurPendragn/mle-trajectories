import gc
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# Load datasets
input_dir = "./input"
working_dir = "./working"
os.makedirs(working_dir, exist_ok=True)

print("Loading data...")
train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
test_ids = test_df["Id"].values

# Prune extremely rare training classes to allow valid StratifiedKFold
class_counts = train_df["Cover_Type"].value_counts()
rare_classes = class_counts[class_counts < 5].index.tolist()
if rare_classes:
    print(f"Pruning rare class(es) with <5 instances: {rare_classes}")
    train_df = train_df[~train_df["Cover_Type"].isin(rare_classes)].reset_index(
        drop=True
    )

# Map labels to 0-indexed contiguous integers
unique_classes = np.sort(train_df["Cover_Type"].unique())
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
idx_to_class = {idx: cls for idx, cls in enumerate(unique_classes)}
num_classes = len(unique_classes)
y_train = train_df["Cover_Type"].map(class_to_idx).values.astype(np.int64)


# Feature Engineering
def engineer_features(df):
    aspect_rad = np.radians(df["Aspect"])
    df["Aspect_Sin"] = np.sin(aspect_rad).astype(np.float32)
    df["Aspect_Cos"] = np.cos(aspect_rad).astype(np.float32)

    h_hydro = df["Horizontal_Distance_To_Hydrology"].astype(np.float32)
    v_hydro = df["Vertical_Distance_To_Hydrology"].astype(np.float32)
    df["Distance_To_Hydrology_3D"] = np.sqrt(h_hydro**2 + v_hydro**2).astype(np.float32)
    df["Hydrology_Elevation"] = (df["Elevation"] - v_hydro).astype(np.float32)
    df["Abs_Vertical_Dist_Hydrology"] = np.abs(v_hydro).astype(np.float32)

    h_road = df["Horizontal_Distance_To_Roadways"].astype(np.float32)
    h_fire = df["Horizontal_Distance_To_Fire_Points"].astype(np.float32)
    df["Hydro_plus_Road"] = (h_hydro + h_road).astype(np.float32)
    df["Hydro_minus_Road"] = (h_hydro - h_road).astype(np.float32)
    df["Hydro_plus_Fire"] = (h_hydro + h_fire).astype(np.float32)
    df["Hydro_minus_Fire"] = (h_hydro - h_fire).astype(np.float32)
    df["Road_plus_Fire"] = (h_road + h_fire).astype(np.float32)
    df["Road_minus_Fire"] = (h_road - h_fire).astype(np.float32)
    df["Total_Distance"] = (h_hydro + h_road + h_fire).astype(np.float32)
    df["Mean_Distance"] = (df["Total_Distance"] / 3.0).astype(np.float32)

    h_9am = df["Hillshade_9am"].astype(np.float32)
    h_noon = df["Hillshade_Noon"].astype(np.float32)
    h_3pm = df["Hillshade_3pm"].astype(np.float32)
    df["Hillshade_Mean"] = ((h_9am + h_noon + h_3pm) / 3.0).astype(np.float32)
    df["Hillshade_9am_minus_3pm"] = (h_9am - h_3pm).astype(np.float32)
    df["Hillshade_9am_minus_Noon"] = (h_9am - h_noon).astype(np.float32)
    df["Hillshade_Noon_minus_3pm"] = (h_noon - h_3pm).astype(np.float32)

    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    df["Wilderness_Count"] = df[wild_cols].sum(axis=1).astype(np.float32)
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
    df["Soil_Count"] = df[soil_cols].sum(axis=1).astype(np.float32)

    return df


print("Engineering features...")
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

ignore_cols = {"Id", "Cover_Type"}
feature_cols = [c for c in train_df.columns if c not in ignore_cols]

# Drop constant columns (such as Soil_Type7 and Soil_Type15)
constant_cols = [c for c in feature_cols if train_df[c].nunique() <= 1]
if constant_cols:
    print(f"Dropping uninformative constant columns: {constant_cols}")
    feature_cols = [c for c in feature_cols if c not in constant_cols]

print(f"Number of feature columns: {len(feature_cols)}")
X_train_full = train_df[feature_cols].values.astype(np.float32)
X_test_full = test_df[feature_cols].values.astype(np.float32)

del train_df, test_df
gc.collect()


# Tabular ResNet Architecture
class ResidualBlock(nn.Module):

    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        out = self.act(out + residual)
        return out


class TabularResNet(nn.Module):

    def __init__(
        self,
        in_features,
        num_classes,
        hidden_dim=256,
        num_blocks=3,
        dropout=0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout / 2), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# 5-Fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_train_full), num_classes), dtype=np.float32)
test_preds = np.zeros((len(X_test_full), num_classes), dtype=np.float32)

batch_size = 4096
epochs = 10
lr = 3e-3

start_time = time.time()
print(
    f"Starting 5-Fold Cross-Validation training across {len(X_train_full)} samples..."
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    fold_start = time.time()
    print(f"\n--- Fold {fold + 1} / {n_splits} ---")

    # Feature scaling fitted strictly on training fold
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_full[train_idx])
    X_va = scaler.transform(X_train_full[val_idx])
    y_tr = y_train[train_idx]
    y_va = y_train[val_idx]

    train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_dataset = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=True
    )

    model = TabularResNet(
        in_features=len(feature_cols),
        num_classes=num_classes,
        hidden_dim=256,
        num_blocks=3,
        dropout=0.15,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_y).sum().item()

        epoch_val_acc = correct / len(val_dataset)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"Fold {fold + 1} Best Validation Accuracy: {best_val_acc:.6f} (elapsed: {time.time() - fold_start:.1f}s)"
    )

    # Compute out-of-fold predictions with best weights
    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()

    oof_fold_probs = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            probs = torch.softmax(model(batch_x), dim=-1).cpu().numpy()
            oof_fold_probs.append(probs)
    oof_preds[val_idx] = np.concatenate(oof_fold_probs, axis=0)

    # Predict on test set for this fold
    X_te = scaler.transform(X_test_full)
    test_dataset = TensorDataset(torch.from_numpy(X_te))
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=True
    )

    fold_test_probs = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            probs = torch.softmax(model(batch_x), dim=-1).cpu().numpy()
            fold_test_probs.append(probs)
    test_preds += np.concatenate(fold_test_probs, axis=0) / n_splits

    del (
        model,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        val_loader,
        test_loader,
        best_weights,
    )
    del train_dataset, val_dataset, test_dataset, X_tr, X_va, X_te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Overall Out-of-Fold Validation Metric
oof_class_preds = oof_preds.argmax(axis=1)
overall_acc = accuracy_score(y_train, oof_class_preds)
print(f"\n==========================================")
print(f"Overall OOF Validation Accuracy: {overall_acc:.6f}")
print(f"Total CV Runtime: {(time.time() - start_time) / 60:.2f} minutes")
print(f"==========================================")

# Generate and save submission file
final_test_classes = test_preds.argmax(axis=1)
final_test_labels = [idx_to_class[i] for i in final_test_classes]

submission_path = os.path.join(working_dir, "submission.csv")
sub_df = pd.DataFrame({"Id": test_ids, "Cover_Type": final_test_labels})
sub_df.to_csv(submission_path, index=False)

print(f"\nSubmission successfully saved to {submission_path}")
print(f"Submission shape: {sub_df.shape}")
print("Class prediction distribution in test set:")
print(sub_df["Cover_Type"].value_counts().sort_index())
