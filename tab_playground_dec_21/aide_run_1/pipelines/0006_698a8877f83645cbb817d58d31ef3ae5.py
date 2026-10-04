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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    num_threads = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 4)
    )
    torch.set_num_threads(num_threads)
    print(f"Using compute device: {device} with {torch.get_num_threads()} CPU threads")
else:
    print(f"Using compute device: {device}")

# Directories
input_dir = "./input"
working_dir = "./working"
os.makedirs(working_dir, exist_ok=True)

print("Loading data...")
train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
test_ids = test_df["Id"].values

# Prune rare training classes (<5 instances) for valid StratifiedKFold
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


# Vectorized Feature Engineering
def engineer_features(df):
    aspect_rad = np.radians(df["Aspect"].values.astype(np.float32))
    h_hydro = df["Horizontal_Distance_To_Hydrology"].values.astype(np.float32)
    v_hydro = df["Vertical_Distance_To_Hydrology"].values.astype(np.float32)
    elev = df["Elevation"].values.astype(np.float32)
    h_road = df["Horizontal_Distance_To_Roadways"].values.astype(np.float32)
    h_fire = df["Horizontal_Distance_To_Fire_Points"].values.astype(np.float32)
    h_9am = df["Hillshade_9am"].values.astype(np.float32)
    h_noon = df["Hillshade_Noon"].values.astype(np.float32)
    h_3pm = df["Hillshade_3pm"].values.astype(np.float32)

    dist_3d = np.sqrt(h_hydro**2 + v_hydro**2).astype(np.float32)
    tot_dist = (h_hydro + h_road + h_fire).astype(np.float32)

    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]

    new_cols = {
        "Aspect_Sin": np.sin(aspect_rad).astype(np.float32),
        "Aspect_Cos": np.cos(aspect_rad).astype(np.float32),
        "Distance_To_Hydrology_3D": dist_3d,
        "Hydrology_Elevation": (elev - v_hydro).astype(np.float32),
        "Abs_Vertical_Dist_Hydrology": np.abs(v_hydro).astype(np.float32),
        "Hydro_plus_Road": (h_hydro + h_road).astype(np.float32),
        "Hydro_minus_Road": (h_hydro - h_road).astype(np.float32),
        "Hydro_plus_Fire": (h_hydro + h_fire).astype(np.float32),
        "Hydro_minus_Fire": (h_hydro - h_fire).astype(np.float32),
        "Road_plus_Fire": (h_road + h_fire).astype(np.float32),
        "Road_minus_Fire": (h_road - h_fire).astype(np.float32),
        "Total_Distance": tot_dist,
        "Mean_Distance": (tot_dist / 3.0).astype(np.float32),
        "Hillshade_Mean": ((h_9am + h_noon + h_3pm) / 3.0).astype(np.float32),
        "Hillshade_9am_minus_3pm": (h_9am - h_3pm).astype(np.float32),
        "Hillshade_9am_minus_Noon": (h_9am - h_noon).astype(np.float32),
        "Hillshade_Noon_minus_3pm": (h_noon - h_3pm).astype(np.float32),
        "Wilderness_Count": df[wild_cols].sum(axis=1).values.astype(np.float32),
        "Soil_Count": df[soil_cols].sum(axis=1).values.astype(np.float32),
    }
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


print("Engineering features...")
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

ignore_cols = {"Id", "Cover_Type"}
feature_cols = [c for c in train_df.columns if c not in ignore_cols]

# Drop uninformative constant columns
constant_cols = [c for c in feature_cols if train_df[c].nunique() <= 1]
if constant_cols:
    print(f"Dropping uninformative constant columns: {constant_cols}")
    feature_cols = [c for c in feature_cols if c not in constant_cols]

print(f"Number of feature columns: {len(feature_cols)}")
X_train_full = train_df[feature_cols].values.astype(np.float32)
X_test_full = test_df[feature_cols].values.astype(np.float32)

del train_df, test_df
gc.collect()

# Standardize in-place to minimize peak memory
print("Standardizing features in-place...")
scaler = StandardScaler(copy=False)
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)
X_test_tensor = torch.from_numpy(X_test_full)


# Fast Tabular ResNet Architecture
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
        num_blocks=2,
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

batch_size = 8192
val_batch_size = 16384
epochs = 8
lr = 4e-3

start_time = time.time()
print(
    f"Starting 5-Fold Cross-Validation training across {len(X_train_full)} samples..."
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    fold_start = time.time()
    print(f"\n--- Fold {fold + 1} / {n_splits} ---")

    X_tr_t = torch.from_numpy(X_train_full[train_idx])
    y_tr_t = torch.from_numpy(y_train[train_idx])
    X_va_t = torch.from_numpy(X_train_full[val_idx])
    y_va_t = torch.from_numpy(y_train[val_idx])

    n_train = len(X_tr_t)
    n_val = len(X_va_t)

    model = TabularResNet(
        in_features=len(feature_cols),
        num_classes=num_classes,
        hidden_dim=256,
        num_blocks=2,
        dropout=0.15,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_acc = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        # Fast direct batch indexing avoids PyTorch DataLoader item lookup overhead
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i : i + batch_size]
            batch_x = X_tr_t[batch_idx].to(device)
            batch_y = y_tr_t[batch_idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        scheduler.step()
        train_loss /= n_train

        # Fast contiguous validation evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, n_val, val_batch_size):
                batch_x = X_va_t[i : i + val_batch_size].to(device)
                batch_y = y_va_t[i : i + val_batch_size].to(device)
                preds = model(batch_x).argmax(dim=-1)
                correct += (preds == batch_y).sum().item()

        epoch_val_acc = correct / n_val
        print(
            f"Epoch {epoch:2d}/{epochs:2d} - Loss: {train_loss:.4f} - Val Acc:"
            f" {epoch_val_acc:.6f} - Elapsed: {time.time() - epoch_start:.1f}s"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"Fold {fold + 1} Best Validation Accuracy: {best_val_acc:.6f} (elapsed:"
        f" {time.time() - fold_start:.1f}s)"
    )

    # Compute out-of-fold predictions with best weights
    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()

    oof_fold_probs = []
    with torch.no_grad():
        for i in range(0, n_val, val_batch_size):
            batch_x = X_va_t[i : i + val_batch_size].to(device)
            probs = torch.softmax(model(batch_x), dim=-1).cpu().numpy()
            oof_fold_probs.append(probs)
    oof_preds[val_idx] = np.concatenate(oof_fold_probs, axis=0)

    # Predict on test set for this fold
    fold_test_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), val_batch_size):
            batch_x = X_test_tensor[i : i + val_batch_size].to(device)
            probs = torch.softmax(model(batch_x), dim=-1).cpu().numpy()
            fold_test_probs.append(probs)
    test_preds += np.concatenate(fold_test_probs, axis=0) / n_splits

    del (
        model,
        optimizer,
        scheduler,
        best_weights,
        X_tr_t,
        y_tr_t,
        X_va_t,
        y_va_t,
    )
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
