import copy
import gc
import math
import os
import random
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class ResNetBlock(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act2(out)
        out = self.dropout2(out)
        return out


class TabularResNet(nn.Module):

    def __init__(self, in_features, hidden_dim=256, num_blocks=3, num_classes=6):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout=0.1) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def engineer_features(df):
    df = df.copy()
    wilderness_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]

    # Trigonometric cyclic aspect
    aspect_rad = np.radians(df["Aspect"] % 360)
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)

    # 3D Euclidean distance and absolute elevation to hydrology
    df["Hydrology_Distance_3D"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )
    df["Elevation_Hydrology_Diff"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )
    df["Elevation_Hydrology_Sum"] = (
        df["Elevation"] + df["Vertical_Distance_To_Hydrology"]
    )

    # Pairwise distance interactions
    df["Dist_Hydro_Road_Sum"] = (
        df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Roadways"]
    )
    df["Dist_Hydro_Road_Diff"] = (
        df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Roadways"]
    )
    df["Dist_Hydro_Fire_Sum"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Dist_Hydro_Fire_Diff"] = (
        df["Horizontal_Distance_To_Hydrology"]
        - df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Dist_Road_Fire_Sum"] = (
        df["Horizontal_Distance_To_Roadways"] + df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Dist_Road_Fire_Diff"] = (
        df["Horizontal_Distance_To_Roadways"] - df["Horizontal_Distance_To_Fire_Points"]
    )

    # Hillshade lighting combinations
    df["Hillshade_Diff_9_Noon"] = df["Hillshade_9am"] - df["Hillshade_Noon"]
    df["Hillshade_Diff_Noon_3"] = df["Hillshade_Noon"] - df["Hillshade_3pm"]
    df["Hillshade_Diff_9_3"] = df["Hillshade_9am"] - df["Hillshade_3pm"]
    df["Hillshade_Sum"] = (
        df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]
    )
    df["Hillshade_Mean"] = df["Hillshade_Sum"] / 3.0

    # Summary indicator counts
    df["Wilderness_Area_Count"] = df[wilderness_cols].sum(axis=1)
    df["Soil_Type_Count"] = df[soil_cols].sum(axis=1)

    return df


def predict_proba(model, X_tensor, device, num_classes, batch_size=16384):
    model.eval()
    n = len(X_tensor)
    probs = np.empty((n, num_classes), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = X_tensor[start:end].to(device, non_blocking=True)
            logits = model(xb)
            probs[start:end] = F.softmax(logits, dim=-1).cpu().numpy()
    return probs


def main():
    start_time = time.time()
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computational device: {device}")

    print("Loading data...")
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")

    test_ids = test_df["Id"].values

    # Remove single-instance class 5 to permit valid stratified splitting
    train_df = train_df[train_df["Cover_Type"] != 5].reset_index(drop=True)

    y_raw = train_df["Cover_Type"].values
    train_df.drop(columns=["Id", "Cover_Type"], inplace=True)
    test_df.drop(columns=["Id"], inplace=True)

    # Encode target labels to contiguous 0-indexed integers
    classes = np.sort(np.unique(y_raw))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    num_classes = len(classes)

    print("Engineering features...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Systematically prune constant features
    constant_cols = [c for c in train_df.columns if train_df[c].nunique() <= 1]
    if constant_cols:
        print(f"Pruning constant columns: {constant_cols}")
        train_df.drop(columns=constant_cols, inplace=True)
        test_df.drop(columns=constant_cols, inplace=True)

    feature_cols = train_df.columns.tolist()
    num_features = len(feature_cols)
    print(f"Total engineered features: {num_features}")

    X_train_np = train_df.values.astype(np.float32)
    X_test_np = test_df.values.astype(np.float32)
    del train_df, test_df
    gc.collect()

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_probs = np.zeros((len(X_train_np), num_classes), dtype=np.float32)
    test_probs = np.zeros((len(X_test_np), num_classes), dtype=np.float32)

    batch_size = 4096 if torch.cuda.is_available() else 8192
    epochs = 10 if torch.cuda.is_available() else 6

    print(
        f"Starting {n_splits}-fold Stratified CV (epochs={epochs},"
        f" batch_size={batch_size})..."
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y)):
        fold_start = time.time()
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Feature standardization
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_np[train_idx]).astype(np.float32)
        X_va = scaler.transform(X_train_np[val_idx]).astype(np.float32)
        X_te = scaler.transform(X_test_np).astype(np.float32)

        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y[train_idx])
        X_va_t = torch.from_numpy(X_va)
        X_te_t = torch.from_numpy(X_te)

        del X_tr, X_va, X_te
        gc.collect()

        model = TabularResNet(
            in_features=num_features,
            hidden_dim=256,
            num_blocks=3,
            num_classes=num_classes,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

        n_batches = int(math.ceil(len(train_idx) / batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-3,
            total_steps=epochs * n_batches,
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1000,
        )

        best_val_acc = 0.0
        best_weights = None

        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(train_idx))

            for step in range(n_batches):
                b_idx = perm[step * batch_size : (step + 1) * batch_size]
                xb = X_tr_t[b_idx].to(device, non_blocking=True)
                yb = y_tr_t[b_idx].to(device, non_blocking=True)

                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                scheduler.step()

            val_preds_fold = predict_proba(
                model, X_va_t, device, num_classes, batch_size=16384
            )
            val_acc = accuracy_score(y[val_idx], np.argmax(val_preds_fold, axis=1))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

            print(
                f"Epoch {epoch:02d}/{epochs:02d} - Val Accuracy:"
                f" {val_acc:.6f} (Best: {best_val_acc:.6f})"
            )

        model.load_state_dict(best_weights)
        fold_oof = predict_proba(model, X_va_t, device, num_classes, batch_size=16384)
        oof_probs[val_idx] = fold_oof

        fold_test = predict_proba(model, X_te_t, device, num_classes, batch_size=16384)
        test_probs += fold_test / n_splits

        fold_time = time.time() - fold_start
        print(
            f"Fold {fold + 1} Best Accuracy: {best_val_acc:.6f} (Time:"
            f" {fold_time / 60:.2f} min)"
        )

        del model, X_tr_t, y_tr_t, X_va_t, X_te_t, best_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_preds = np.argmax(oof_probs, axis=1)
    overall_accuracy = accuracy_score(y, oof_preds)
    print(f"\nValidation Accuracy: {overall_accuracy:.6f}")

    print("Generating submission...")
    final_test_preds = np.argmax(test_probs, axis=1)
    final_test_labels = [idx_to_class[p] for p in final_test_preds]

    submission = pd.DataFrame({"Id": test_ids, "Cover_Type": final_test_labels})

    os.makedirs("./working", exist_ok=True)
    submission_path = "./working/submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Submission successfully saved to {submission_path}")
    print(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
