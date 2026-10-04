import os
import gc
import copy
import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, num_features, embed_dim) * 0.02)
        self.biases = nn.Parameter(torch.zeros(1, num_features, embed_dim))

    def forward(self, x):
        # x: (batch_size, num_features) -> (batch_size, num_features, embed_dim)
        return x.unsqueeze(-1) * self.weights + self.biases


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout1(attn_out)

        normed2 = self.norm2(x)
        ffn_out = self.ffn(normed2)
        x = x + self.dropout2(ffn_out)
        return x


class WideFTTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes=7,
        embed_dim=48,
        num_heads=4,
        num_layers=2,
        ffn_ratio=2,
        dropout=0.05,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    embed_dim * ffn_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.linear_skip = nn.Linear(num_features, num_classes)

        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.linear_skip.weight, std=0.01)
        nn.init.zeros_(self.linear_skip.bias)

    def forward(self, x):
        batch_size = x.size(0)
        tokens = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        z = torch.cat([cls_tokens, tokens], dim=1)

        for block in self.blocks:
            z = block(z)

        cls_out = self.final_norm(z[:, 0])
        logits = self.head(cls_out) + self.linear_skip(x)
        return logits


def engineer_features(df):
    feats = pd.DataFrame(index=df.index)

    base_cont = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
    ]
    for col in base_cont:
        feats[col] = df[col].astype(np.float32)

    h_dist = df["Horizontal_Distance_To_Hydrology"].values.astype(np.float32)
    v_dist = df["Vertical_Distance_To_Hydrology"].values.astype(np.float32)
    elev = df["Elevation"].values.astype(np.float32)
    r_dist = df["Horizontal_Distance_To_Roadways"].values.astype(np.float32)
    f_dist = df["Horizontal_Distance_To_Fire_Points"].values.astype(np.float32)

    feats["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        h_dist**2 + v_dist**2
    ).astype(np.float32)
    feats["Hydrology_Elevation"] = (elev - v_dist).astype(np.float32)
    feats["Hydrology_Elevation_Sum"] = (elev + v_dist).astype(np.float32)

    feats["Hydro_Road_Sum"] = (h_dist + r_dist).astype(np.float32)
    feats["Hydro_Road_Diff"] = np.abs(h_dist - r_dist).astype(np.float32)
    feats["Hydro_Fire_Sum"] = (h_dist + f_dist).astype(np.float32)
    feats["Hydro_Fire_Diff"] = np.abs(h_dist - f_dist).astype(np.float32)
    feats["Road_Fire_Sum"] = (r_dist + f_dist).astype(np.float32)
    feats["Road_Fire_Diff"] = np.abs(r_dist - f_dist).astype(np.float32)

    aspect_rad = df["Aspect"].values.astype(np.float32) * (np.pi / 180.0)
    feats["Aspect_Sin"] = np.sin(aspect_rad).astype(np.float32)
    feats["Aspect_Cos"] = np.cos(aspect_rad).astype(np.float32)

    h9 = df["Hillshade_9am"].values.astype(np.float32)
    h12 = df["Hillshade_Noon"].values.astype(np.float32)
    h15 = df["Hillshade_3pm"].values.astype(np.float32)
    feats["Hillshade_9_12_Diff"] = (h9 - h12).astype(np.float32)
    feats["Hillshade_12_15_Diff"] = (h12 - h15).astype(np.float32)
    feats["Hillshade_9_15_Diff"] = (h9 - h15).astype(np.float32)
    feats["Hillshade_Mean"] = ((h9 + h12 + h15) / 3.0).astype(np.float32)
    feats["Hillshade_Std"] = np.std(np.stack([h9, h12, h15], axis=1), axis=1).astype(
        np.float32
    )

    feats["Slope_Hydro_Interaction"] = (
        df["Slope"].values.astype(np.float32) * h_dist / 100.0
    ).astype(np.float32)

    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]

    for c in wild_cols:
        feats[c] = df[c].astype(np.float32)
    for c in soil_cols:
        feats[c] = df[c].astype(np.float32)

    feats["Wilderness_Count"] = df[wild_cols].sum(axis=1).astype(np.float32)
    feats["Soil_Count"] = df[soil_cols].sum(axis=1).astype(np.float32)
    feats["Wilderness_Code"] = df[wild_cols].values.argmax(axis=1).astype(np.float32)
    feats["Soil_Code"] = df[soil_cols].values.argmax(axis=1).astype(np.float32)

    return feats


def main():
    start_time = time.time()
    seed_everything(42)

    print("Loading data...")
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train raw shape: {train_df.shape}, Test raw shape: {test_df.shape}")

    # Remove single-instance / rare classes to ensure stratified CV validity
    class_counts = train_df["Cover_Type"].value_counts()
    rare_classes = class_counts[class_counts < 5].index.tolist()
    if rare_classes:
        print(f"Filtering out rare classes with < 5 instances: {rare_classes}")
        train_df = train_df[~train_df["Cover_Type"].isin(rare_classes)].reset_index(
            drop=True
        )

    test_ids = test_df["Id"].values
    y_raw = train_df["Cover_Type"].values
    # Classes are 1-7, mapped to 0-6 for cross-entropy
    y = (y_raw - 1).astype(np.int64)

    print("Engineering features...")
    train_feats = engineer_features(train_df)
    test_feats = engineer_features(test_df)

    del train_df, test_df
    gc.collect()

    # Drop zero-variance constant features
    constant_cols = [c for c in train_feats.columns if train_feats[c].std() == 0.0]
    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        train_feats.drop(columns=constant_cols, inplace=True)
        test_feats.drop(columns=constant_cols, inplace=True)

    num_features = train_feats.shape[1]
    print(f"Final feature count: {num_features}")

    X_train = train_feats.values.astype(np.float32)
    X_test = test_feats.values.astype(np.float32)
    del train_feats, test_feats
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    print(f"Compute device: {device} (is_cuda={is_cuda})")
    if is_cuda:
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    n_splits = 5
    epochs = 4
    batch_size = 4096
    eval_batch_size = 8192

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_probs = np.zeros((len(y), 7), dtype=np.float32)
    test_probs = np.zeros((len(X_test), 7), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        fold_start = time.time()
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[train_idx])
        X_va = scaler.transform(X_train[val_idx])
        X_te = scaler.transform(X_test)

        train_dataset = TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y[train_idx])
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_va), torch.from_numpy(y[val_idx])
        )
        test_dataset = TensorDataset(torch.from_numpy(X_te))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=is_cuda,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            pin_memory=is_cuda,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            pin_memory=is_cuda,
            num_workers=0,
        )

        model = WideFTTransformer(
            num_features=num_features,
            num_classes=7,
            embed_dim=48,
            num_heads=4,
            num_layers=2,
            ffn_ratio=2,
            dropout=0.05,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2.5e-3,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0,
        )
        amp_scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=is_cuda):
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)

                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()

                train_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == batch_y).sum().item()
                train_total += batch_x.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=is_cuda):
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y)

                    val_loss += loss.item() * batch_x.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_x.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.5f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.5f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

        print(f"Fold {fold + 1} Best Validation Accuracy: {best_val_acc:.6f}")

        model.load_state_dict(best_state)
        model.eval()

        oof_fold = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=is_cuda):
                    logits = model(batch_x)
                    probs = torch.softmax(logits, dim=1)
                oof_fold.append(probs.cpu().numpy())
        oof_probs[val_idx] = np.concatenate(oof_fold, axis=0)

        test_fold = []
        with torch.no_grad():
            for (batch_x,) in test_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=is_cuda):
                    logits = model(batch_x)
                    probs = torch.softmax(logits, dim=1)
                test_fold.append(probs.cpu().numpy())
        test_probs += np.concatenate(test_fold, axis=0) / n_splits

        fold_time = (time.time() - fold_start) / 60.0
        print(f"Fold {fold + 1} completed in {fold_time:.2f} minutes.")

        del X_tr, X_va, X_te, train_dataset, val_dataset, test_dataset
        del train_loader, val_loader, test_loader, model, optimizer, scheduler
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

    oof_preds = np.argmax(oof_probs, axis=1) + 1
    overall_accuracy = accuracy_score(y_raw, oof_preds)
    print("\n==========================================")
    print(f"Overall OOF Validation Accuracy: {overall_accuracy:.6f}")
    print("==========================================")

    test_preds = np.argmax(test_probs, axis=1) + 1
    os.makedirs("./working", exist_ok=True)
    submission_path = "./working/submission.csv"

    sub = pd.DataFrame({"Id": test_ids, "Cover_Type": test_preds})
    sub.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path} with shape {sub.shape}")
    print(sub.head(10))

    elapsed_total = (time.time() - start_time) / 60.0
    print(f"Total pipeline execution time: {elapsed_total:.2f} minutes.")


if __name__ == "__main__":
    main()
