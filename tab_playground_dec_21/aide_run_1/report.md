# Technical Report: Forest Cover Type Prediction

## Introduction

This report outlines the development and empirical evaluation of machine learning pipelines for predicting forest cover types from 30m $\times$ 30m cartographic patches in the Roosevelt National Forest. The dataset consists of 4,000,000 training observations and 1,000,000 test observations evaluated on multi-class classification accuracy across seven forest cover categories.

All models were evaluated using a 5-fold stratified cross-validation (CV) scheme. Across iterations spanning gradient boosted decision trees (LightGBM, XGBoost) and deep tabular neural networks (Tabular ResNet), histogram-based XGBoost with extensive geomorphological feature engineering achieved the highest out-of-fold accuracy of **0.962401**.

---

## Preprocessing

### Data Cleaning and Stratification Integrity
- **Rare Class Pruning**: The target distribution contained a single occurrence of `Cover_Type = 5` within the 4-million-row training set. Because singletons prevent valid stratified $K$-fold partitioning, instances with fewer than 5 occurrences were pruned from the training set.
- **Zero-Variance Feature Removal**: Features with zero variance across all training samples (specifically `Soil_Type7` and `Soil_Type15`) were pruned to eliminate redundant model parameters.

### Domain-Specific Feature Engineering
1. **Hydrological and Elevation Dynamics**:
   - 3D Euclidean distance: $\sqrt{\text{Horizontal\_Distance\_To\_Hydrology}^2 + \text{Vertical\_Distance\_To\_Hydrology}^2}$
   - Manhattan flow distance: $|\text{Horizontal\_Distance\_To\_Hydrology}| + |\text{Vertical\_Distance\_To\_Hydrology}|$
   - Hydrological slope gradient: $\arctan2(\text{Vertical\_Distance}, \text{Horizontal\_Distance})$
   - Absolute hydrology elevations: $\text{Elevation} \pm \text{Vertical\_Distance\_To\_Hydrology}$
2. **Cartographic Landmark Interactions**:
   - Pairwise sums and absolute differences across horizontal distances to hydrology, roadways, and wildfire ignition points.
   - Aggregate landmark accessibility: total, minimum, and mean distances across all three infrastructure landmarks.
3. **Topography and Solar Exposure**:
   - Trigonometric cyclical aspect encoding: $\sin(\text{Aspect})$ and $\cos(\text{Aspect})$.
   - Topographic exposure interactions: Northness ($\cos(\text{Aspect}) \times \text{Slope}$) and Eastness ($\sin(\text{Aspect}) \times \text{Slope}$).
   - Multi-temporal hillshade statistics: mean, min, max, range, and diurnal differences ($9\text{am} - \text{Noon}$, $\text{Noon} - 3\text{pm}$, $9\text{am} - 3\text{pm}$).
4. **Ecological Land Unit (ELU) Decomposition**:
   - Row-wise active counts for wilderness areas and soil designations.
   - Argmax mapping of 40 sparse binary soil indicators into a single categorical feature index.
   - USFS ELU structural mapping: decomposition of soil index into physical **Climatic Zones** and **Geologic Zones**.

### Standardization
For deep learning pipelines, continuous and indicator features were normalized per fold using an in-place `StandardScaler` to maintain low peak memory footprints.

---

## Modellind Methods

### LightGBM
- **Architecture**: Leaf-wise tree growth with histogram binning (`max_bin=127`–$255$).
- **Hyperparameters**: Learning rate $\eta \in [0.12, 0.15]$, `num_leaves` $\in [63, 96]$, `max_depth` $\in [9, 10]$, `subsample` $= 0.8$, `colsample_bytree` $= 0.8$, `min_child_samples` $\in [50, 100]$.
- **Objective**: Multi-class cross-entropy (`multi_logloss`) with early stopping (30–35 rounds).

### XGBoost
- **Architecture**: Depth-wise gradient boosting using the GPU/CPU histogram tree method (`tree_method="hist"`).
- **Hyperparameters**: `max_depth` $= 10$ (GPU) / $8$ (CPU), $\eta = 0.10$–$0.12$, `subsample` $= 0.8$, `colsample_bytree` $\in [0.7, 0.8]$, `n_estimators` up to 1500 with early stopping rounds set to 30–40.
- **Objective**: Softmax probability estimation (`multi:softprob`).

### Tabular ResNet
- **Architecture**: Linear projection into hidden dimension ($d=256$), followed by 2 to 3 residual blocks (`Linear` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `SiLU` $\rightarrow$ `Dropout(0.15)` $\rightarrow$ `Linear` $\rightarrow$ `BatchNorm1d` with skip connection), followed by a classification head.
- **Training Optimization**: Standard PyTorch `DataLoader` was replaced with direct contiguous tensor indexing and in-place shuffling, mitigating Python-level collation overhead across 4M samples.
- **Optimization Strategy**: AdamW ($\text{weight\_decay}=10^{-4}$), initial learning rate $3 \times 10^{-3}$ to $4 \times 10^{-3}$, scheduled via Cosine Annealing or OneCycleLR over 8–10 epochs with batch size 8192.

---

## Results Discussion

### Empirical Comparison

| Model | Feature Set | 5-Fold CV Accuracy | Training & Inference Time |
| :--- | :--- | :---: | :---: |
| **LightGBM Baseline** | Core Spatial + Hillshade | 0.957972 | ~8.5 min |
| **LightGBM (Tuned)** | Histogram-binned ($max\_bin=127$) | 0.957489 | ~6.8 min |
| **Tabular ResNet (3-Block)** | Standardized Core Features | 0.960746 | ~28.0 min |
| **Fast Tabular ResNet (2-Block)** | Direct Tensor Slicing + Core Feats | 0.961626 | ~11.5 min |
| **XGBoost (Hist Baseline)** | Spatial + Landmark Combinations | **0.962401** | ~12.9 min |
| **XGBoost (ELU & Geomorph)** | Full Hydrology Slope + ELU Maps | **0.962401** | ~14.2 min |

### Key Findings
1. **Tree-Based Dominance**: XGBoost demonstrated the highest multi-class accuracy (0.962401), slightly outperforming Tabular ResNet (0.961626) and outperforming LightGBM (0.957972).
2. **Tabular Deep Learning Efficiency**: Standard PyTorch `DataLoader` configurations failed on this 4-million-row scale due to Python item indexing overhead. Direct GPU/CPU tensor slicing reduced per-epoch training durations by over $80\%$, enabling the 2-block Tabular ResNet to train in under 12 minutes while remaining competitive with tree models.
3. **Impact of Domain Features**: Relative elevation to hydrology, $\arctan2$ hydrological slope angles, and cyclical aspect decomposition improved convergence rates and boundary resolution between closely related cover types (e.g., Spruce/Fir vs. Lodgepole Pine).
4. **Fold Stability**: Validation performance showed minimal variance across all 5 folds (standard deviation $< 0.0003$), confirming pipeline stability on large-scale cartographic distributions.

---

## Future Work

- **Ensemble Blending**: Combining out-of-fold probability distributions from the best XGBoost and Tabular ResNet models via weighted Nelder-Mead optimization or stacking meta-learners.
- **Categorical Embeddings**: Implementing trainable entity embeddings for the 40 ELU soil classifications within the neural architecture rather than static scalar mapping.
- **Test-Set Pseudo-Labeling**: Exploiting high-confidence predictions on the 1,000,000-instance test partition to perform semi-supervised self-training.
- **Quantile Hydrology Binning**: Introducing non-linear spatial clustering based on nearest elevation contour curves to model complex drainage basins.