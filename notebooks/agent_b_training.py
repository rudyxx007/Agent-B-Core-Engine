# %% [markdown]
# # Agent-B: iTransformer Training Notebook
# 
# **What this does:** Trains an iTransformer model on Brent Crude Oil data to predict
# price quantiles (10th/50th/90th percentile) across 1-Day, 1-Month, and 3-Month horizons.
#
# **Requirements:**
# - Kaggle GPU (T4 x2) 
# - Dataset: `agent-b-training-data` (your uploaded CSV)
# - Secret: `HF_TOKEN` (your Hugging Face write token)
#
# **Runtime:** ~1-2 hours on Kaggle GPU

# %% [markdown]
# ## Cell 1: Install Dependencies & Setup

# %%
import subprocess
import sys

# Install required packages not available on Kaggle by default
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "huggingface_hub"])

import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## Cell 2: Load Data

# %%
# === CONFIGURATION ===
# Adjust this path if your Kaggle dataset has a different name
KAGGLE_DATA_PATHS = [
    "/kaggle/input/agent-b-training-data/historical_features.csv",
    "/kaggle/input/agent-b-training-data/data/historical_features.csv",
    "data/historical_features.csv",  # Local fallback
]

# Find the CSV
DATA_PATH = None
for p in KAGGLE_DATA_PATHS:
    if os.path.exists(p):
        DATA_PATH = p
        break

if DATA_PATH is None:
    # List what's actually in the input directory
    input_dir = "/kaggle/input"
    if os.path.exists(input_dir):
        print("Available datasets in /kaggle/input/:")
        for d in os.listdir(input_dir):
            full = os.path.join(input_dir, d)
            print(f"  {d}/")
            if os.path.isdir(full):
                for f in os.listdir(full)[:10]:
                    print(f"    {f}")
    raise FileNotFoundError("Cannot find historical_features.csv. Check your dataset name.")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Columns: {list(df.columns)}")
df.head()

# %% [markdown]
# ## Cell 3: Define Model Features & Targets

# %%
# === FEATURES (model inputs) ===
# Per spec Section 3.1: [Z_Score_Brent, Z_Score_DXY, Holiday_Flag, Sentiment_Score]
# We add technical indicators for richer signal
FEATURE_COLS = [
    "z_brent",           # Z-score normalized Brent price
    "z_dxy",             # Z-score normalized DXY
    "holiday_flag",      # Binary: is today a structural holiday?
    "sentiment_score",   # Normalized sentiment [-1, +1]
    "rsi_14",            # RSI momentum indicator
    "macd",              # MACD trend indicator
    "macd_signal",       # MACD signal line
]

# === TARGETS (model outputs) ===
# We predict future brent_close at different horizons
# The actual quantile splitting is handled by the loss function
TARGET_COL = "brent_close"       # Raw price target for quantile regression
VOLATILITY_COL = "volatility_spread"  # High - Low spread
MA_CROSS_COL = "ma_crossover"        # Binary MA crossover

# === HORIZONS ===
HORIZON_1D = 1     # 1 trading day ahead
HORIZON_1M = 21    # ~1 month (21 trading days)
HORIZON_3M = 63    # ~3 months (63 trading days)
HORIZONS = [HORIZON_1D, HORIZON_1M, HORIZON_3M]
HORIZON_NAMES = ["1D", "1M", "3M"]

# === HYPERPARAMETERS ===
SEQ_LEN = 90         # 90-day lookback (per spec Section 3.1)
D_MODEL = 128        # Embedding dimension
N_HEADS = 8          # Attention heads
E_LAYERS = 3         # Encoder layers
D_FF = 256           # Feed-forward dimension
DROPOUT = 0.1        # Dropout rate
QUANTILES = [0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles
N_QUANTILES = len(QUANTILES)
N_HORIZONS = len(HORIZONS)

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50          # Per fold
PATIENCE = 10        # Early stopping patience

# Walk-Forward parameters
TRAIN_YEARS = 10     # Per spec: rolling 10-year training window
TEST_YEARS = 1       # Per spec: 1-year test window

N_FEATURES = len(FEATURE_COLS)
print(f"Features ({N_FEATURES}): {FEATURE_COLS}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Horizons: {dict(zip(HORIZON_NAMES, HORIZONS))}")
print(f"Quantiles: {QUANTILES}")
print(f"Model: d_model={D_MODEL}, heads={N_HEADS}, layers={E_LAYERS}")

# %% [markdown]
# ## Cell 4: Prepare Target Columns

# %%
# Create future target columns for each horizon
for h, name in zip(HORIZONS, HORIZON_NAMES):
    df[f"target_{name}"] = df[TARGET_COL].shift(-h)
    df[f"target_vol_{name}"] = df[VOLATILITY_COL].shift(-h)

# MA crossover target (next day)
df["target_ma"] = df[MA_CROSS_COL].shift(-1)

# Drop rows where targets are NaN (last rows that don't have future data)
max_horizon = max(HORIZONS)
df_clean = df.iloc[:-(max_horizon)].copy()
df_clean = df_clean.dropna(subset=[f"target_{name}" for name in HORIZON_NAMES])
df_clean = df_clean.dropna(subset=FEATURE_COLS)

print(f"Rows after target creation: {len(df_clean)} (dropped last {max_horizon} rows)")
print(f"Date range: {df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}")

# %% [markdown]
# ## Cell 5: Dataset Class

# %%
class BrentDataset(Dataset):
    """
    Creates (input_sequence, targets) pairs for the iTransformer.
    
    Input:  [SEQ_LEN, N_FEATURES] - 90 days of feature history
    Output: Dictionary with quantile targets for each horizon
    """
    def __init__(self, dataframe, feature_cols, seq_len=90):
        self.data = dataframe.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        
        # Pre-extract arrays for speed
        self.features = self.data[feature_cols].values.astype(np.float32)
        
        # Targets for each horizon
        self.targets_1d = self.data["target_1D"].values.astype(np.float32)
        self.targets_1m = self.data["target_1M"].values.astype(np.float32)
        self.targets_3m = self.data["target_3M"].values.astype(np.float32)
        
        # Volatility targets
        self.vol_1d = self.data["target_vol_1D"].values.astype(np.float32)
        
        # MA crossover target
        self.ma_target = self.data["target_ma"].values.astype(np.float32)
        
        # Valid indices (need SEQ_LEN history)
        self.valid_start = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        actual_idx = idx + self.seq_len
        
        # Input sequence: [SEQ_LEN, N_FEATURES]
        x = self.features[idx:actual_idx]
        
        # Targets
        targets = {
            "target_1D": self.targets_1d[actual_idx],
            "target_1M": self.targets_1m[actual_idx],
            "target_3M": self.targets_3m[actual_idx],
            "vol_1D": self.vol_1d[actual_idx],
            "ma_cross": self.ma_target[actual_idx],
        }
        
        return torch.FloatTensor(x), targets

# Quick test
test_ds = BrentDataset(df_clean.iloc[:200], FEATURE_COLS, SEQ_LEN)
x, y = test_ds[0]
print(f"Input shape: {x.shape} (expected: [{SEQ_LEN}, {N_FEATURES}])")
print(f"Targets: { {k: f'{v:.2f}' for k, v in y.items()} }")

# %% [markdown]
# ## Cell 6: iTransformer Architecture
# 
# The key insight of iTransformer: instead of treating time steps as tokens 
# (standard Transformer), we treat each **feature/variate** as a token.
# This lets attention capture cross-variate correlations while FFN learns 
# per-variate temporal patterns.

# %%
class InvertedEmbedding(nn.Module):
    """
    Embeds each variate's full time series into d_model.
    Input:  [B, T, N] (batch, time_steps, num_variates)
    Output: [B, N, D] (batch, num_variates, d_model)
    
    Each variate's T-length time series is projected to d_model via a linear layer.
    """
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
    
    def forward(self, x):
        # x: [B, T, N] -> transpose to [B, N, T] -> project to [B, N, D]
        x = x.permute(0, 2, 1)  # [B, N, T]
        x = self.proj(x)         # [B, N, D]
        return x


class FullAttention(nn.Module):
    """Standard scaled dot-product multi-head attention."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, D = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_o(out)
        
        return out


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer: Attention + FFN + LayerNorm."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = FullAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attention(self.norm1(x))
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class iTransformerForecaster(nn.Module):
    """
    iTransformer for multi-horizon quantile forecasting.
    
    Architecture:
    1. Invert: [B, T, N_features] -> [B, N_features, d_model]
    2. Encode: Stack of Transformer layers (attention across features)
    3. Project: Separate heads for quantiles, volatility, MA crossover
    
    Output: 
    - quantile_preds: [B, N_horizons, N_quantiles] = [B, 3, 3]
    - volatility_pred: [B, 1]
    - ma_cross_pred: [B, 1]
    """
    def __init__(self, seq_len, n_features, d_model, n_heads, e_layers, 
                 d_ff, dropout, n_horizons, n_quantiles):
        super().__init__()
        
        self.n_features = n_features
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        
        # 1. Inverted embedding
        self.embedding = InvertedEmbedding(seq_len, d_model)
        
        # 2. Encoder stack
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])
        
        # 3. Output projection heads
        # Pool features -> single representation, then project to outputs
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool across features
        
        # Quantile head: predicts [n_horizons * n_quantiles] values
        self.quantile_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_horizons * n_quantiles),
        )
        
        # Volatility head: predicts 1 value (predicted High-Low spread)
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # MA crossover head: predicts 1 value (binary classification)
        self.ma_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        
    def forward(self, x):
        """
        x: [B, T, N_features] - input sequence
        Returns dict of predictions
        """
        # Inverted embedding: [B, T, N] -> [B, N, D]
        h = self.embedding(x)
        
        # Encoder: stack of attention layers
        for layer in self.encoder:
            h = layer(h)
        
        # Pool across features: [B, N, D] -> [B, D]
        h = h.permute(0, 2, 1)  # [B, D, N]
        h = self.pool(h).squeeze(-1)  # [B, D]
        
        # Quantile predictions: [B, n_horizons * n_quantiles]
        q_raw = self.quantile_head(h)
        q_preds = q_raw.view(-1, self.n_horizons, self.n_quantiles)
        
        # Volatility prediction
        vol_pred = self.volatility_head(h)
        
        # MA crossover prediction (logit)
        ma_pred = self.ma_head(h)
        
        return {
            "quantiles": q_preds,       # [B, 3, 3]
            "volatility": vol_pred,      # [B, 1]
            "ma_logit": ma_pred,         # [B, 1]
        }

# Test the model
model_test = iTransformerForecaster(
    seq_len=SEQ_LEN, n_features=N_FEATURES, d_model=D_MODEL,
    n_heads=N_HEADS, e_layers=E_LAYERS, d_ff=D_FF, dropout=DROPOUT,
    n_horizons=N_HORIZONS, n_quantiles=N_QUANTILES,
).to(device)

# Count parameters
n_params = sum(p.numel() for p in model_test.parameters())
print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

# Forward pass test
with torch.no_grad():
    test_input = torch.randn(2, SEQ_LEN, N_FEATURES).to(device)
    test_out = model_test(test_input)
    print(f"Quantile output shape: {test_out['quantiles'].shape} (expected: [2, 3, 3])")
    print(f"Volatility output shape: {test_out['volatility'].shape} (expected: [2, 1])")
    print(f"MA crossover output shape: {test_out['ma_logit'].shape} (expected: [2, 1])")

del model_test
print("Model architecture verified.")

# %% [markdown]
# ## Cell 7: Loss Functions
#
# Per spec Section 3.2: Pinball Loss (Quantile Regression) replaces MSE.
# We also add an anti-crossing penalty to ensure Q10 < Q50 < Q90.

# %%
class PinballLoss(nn.Module):
    """
    Quantile (Pinball) Loss for quantile regression.
    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, targets):
        """
        preds: [B, N_quantiles] - predicted quantiles
        targets: [B] - actual values
        """
        targets = targets.unsqueeze(1).expand_as(preds)  # [B, N_quantiles]
        errors = targets - preds
        
        losses = []
        for i, q in enumerate(self.quantiles):
            loss_q = torch.max(q * errors[:, i], (q - 1) * errors[:, i])
            losses.append(loss_q.mean())
        
        return sum(losses) / len(losses)


class AntiCrossingPenalty(nn.Module):
    """
    Penalty to enforce quantile ordering: Q10 < Q50 < Q90.
    If quantiles cross, adds a penalty proportional to the violation.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, preds):
        """preds: [B, N_quantiles] where columns are ordered quantiles"""
        penalty = 0.0
        for i in range(preds.shape[1] - 1):
            # Penalize when lower quantile > higher quantile
            violations = F.relu(preds[:, i] - preds[:, i + 1])
            penalty += violations.mean()
        return self.weight * penalty


class CompositeLoss(nn.Module):
    """
    Combined loss for Agent-B:
    - Pinball Loss for each horizon's quantile predictions
    - MSE for volatility prediction
    - BCE for MA crossover prediction
    - Anti-crossing penalty for quantile ordering
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9], crossing_weight=1.0):
        super().__init__()
        self.pinball = PinballLoss(quantiles)
        self.anti_crossing = AntiCrossingPenalty(crossing_weight)
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        """
        predictions: dict from model forward pass
        targets: dict from dataset
        """
        q_preds = predictions["quantiles"]  # [B, 3, 3]
        vol_pred = predictions["volatility"]  # [B, 1]
        ma_pred = predictions["ma_logit"]  # [B, 1]
        
        # Pinball loss for each horizon
        horizon_names = ["1D", "1M", "3M"]
        pinball_loss = 0.0
        crossing_loss = 0.0
        
        for h_idx, h_name in enumerate(horizon_names):
            q_h = q_preds[:, h_idx, :]  # [B, 3]
            t_h = targets[f"target_{h_name}"]  # [B]
            
            pinball_loss += self.pinball(q_h, t_h)
            crossing_loss += self.anti_crossing(q_h)
        
        pinball_loss /= len(horizon_names)
        crossing_loss /= len(horizon_names)
        
        # Volatility loss (MSE)
        vol_loss = self.mse(vol_pred.squeeze(-1), targets["vol_1D"])
        
        # MA crossover loss (BCE)
        ma_loss = self.bce(ma_pred.squeeze(-1), targets["ma_cross"])
        
        # Weighted total
        total = pinball_loss + 0.5 * crossing_loss + 0.1 * vol_loss + 0.1 * ma_loss
        
        return total, {
            "pinball": pinball_loss.item(),
            "crossing": crossing_loss.item(),
            "volatility": vol_loss.item(),
            "ma_cross": ma_loss.item(),
            "total": total.item(),
        }

print("Loss functions defined.")

# %% [markdown]
# ## Cell 8: Walk-Forward Validation Setup

# %%
def create_walk_forward_folds(df, train_years=10, test_years=1):
    """
    Create Walk-Forward validation folds.
    
    Per spec: rolling 10-year train window, 1-year test window.
    Each fold slides forward by 1 year.
    
    Returns list of (train_df, test_df) tuples.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    
    min_year = df["year"].min()
    max_year = df["year"].max()
    
    folds = []
    
    # First fold starts at min_year, trains for train_years, tests on next year
    for test_start_year in range(min_year + train_years, max_year + 1):
        train_start_year = test_start_year - train_years
        test_end_year = test_start_year + test_years - 1
        
        train_mask = (df["year"] >= train_start_year) & (df["year"] < test_start_year)
        test_mask = (df["year"] >= test_start_year) & (df["year"] <= test_end_year)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        if len(train_df) > SEQ_LEN and len(test_df) > 0:
            folds.append({
                "train": train_df,
                "test": test_df,
                "train_years": f"{train_start_year}-{test_start_year-1}",
                "test_year": str(test_start_year),
            })
    
    return folds

folds = create_walk_forward_folds(df_clean, TRAIN_YEARS, TEST_YEARS)
print(f"Created {len(folds)} Walk-Forward folds:")
for i, fold in enumerate(folds):
    print(f"  Fold {i+1}: Train {fold['train_years']} ({len(fold['train'])} rows) | Test {fold['test_year']} ({len(fold['test'])} rows)")

# %% [markdown]
# ## Cell 9: Training Loop

# %%
def train_one_fold(model, train_df, test_df, fold_name, epochs=EPOCHS, patience=PATIENCE):
    """Train model on one Walk-Forward fold with early stopping."""
    
    train_ds = BrentDataset(train_df, FEATURE_COLS, SEQ_LEN)
    test_ds = BrentDataset(test_df, FEATURE_COLS, SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=0, drop_last=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CompositeLoss(QUANTILES, crossing_weight=1.0)
    
    best_test_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "test_loss": []}
    
    for epoch in range(epochs):
        # === TRAIN ===
        model.train()
        train_losses = []
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            
            optimizer.zero_grad()
            preds = model(x_batch)
            loss, loss_dict = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss_dict["total"])
        
        scheduler.step()
        avg_train = np.mean(train_losses)
        
        # === EVALUATE ===
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = {k: v.to(device) for k, v in y_batch.items()}
                
                preds = model(x_batch)
                loss, loss_dict = criterion(preds, y_batch)
                test_losses.append(loss_dict["total"])
        
        avg_test = np.mean(test_losses)
        history["train_loss"].append(avg_train)
        history["test_loss"].append(avg_test)
        
        # Early stopping
        if avg_test < best_test_loss:
            best_test_loss = avg_test
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train:.4f} | Test: {avg_test:.4f} | "
                  f"Best: {best_test_loss:.4f} | Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_test_loss, history

print("Training function defined.")

# %% [markdown]
# ## Cell 10: Run Walk-Forward Training

# %%
print("=" * 60)
print("  AGENT-B: Walk-Forward Training")
print(f"  {len(folds)} folds, {EPOCHS} max epochs each")
print("=" * 60)

fold_results = []

for i, fold in enumerate(folds):
    print(f"\n{'='*60}")
    print(f"  Fold {i+1}/{len(folds)}: Train {fold['train_years']} | Test {fold['test_year']}")
    print(f"  Train: {len(fold['train'])} rows | Test: {len(fold['test'])} rows")
    print(f"{'='*60}")
    
    # Create fresh model for each fold
    model = iTransformerForecaster(
        seq_len=SEQ_LEN, n_features=N_FEATURES, d_model=D_MODEL,
        n_heads=N_HEADS, e_layers=E_LAYERS, d_ff=D_FF, dropout=DROPOUT,
        n_horizons=N_HORIZONS, n_quantiles=N_QUANTILES,
    ).to(device)
    
    model, best_loss, history = train_one_fold(
        model, fold["train"], fold["test"], 
        fold_name=f"fold_{i+1}"
    )
    
    fold_results.append({
        "fold": i + 1,
        "train_years": fold["train_years"],
        "test_year": fold["test_year"],
        "best_loss": best_loss,
        "final_train_loss": history["train_loss"][-1],
    })
    
    print(f"  Fold {i+1} complete. Best test loss: {best_loss:.4f}")

# Print summary
print("\n" + "=" * 60)
print("  WALK-FORWARD RESULTS SUMMARY")
print("=" * 60)
for r in fold_results:
    print(f"  Fold {r['fold']}: Train {r['train_years']} | Test {r['test_year']} | "
          f"Loss: {r['best_loss']:.4f}")

avg_loss = np.mean([r["best_loss"] for r in fold_results])
print(f"\n  Average test loss across folds: {avg_loss:.4f}")

# %% [markdown]
# ## Cell 11: Final Training (Full Available Data)
#
# After validation, train the production model on ALL available data
# using the most recent window.

# %%
print("=" * 60)
print("  FINAL TRAINING: Production Model")
print("  Training on most recent data for deployment")
print("=" * 60)

# Use last TRAIN_YEARS of data for final training
final_df = df_clean.copy()
final_year_cutoff = final_df["date"].dt.year.max() - TRAIN_YEARS
final_train = final_df[final_df["date"].dt.year >= final_year_cutoff].copy()

print(f"Final training data: {len(final_train)} rows")
print(f"Date range: {final_train['date'].min().strftime('%Y-%m-%d')} to "
      f"{final_train['date'].max().strftime('%Y-%m-%d')}")

# Create production model
production_model = iTransformerForecaster(
    seq_len=SEQ_LEN, n_features=N_FEATURES, d_model=D_MODEL,
    n_heads=N_HEADS, e_layers=E_LAYERS, d_ff=D_FF, dropout=DROPOUT,
    n_horizons=N_HORIZONS, n_quantiles=N_QUANTILES,
).to(device)

# Train with slightly more epochs for production
# Use last 10% as validation
split_idx = int(len(final_train) * 0.9)
train_part = final_train.iloc[:split_idx]
val_part = final_train.iloc[split_idx:]

production_model, prod_loss, prod_history = train_one_fold(
    production_model, train_part, val_part,
    fold_name="production", epochs=EPOCHS * 2, patience=15,
)

print(f"\nProduction model trained. Best validation loss: {prod_loss:.4f}")

# %% [markdown]
# ## Cell 12: Save Model & Upload to Hugging Face

# %%
# === Save model locally ===
SAVE_DIR = Path("./agent_b_model")
SAVE_DIR.mkdir(exist_ok=True)

# Model weights
torch.save(production_model.state_dict(), SAVE_DIR / "model.pth")

# Model config (needed to reconstruct the model for inference)
config = {
    "model_type": "iTransformer",
    "seq_len": SEQ_LEN,
    "n_features": N_FEATURES,
    "feature_cols": FEATURE_COLS,
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "e_layers": E_LAYERS,
    "d_ff": D_FF,
    "dropout": DROPOUT,
    "n_horizons": N_HORIZONS,
    "n_quantiles": N_QUANTILES,
    "quantiles": QUANTILES,
    "horizons": dict(zip(HORIZON_NAMES, HORIZONS)),
    "train_date_range": {
        "start": final_train["date"].min().strftime("%Y-%m-%d"),
        "end": final_train["date"].max().strftime("%Y-%m-%d"),
    },
    "walk_forward_results": fold_results,
    "production_loss": prod_loss,
    "trained_at": datetime.now().isoformat(),
}

with open(SAVE_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2, default=str)

print(f"Model saved to {SAVE_DIR}/")
print(f"  model.pth: {(SAVE_DIR / 'model.pth').stat().st_size / 1e6:.2f} MB")
print(f"  config.json: saved")

# === Upload to Hugging Face ===
print("\nUploading to Hugging Face Hub...")

try:
    from huggingface_hub import HfApi, login
    from kaggle_secrets import UserSecretsClient
    
    # Get HF token from Kaggle Secrets
    try:
        secrets = UserSecretsClient()
        hf_token = secrets.get_secret("HF_TOKEN")
    except Exception:
        hf_token = os.environ.get("HF_TOKEN", "")
    
    if not hf_token:
        print("WARNING: No HF_TOKEN found. Skipping upload.")
        print("You can manually upload the files from ./agent_b_model/")
    else:
        login(token=hf_token)
        api = HfApi()
        
        # Create or get repo
        repo_id = api.whoami()["name"] + "/agent-b-itransformer"
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        except Exception as e:
            print(f"Repo creation note: {e}")
        
        # Upload files
        api.upload_folder(
            folder_path=str(SAVE_DIR),
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"\n[OK] Model uploaded to: https://huggingface.co/{repo_id}")
        print("Files uploaded: model.pth, config.json")
        
except Exception as e:
    print(f"Upload failed: {e}")
    print("Model is saved locally at ./agent_b_model/")
    print("You can manually upload to Hugging Face later.")

# %% [markdown]
# ## Cell 13: Summary
# 
# Training complete! Here's what was produced:
# - **model.pth** - Trained iTransformer weights
# - **config.json** - Model architecture config + training metadata
# 
# Next steps:
# 1. Verify the model on Hugging Face
# 2. Move to Phase 3: Modal daily inference pipeline

# %%
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  Model: iTransformer ({sum(p.numel() for p in production_model.parameters()):,} params)")
print(f"  Walk-Forward folds: {len(fold_results)}")
print(f"  Avg test loss: {avg_loss:.4f}")
print(f"  Production loss: {prod_loss:.4f}")
print(f"  Features: {FEATURE_COLS}")
print(f"  Horizons: 1-Day, 1-Month, 3-Month")
print(f"  Quantiles: 10th, 50th, 90th percentile")
print("=" * 60)
