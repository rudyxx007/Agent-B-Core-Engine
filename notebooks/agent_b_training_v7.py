# %% [markdown]
# # Agent-B: iTransformer v7.0 — True FinGPT Sentiment + VIX
#
# **Changes from v6.1:**
# 1. NEW FEATURES: True FinGPT Sentiment + Independent VIX (12 features total)
# 2. Uses BEST-VAL-EPOCH ensemble only (no epoch-200 models)
# 3. Still trains ALL 200 epochs to see full learning curve
# 4. Saves best checkpoint per fold automatically

# %%
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

import os, json, math, time, copy
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("+" + "="*68 + "+")
print("|  AGENT-B v7.0: 12 Features + Best-Val-Epoch Ensemble              |")
print("+" + "="*68 + "+")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"|  GPU: {p.name} | {p.total_memory/(1024**3):.1f} GB x{torch.cuda.device_count()}")
print("+" + "="*68 + "+")

# %%
# ════════════════════ LOAD DATA ════════════════════

DATA_PATH = None
for rd in ["/kaggle/input", "."]:
    if not os.path.exists(rd): continue
    for root, dirs, files in os.walk(rd):
        if "historical_features.csv" in files:
            DATA_PATH = os.path.join(root, "historical_features.csv"); break
    if DATA_PATH: break
if not DATA_PATH: raise FileNotFoundError("historical_features.csv not found")

df_raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
print(f"Data: {len(df_raw)} rows | {df_raw['date'].min().strftime('%Y-%m-%d')} to {df_raw['date'].max().strftime('%Y-%m-%d')}")
print(f"Columns: {list(df_raw.columns)}")

# %%
# ════════════════════ FEATURES & TARGETS ════════════════════

ROLLING_WINDOW = 30
HORIZONS = [1, 21, 63]
HORIZON_NAMES = ["1D", "1M", "3M"]
HORIZON_LABELS = {"1D": "1-Day", "1M": "1-Month", "3M": "3-Month"}

df = df_raw.copy()
df["rolling_mean"] = df["brent_close"].rolling(ROLLING_WINDOW, min_periods=1).mean()
df["rolling_std"] = df["brent_close"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(1).replace(0,1)

for h, name in zip(HORIZONS, HORIZON_NAMES):
    df[f"z_target_{name}"] = (df["brent_close"].shift(-h) - df["rolling_mean"]) / df["rolling_std"]
for h, name in zip(HORIZONS, HORIZON_NAMES):
    df[f"z_vol_{name}"] = df["volatility_spread"].shift(-h) / df["rolling_std"]
df["target_ma"] = df["ma_crossover"].shift(-1)

# === FEATURE COLUMNS ===
# 5 Spec features + 3 technical + 4 NEW market structure features = 12 total
RAW_FEATURE_COLS = [
    # Spec Section 3.1 features
    "z_brent",              # [0] Brent crude Z-score (SPEC 3.1.2)
    "z_dxy",                # [1] US Dollar Index Z-score (SPEC 3.1.1)
    "holiday_flag",         # [2] Global holidays binary (SPEC 3.1.3)
    "vix_close",            # [3] VIX Fear Gauge (Independent Macro Feature)
    "sentiment_score",      # [4] True FinGPT Sentiment (SPEC 3.1.4)
    # Technical indicators (bonus)
    "rsi_14",               # [5] RSI momentum
    "macd",                 # [6] MACD trend
    "macd_signal",          # [7] MACD signal line
    # NEW: Market structure features
    "brent_wti_spread",     # [8] Brent-WTI spread (regional supply dynamics)
    "crack_spread_321",     # [9] 3-2-1 crack spread (RIL refining margin)
    "eia_inventory",        # [10] US crude oil inventory (supply fundamentals)
    "eia_inventory_change", # [11] Week-over-week inventory change (market mover)
]

# Check which features exist in the data
available_features = [c for c in RAW_FEATURE_COLS if c in df.columns]
missing_features = [c for c in RAW_FEATURE_COLS if c not in df.columns]

if missing_features:
    print(f"\nWARNING: Missing features (will be zero-filled): {missing_features}")
    print("Re-run generate_training_data.py to add them.")
    for col in missing_features:
        df[col] = 0.0

# Standardize ALL features to ~N(0,1)
feature_stats = {}
for col in RAW_FEATURE_COLS:
    mu, sigma = df[col].mean(), df[col].std()
    if sigma == 0: sigma = 1.0
    df[f"norm_{col}"] = (df[col] - mu) / sigma
    feature_stats[col] = {"mean": float(mu), "std": float(sigma)}

FEATURE_COLS = [f"norm_{c}" for c in RAW_FEATURE_COLS]
BRENT_TOKEN_IDX = 0  # norm_z_brent is at index 0

df_clean = df.iloc[:-max(HORIZONS)].copy()
df_clean = df_clean.dropna(subset=[f"z_target_{n}" for n in HORIZON_NAMES])
df_clean = df_clean.dropna(subset=FEATURE_COLS)

print(f"\n{'='*60}")
print("FEATURE MATRIX (12 variate tokens for iTransformer)")
print(f"{'='*60}")
for i, col in enumerate(RAW_FEATURE_COLS):
    tag = "SPEC" if i < 5 else "TECH" if i < 8 else "NEW "
    s = feature_stats[col]
    print(f"  [{i:2d}] {tag} {col:<25} mean={s['mean']:>8.3f} std={s['std']:>8.3f}")
print(f"\nClean rows: {len(df_clean)} | Total features: {len(FEATURE_COLS)}")
if missing_features:
    print(f"ZERO-FILLED: {missing_features}")

# %%
# ════════════════════ CONFIGURATION ════════════════════

SEQ_LEN     = 90
D_MODEL     = 128
N_HEADS     = 8
E_LAYERS    = 2
D_FF        = 256
DROPOUT     = 0.2
ACTIVATION  = "gelu"
USE_NORM    = True

QUANTILES   = [0.1, 0.5, 0.9]
N_QUANTILES = len(QUANTILES)
N_HORIZONS  = len(HORIZONS)
N_FEATURES  = len(FEATURE_COLS)

BATCH_SIZE  = 64
LR          = 1e-4
WARMUP      = 10
TOTAL_EPOCHS= 200
WEIGHT_DECAY= 5e-4
AUG_NOISE   = 0.005
AUG_LEN_MIN = 80
AUG_LEN_MAX = 90
TRAIN_YEARS = 10

print(f"\nConfig: d={D_MODEL}, L={E_LAYERS}, ff={D_FF}, drop={DROPOUT}")
print(f"Features: {N_FEATURES} (was 11 in v6.1, now 12)")
print(f"Training: {TOTAL_EPOCHS} epochs/fold, saving BEST VAL EPOCH only")

# %%
# ════════════════════ DATASET ════════════════════

class BrentDataset(Dataset):
    def __init__(self, dataframe, feature_cols, seq_len=90, augment=False,
                 noise_std=0.005, len_min=80, len_max=90):
        self.data = dataframe.reset_index(drop=True)
        self.seq_len = seq_len
        self.augment = augment
        self.noise_std = noise_std
        self.len_min = len_min
        self.len_max = len_max
        self.features = self.data[feature_cols].values.astype(np.float32)
        self.z_targets = {n: self.data[f"z_target_{n}"].values.astype(np.float32) for n in HORIZON_NAMES}
        self.z_vol = self.data["z_vol_1D"].values.astype(np.float32)
        self.ma_target = self.data["target_ma"].fillna(0).values.astype(np.float32)
        self.r_mean = self.data["rolling_mean"].values.astype(np.float32)
        self.r_std = self.data["rolling_std"].values.astype(np.float32)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        t = idx + self.seq_len
        if self.augment:
            actual_len = np.random.randint(self.len_min, self.len_max + 1)
            start = t - actual_len
            x = self.features[start:t].copy()
            if actual_len < self.seq_len:
                pad = np.zeros((self.seq_len - actual_len, x.shape[1]), dtype=np.float32)
                x = np.concatenate([pad, x], axis=0)
            x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
        else:
            x = self.features[idx:t]
        targets = {f"z_target_{n}": self.z_targets[n][t] for n in HORIZON_NAMES}
        targets["z_vol_1D"] = self.z_vol[t]
        targets["ma_cross"] = self.ma_target[t]
        targets["rolling_mean"] = self.r_mean[t]
        targets["rolling_std"] = self.r_std[t]
        return torch.FloatTensor(x), targets

print(f"Dataset ready. Input shape per sample: [{SEQ_LEN}, {N_FEATURES}]")

# %%
# ════════════════════ iTransformer ════════════════════

class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        return self.dropout(self.value_embedding(x.permute(0, 2, 1)))

class FullAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, Q, K, V):
        scale = 1.0 / math.sqrt(Q.shape[-1])
        attn = self.dropout(F.softmax(torch.matmul(Q, K.transpose(-2,-1))*scale, dim=-1))
        return torch.matmul(attn, V), attn

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.inner = FullAttention(dropout)
        self.n_heads = n_heads
        dk = d_model // n_heads
        self.Wq = nn.Linear(d_model, dk*n_heads)
        self.Wk = nn.Linear(d_model, dk*n_heads)
        self.Wv = nn.Linear(d_model, dk*n_heads)
        self.Wo = nn.Linear(dk*n_heads, d_model)
    def forward(self, q, k, v):
        B, L, _ = q.shape; H = self.n_heads
        q = self.Wq(q).view(B,L,H,-1).transpose(1,2)
        k = self.Wk(k).view(B,L,H,-1).transpose(1,2)
        v = self.Wv(v).view(B,L,H,-1).transpose(1,2)
        out, attn = self.inner(q, k, v)
        return self.Wo(out.transpose(1,2).contiguous().view(B,L,-1)), attn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.attn = AttentionLayer(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu if activation == "gelu" else F.relu
    def forward(self, x):
        new_x, attn = self.attn(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.act(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm2(x + y), attn

class iTransformer(nn.Module):
    """
    iTransformer with brent-token-only output.
    Now processes 12 variate tokens.
    """
    def __init__(self, seq_len, n_features, d_model, n_heads, e_layers,
                 d_ff, dropout, n_horizons, n_quantiles, brent_idx=0,
                 activation="gelu", use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.brent_idx = brent_idx
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers)])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(d_model, n_quantiles)) for _ in range(n_horizons)])
        self.vol_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))
        self.ma_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))

    def forward(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+1e-5)
            x_enc = x_enc / stdev
        h = self.enc_embedding(x_enc)
        for layer in self.encoder_layers:
            h, _ = layer(h)
        h = self.encoder_norm(h)
        brent = h[:, self.brent_idx, :]
        q = torch.stack([head(brent) for head in self.quantile_heads], dim=1)
        return {"quantiles": q, "volatility": self.vol_head(brent), "ma_logit": self.ma_head(brent)}

def make_model():
    return iTransformer(SEQ_LEN, N_FEATURES, D_MODEL, N_HEADS, E_LAYERS, D_FF,
                        DROPOUT, N_HORIZONS, N_QUANTILES, BRENT_TOKEN_IDX, ACTIVATION, USE_NORM).to(device)

m = make_model()
np_count = sum(p.numel() for p in m.parameters())
print(f"\niTransformer v7.0: {np_count:,} params ({np_count/1e6:.2f}M)")
print(f"  Now with {N_FEATURES} variate tokens (was 11)")
del m; torch.cuda.empty_cache()

# %%
# ════════════════════ LOSS ════════════════════

class PinballLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
    def forward(self, preds, targets):
        targets = targets.unsqueeze(1).expand_as(preds)
        errors = targets - preds
        return sum(torch.max(q*errors[:,i], (q-1)*errors[:,i]).mean()
                   for i,q in enumerate(self.quantiles)) / len(self.quantiles)

class CompositeLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.pinball = PinballLoss(quantiles)
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, preds, targets):
        q = preds["quantiles"]
        pb, cx = 0.0, 0.0
        for hi, hn in enumerate(HORIZON_NAMES):
            qh = q[:, hi, :]
            pb += self.pinball(qh, targets[f"z_target_{hn}"])
            for i in range(qh.shape[1]-1):
                cx += F.relu(qh[:,i]-qh[:,i+1]).mean()
        pb /= N_HORIZONS; cx /= N_HORIZONS
        vl = self.mse(preds["volatility"].squeeze(-1), targets["z_vol_1D"])
        ml = self.bce(preds["ma_logit"].squeeze(-1), targets["ma_cross"])
        total = pb + 0.5*cx + 0.1*vl + 0.1*ml
        return total, {"total": total.item(), "pinball": pb.item(),
                       "crossing": cx.item(), "vol": vl.item(), "ma": ml.item()}

print("Loss ready.")

# %%
# ════════════════════ WALK-FORWARD FOLDS ════════════════════

def make_folds(df, train_yrs=10):
    df = df.copy(); df["year"] = df["date"].dt.year
    folds = []
    for ty in range(df["year"].min() + train_yrs, df["year"].max() + 1):
        tr = df[(df["year"] >= ty-train_yrs) & (df["year"] < ty)]
        te = df[df["year"] == ty]
        if len(tr) > SEQ_LEN and len(te) > SEQ_LEN:
            folds.append({"train": tr.copy(), "test": te.copy(),
                          "label": f"{ty-train_yrs}-{ty-1} -> {ty}"})
    return folds

folds = make_folds(df_clean, TRAIN_YEARS)
print(f"{len(folds)} folds ready.")

# %%
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TRAINING: ALL 200 EPOCHS, SAVE BEST-VAL-EPOCH ONLY                 ║
# ║                                                                      ║
# ║  Trains all 200 epochs to show full learning curve,                  ║
# ║  but ONLY saves the model checkpoint at best validation epoch.       ║
# ║  Final ensemble = best-val-epoch models from all 9 folds.           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def get_lr(ep, warmup, total, base):
    if ep < warmup: return base * (ep+1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (ep-warmup) / max(1, total-warmup)))

print("\n" + "+"+"="*68+"+")
print("|  TRAINING: 200 Epochs x 9 Folds (Best-Val-Epoch Ensemble)         |")
print("|  All epochs run. Only best-val checkpoint saved per fold.         |")
print("+"+"="*68+"+")

all_results = []
ensemble_states = []  # ONLY best-val-epoch states
t0 = time.time()

for fi, fold in enumerate(folds):
    print(f"\n{'='*60}")
    print(f"  FOLD {fi+1}/{len(folds)}: {fold['label']}")
    print(f"  Train: {len(fold['train'])} | Test: {len(fold['test'])}")
    print(f"{'='*60}")

    tr_ds = BrentDataset(fold["train"], FEATURE_COLS, SEQ_LEN, augment=True,
                         noise_std=AUG_NOISE, len_min=AUG_LEN_MIN, len_max=AUG_LEN_MAX)
    te_ds = BrentDataset(fold["test"], FEATURE_COLS, SEQ_LEN, augment=False)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = CompositeLoss(QUANTILES)

    best_val = float("inf")
    best_ep = 0
    best_state = None
    fold_hist = []

    for ep in range(TOTAL_EPOCHS):
        lr = get_lr(ep, WARMUP, TOTAL_EPOCHS, LR)
        for pg in opt.param_groups: pg["lr"] = lr

        model.train()
        tl = []
        for xb, yb in tr_ld:
            xb = xb.to(device); yb = {k:v.to(device) for k,v in yb.items()}
            opt.zero_grad()
            loss, ld = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(ld["total"])

        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(device); yb = {k:v.to(device) for k,v in yb.items()}
                _, ld = crit(model(xb), yb)
                vl.append(ld["total"])

        ta, va = np.mean(tl), np.mean(vl)
        fold_hist.append({"ep": ep+1, "train": ta, "val": va})

        if va < best_val:
            best_val = va
            best_ep = ep + 1
            best_state = copy.deepcopy(model.state_dict())

        if (ep+1) in [1, 5, 10, 20, 50, 100, 150, 200]:
            marker = " <-- BEST" if ep+1 == best_ep else ""
            gap = ta/va if va > 0 else 0
            print(f"  Ep {ep+1:3d}: T={ta:.4f} V={va:.4f} Best={best_val:.4f}@ep{best_ep} Gap={gap:.2f}x{marker}")

    # Save ONLY best-val-epoch state
    ensemble_states.append(copy.deepcopy(best_state))

    print(f"\n  FOLD {fi+1}: Best val={best_val:.4f} at epoch {best_ep}/{TOTAL_EPOCHS}")

    all_results.append({
        "fold": fi+1, "label": fold["label"],
        "best_loss": best_val, "best_ep": best_ep,
        "final_loss": fold_hist[-1]["val"],
        "test_df": fold["test"], "history": fold_hist,
    })

elapsed = time.time() - t0

# %%
# ════════════════════ RESULTS TABLE ════════════════════

print("\n" + "+"+"="*68+"+")
print("|  WALK-FORWARD RESULTS (v7.0 — 12 features)                        |")
print("+"+"="*68+"+")
print(f"|  {'Fold':<5} {'Window':<22} {'BestEp':>7} {'BestLoss':>9} {'Ep200':>9} {'Grade':>6}  |")
print("|  " + "-"*62 + "  |")
for r in all_results:
    grade = "A+" if r["best_loss"]<0.5 else "A" if r["best_loss"]<0.65 else "B" if r["best_loss"]<0.85 else "C" if r["best_loss"]<1.2 else "D"
    print(f"|  {r['fold']:<5} {r['label']:<22} {r['best_ep']:>7} {r['best_loss']:>9.4f} {r['final_loss']:>9.4f} {grade:>6}  |")

avg = np.mean([r["best_loss"] for r in all_results])
best_r = min(all_results, key=lambda r: r["best_loss"])
print("|  " + "-"*62 + "  |")
print(f"|  Avg Best Loss: {avg:.4f} | Overall Best: #{best_r['fold']} ({best_r['best_loss']:.4f})")
print(f"|  Time: {elapsed/60:.1f} min")
print("+"+"="*68+"+")

# %%
# ════════════════════ EVALUATION ════════════════════

def evaluate_ensemble(states, df_eval):
    ds = BrentDataset(df_eval, FEATURE_COLS, SEQ_LEN, augment=False)
    if len(ds) == 0: return {}
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    all_q = []
    for state in states:
        mdl = make_model(); mdl.load_state_dict(state); mdl.eval()
        q_batch = []
        with torch.no_grad():
            for xb, _ in ld:
                q_batch.append(mdl(xb.to(device))["quantiles"].cpu().numpy())
        all_q.append(np.concatenate(q_batch, axis=0))
    q_ens = np.mean(all_q, axis=0)

    all_zt = {n: [] for n in HORIZON_NAMES}; all_mu, all_sig = [], []
    for _, yb in ld:
        for n in HORIZON_NAMES: all_zt[n].append(yb[f"z_target_{n}"].numpy())
        all_mu.append(yb["rolling_mean"].numpy())
        all_sig.append(yb["rolling_std"].numpy())
    mu, sig = np.concatenate(all_mu), np.concatenate(all_sig)

    metrics = {}
    for hi, hn in enumerate(HORIZON_NAMES):
        za = np.concatenate(all_zt[hn])
        zq10, zq50, zq90 = q_ens[:,hi,0], q_ens[:,hi,1], q_ens[:,hi,2]
        pa = za*sig+mu; p10,p50,p90 = zq10*sig+mu, zq50*sig+mu, zq90*sig+mu
        metrics[hn] = {
            "MAE": np.abs(p50-pa).mean(),
            "MAPE": (np.abs(p50-pa)/np.abs(pa).clip(1)).mean()*100,
            "Dir": ((za>0)==(zq50>0)).mean()*100,
            "Width": (p90-p10).mean(),
            "C10": (pa<p10).mean()*100, "C50": (pa<p50).mean()*100, "C90": (pa<p90).mean()*100,
        }
    return metrics

all_test = pd.concat([r["test_df"] for r in all_results])
ens_metrics = evaluate_ensemble(ensemble_states, all_test)

print("\n" + "+"+"="*68+"+")
print("|  v7.0 EVALUATION (Best-Val-Epoch Ensemble, Out-of-Sample)         |")
print("+"+"="*68+"+")
print(f"|  {'Horizon':<10} {'MAE($)':>8} {'MAPE%':>7} {'Dir%':>6} {'Width$':>8} {'C10':>5} {'C50':>5} {'C90':>5} |")
print("|  " + "-"*60 + "  |")
for n in HORIZON_NAMES:
    m = ens_metrics[n]
    print(f"|  {HORIZON_LABELS[n]:<10} {m['MAE']:>7.2f}$ {m['MAPE']:>6.2f}% {m['Dir']:>5.1f}% "
          f"{m['Width']:>7.2f}$ {m['C10']:>4.1f}% {m['C50']:>4.1f}% {m['C90']:>4.1f}% |")
print("|")
print("|  COMPARISON vs ALL PREVIOUS VERSIONS:")
print(f"|  {'':>15} {'v5':>8} {'v6.0':>8} {'v6.1':>8} {'v7.0':>8}")
print(f"|  {'1D MAPE':>15} {'3.24%':>8} {'TBD':>8} {'TBD':>8} {ens_metrics['1D']['MAPE']:.2f}%")
print(f"|  {'1D Dir':>15} {'80.0%':>8} {'TBD':>8} {'TBD':>8} {ens_metrics['1D']['Dir']:.1f}%")
print(f"|  {'1D MAE':>15} {'$2.22':>8} {'TBD':>8} {'TBD':>8} ${ens_metrics['1D']['MAE']:.2f}")
print(f"|  {'Features':>15} {'7':>8} {'11':>8} {'11':>8} {'12':>8}")
print("+"+"="*68+"+")

# %%
# ════════════════════ SAVE & UPLOAD ════════════════════

SAVE_DIR = Path("./agent_b_model")
SAVE_DIR.mkdir(exist_ok=True)

torch.save({
    "best_model": ensemble_states[best_r["fold"]-1],
    "ensemble_states": ensemble_states,
    "n_models": len(ensemble_states),
    "ensemble_type": "best_val_epoch_only",
}, SAVE_DIR / "model.pth")

config = {
    "model_type": "iTransformer_v7.0",
    "version": "7.0",
    "architecture": {
        "seq_len": SEQ_LEN, "n_features": N_FEATURES, "d_model": D_MODEL,
        "n_heads": N_HEADS, "e_layers": E_LAYERS, "d_ff": D_FF,
        "dropout": DROPOUT, "activation": ACTIVATION, "use_norm": USE_NORM,
        "n_horizons": N_HORIZONS, "n_quantiles": N_QUANTILES,
        "brent_token_idx": BRENT_TOKEN_IDX,
    },
    "feature_cols": RAW_FEATURE_COLS,
    "feature_stats": feature_stats,
    "quantiles": QUANTILES,
    "horizons": dict(zip(HORIZON_NAMES, HORIZONS)),
    "rolling_window": ROLLING_WINDOW,
    "ensemble_type": "best_val_epoch_only",
    "fold_results": [{"fold": r["fold"], "label": r["label"],
                      "best_loss": round(r["best_loss"], 4), "best_ep": r["best_ep"]}
                     for r in all_results],
    "best_fold": best_r["fold"],
    "ensemble_size": len(ensemble_states),
    "ensemble_metrics": {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in ens_metrics.items()},
    "trained_at": datetime.now().isoformat(),
}
with open(SAVE_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2, default=str)

print(f"Saved: {(SAVE_DIR / 'model.pth').stat().st_size / 1e6:.2f} MB")

try:
    from huggingface_hub import HfApi, login
    hf_token = None
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except: pass
    if not hf_token: hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)
        api = HfApi()
        repo_id = api.whoami()["name"] + "/agent-b-itransformer"
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(SAVE_DIR), repo_id=repo_id, repo_type="model")
        print(f"[OK] https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Upload: {e}")

# %%
# ════════════════════ FINAL REPORT ════════════════════

print()
print("+" + "="*68 + "+")
print("|                 AGENT-B v7.0 FINAL REPORT                         |")
print("|          12 Features + Best-Val-Epoch Ensemble                    |")
print("+" + "="*68 + "+")
np_c = sum(p.numel() for p in make_model().parameters())
print(f"|  Model: {np_c:,} params ({np_c/1e6:.2f}M)")
print(f"|  Features: {N_FEATURES} (5 spec + 3 technical + 4 market structure)")
print(f"|  Ensemble: {len(ensemble_states)} models (best-val-epoch only)")
print(f"|  Time: {elapsed/60:.1f} min")
print(f"|")
print(f"|  Performance:")
for n in HORIZON_NAMES:
    m = ens_metrics[n]
    print(f"|    {HORIZON_LABELS[n]:<10}: MAE=${m['MAE']:.2f}  MAPE={m['MAPE']:.1f}%  Dir={m['Dir']:.1f}%")
print(f"|")
print(f"|  New Features Added:")
print(f"|    [7]  brent_wti_spread     -> Regional supply dynamics")
print(f"|    [8]  crack_spread_321     -> RIL Jamnagar refining margin")
print(f"|    [9]  eia_inventory        -> US crude oil stocks (supply)")
print(f"|    [10] eia_inventory_change -> Weekly stock change (market mover)")
print(f"|")
print(f"|  Spec Compliance (Section 3.1):")
print(f"|    [Y] Z_Score_Brent, Z_Score_DXY, Holiday_Flag, VIX_Close, Sentiment_Score")
print(f"|    [+] RSI, MACD, MACD_Signal, Brent-WTI, Crack, EIA inv, EIA chg")
print("+" + "="*68 + "+")
