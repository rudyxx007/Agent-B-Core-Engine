"""
scripts/modal_app/inference.py — iTransformer Live Inference

Downloads the deployed model weights from Hugging Face, instantiates the 
iTransformer architecture, and returns the quantile predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from huggingface_hub import hf_hub_download

# ============================================================
# iTransformer Architecture (Must match exactly)
# ============================================================
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
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="silu"):
        super().__init__()
        self.attn = AttentionLayer(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.silu
    def forward(self, x):
        new_x, attn = self.attn(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.act(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm2(x + y), attn

class iTransformer(nn.Module):
    def __init__(self, seq_len=90, n_features=12, d_model=128, n_heads=8, e_layers=2,
                 d_ff=256, dropout=0.3, n_horizons=3, n_quantiles=3, brent_idx=0):
        super().__init__()
        self.use_norm = True
        self.brent_idx = brent_idx
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, "silu") for _ in range(e_layers)])
        self.encoder_norm = nn.LayerNorm(d_model)
        
        # Global Pooling requires 2*d_model
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(2*d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(d_model, n_quantiles)) for _ in range(n_horizons)])
        self.vol_head = nn.Sequential(nn.Linear(2*d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))
        self.ma_head = nn.Sequential(nn.Linear(2*d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))

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
        
        # Global Pooling
        brent = h[:, self.brent_idx, :]
        global_pool = h.mean(dim=1)
        combined = torch.cat([brent, global_pool], dim=-1)
        
        q = torch.stack([head(combined) for head in self.quantile_heads], dim=1)
        return {"quantiles": q, "volatility": self.vol_head(combined), "ma_logit": self.ma_head(combined)}

# ============================================================
# Inference Logic
# ============================================================
def load_model() -> iTransformer:
    print("Downloading iTransformer weights from Hugging Face...")
    model_path = hf_hub_download(repo_id="rudyxx07/agent-b-itransformer", filename="model.pth")
    
    model = iTransformer(d_model=256, d_ff=512)
    # Handle CPU mapping in case Modal container is missing GPU temporarily
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def predict_horizons(features_tensor: torch.Tensor) -> dict:
    """
    Runs the forward pass and returns unscaled Z-score predictions.
    """
    model = load_model()
    device = next(model.parameters()).device
    features_tensor = features_tensor.to(device)
    
    with torch.no_grad():
        preds = model(features_tensor)
        
    quantiles = preds["quantiles"].squeeze(0).cpu().numpy()  # [3 horizons, 3 quantiles]
    vol = preds["volatility"].item()
    ma_cross = torch.sigmoid(preds["ma_logit"]).item() > 0.5
    
    return {
        "pred_1d_10th": float(quantiles[0][0]),
        "pred_1d_50th": float(quantiles[0][1]),
        "pred_1d_90th": float(quantiles[0][2]),
        "pred_1m_10th": float(quantiles[1][0]),
        "pred_1m_50th": float(quantiles[1][1]),
        "pred_1m_90th": float(quantiles[1][2]),
        "pred_3m_10th": float(quantiles[2][0]),
        "pred_3m_50th": float(quantiles[2][1]),
        "pred_3m_90th": float(quantiles[2][2]),
        "pred_volatility": float(vol),
        "pred_ma_crossover": int(ma_cross)
    }
