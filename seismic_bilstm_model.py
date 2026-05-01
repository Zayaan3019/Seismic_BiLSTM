"""
Bidirectional LSTM Seismic Ground Motion Model
===============================================
Architecture: Feature Encoder MLP → Period Embeddings → BiLSTM → Output Head

Inputs  (8): Mw, Ztor, Finite_Fault_Model, Rjb, log(Vs30), log(Rjb), Mw², Mw·log(Rjb)
Outputs (30): PGA_g + T0pt010S … T10pt000S (0.01 s – 10.0 s) + PGV_cm_sec

The 30 outputs are treated as a sequence ordered by pseudo-period.
PGA is assigned pseudo-period 0.005 s; PGV is assigned pseudo-period 15.0 s
(appended after the spectral sequence). A bidirectional LSTM processes this
sequence in both short→long and long→short directions, capturing spectral
correlation across the full 0.01–10.0 s range and ground-velocity output.

Generates: Fig 01–05, Table 01, and saves model artefacts to sda_results/
"""

import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from sda_pipeline_utils import (
    FAULT_LABEL_MAP,
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    RANDOM_STATE,
    get_fault_labels,
    output_to_label,
    prepare_dataset,
    split_data,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Direct file logger — writes progress to training_log.txt with every flush
# so monitoring works regardless of stdout buffering mode.
import io as _io
_log_path = os.path.join("sda_results", "training_log.txt")
os.makedirs("sda_results", exist_ok=True)
_log_fh = open(_log_path, "w", encoding="utf-8", buffering=1)  # line-buffered

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)
    _log_fh.write(" ".join(str(a) for a in args) + "\n")
    _log_fh.flush()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_INPUTS  = len(INPUT_COLUMNS)   # 8 (includes Mw_sq and Mw_logR physics features)
N_OUTPUTS = len(OUTPUT_COLUMNS)  # 30 (29 spectral + PGV)
DEVICE    = torch.device("cpu")

# One log-period value per output column, used for continuous period encoding.
# PGA is assigned pseudo-period 0.005 s (half of shortest PSA period).
# PGV is assigned pseudo-period 15.0 s (appended after the spectral sequence).
PERIODS_S = np.array([
    0.005,
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    15.0,   # PGV pseudo-period
], dtype=np.float32)  # shape (30,)

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class SeismicBiLSTM(nn.Module):
    """
    Encoder-sequence BiLSTM for multi-output seismic ground motion prediction.

    Pipeline
    --------
    1. Feature Encoder : 8-feature vector → context embedding (256-dim)
                         with residual projection for gradient flow
    2. Period Encoding : continuous log(T) MLP (1 → 32 → 64), physics-grounded
    3. Sequence input  : [context ⊕ period_embed] for each period → (batch, 29, 320)
    4. BiLSTM          : processes spectral sequence forward AND backward
    5. Output head     : per-timestep projection (512 → 128 → 32 → 1) → (batch, 29)
    """

    def __init__(
        self,
        n_inputs: int = N_INPUTS,
        n_periods: int = N_OUTPUTS,
        encoder_dims: tuple = (128, 256, 512, 512, 256),
        period_embed_dim: int = 64,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.n_periods = n_periods
        context_dim = encoder_dims[-1]

        # -- Feature Encoder with residual projection --
        layers = []
        in_dim = n_inputs
        for i, out_dim in enumerate(encoder_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            if i >= 1:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*layers)
        # Residual projection from input → context_dim for skip connection
        self.input_proj = nn.Linear(n_inputs, context_dim, bias=False)

        # -- Continuous period encoding: log(T) → 32 → period_embed_dim --
        self.period_net = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, period_embed_dim)
        )
        log_T = torch.FloatTensor(np.log(PERIODS_S[:n_periods]))
        self.register_buffer("log_T", log_T)   # (n_periods,) — saved with state dict

        # -- BiLSTM --
        lstm_input_dim = context_dim + period_embed_dim   # 256 + 64 = 320
        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # -- Output Head --
        bilstm_out_dim = lstm_hidden * 2   # 512 (forward + backward)
        self.output_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        # Encoder with residual skip connection
        context = self.encoder(x) + self.input_proj(x)                 # (batch, 256)

        p_emb = self.period_net(self.log_T.unsqueeze(-1))               # (29, 64)
        p_emb = p_emb.unsqueeze(0).expand(batch, -1, -1)               # (batch, 29, 64)

        ctx = context.unsqueeze(1).expand(-1, self.n_periods, -1)       # (batch, 29, 256)
        seq = torch.cat([ctx, p_emb], dim=-1)                           # (batch, 29, 320)

        lstm_out, _ = self.bilstm(seq)                                  # (batch, 29, 512)
        out = self.output_head(lstm_out).squeeze(-1)                    # (batch, 29)
        return out


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class BiLSTMWrapper:
    """
    Wraps SeismicBiLSTM to expose the same .predict() interface as
    sklearn estimators, so downstream scripts work without modification.
    Inputs/outputs are in *scaled* space (StandardScaler applied externally).
    """

    def __init__(self, model: SeismicBiLSTM, device=DEVICE):
        self.model  = model
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            return self.model(t).cpu().numpy()

    def __repr__(self):
        return f"BiLSTMWrapper({self.model})"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse":  float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def print_metric_block(name: str, metrics: dict) -> None:
    print(f"\n   {name} set metrics:")
    print("   " + "-" * 52)
    print(f"   MSE : {metrics['mse']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   MAE : {metrics['mae']:.6f}")
    print(f"   R2  : {metrics['r2']:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 80)
print("BIDIRECTIONAL LSTM SEISMIC GROUND MOTION PREDICTION MODEL")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[1/10] Loading and preprocessing data...")
bundle   = prepare_dataset("NGA_Subduction_filtered.csv")
clean_df = bundle["clean_df"]
X_df     = bundle["X_df"]
y_raw_df = bundle["y_raw_df"]
y_log_df = bundle["y_log_df"]
X        = bundle["X"]
y        = bundle["y_log"]

print(f"   Dataset loaded : {len(clean_df)} clean records")
print(f"   Input features : {N_INPUTS}  {INPUT_COLUMNS}")
print(f"   Output features: {N_OUTPUTS}")
print("   Output domain  : natural-log transformed")

# ============================================================================
# 2. OUTPUT DIRECTORY
# ============================================================================

output_dir = "sda_results"
os.makedirs(output_dir, exist_ok=True)
print(f"\n[2/10] Output directory ready: {output_dir}/")

# ============================================================================
# 3. FIG 01 / FIG 02 / FIG 03  — data-distribution figures
# ============================================================================

print("\n[3/10] Generating data distribution and frequency plots...")

# Fig 01 — Mw–Rjb scatter (top) + PGA–Rjb by magnitude class (bottom)
# Bins and colours chosen to match the reference figure style:
#   3≤Mw<4 (blue), 4≤Mw<5 (green), 5≤Mw<6 (orange), 6≤Mw<7 (gold), 7≤Mw<8 (red/magenta)
# Mw≥8 events (outside reference range) are assigned a 6th grey bin so no
# data are silently dropped — they appear in the top scatter but not the PGA plot.
magnitude_bins   = [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 12)]
colors_mag_fig01 = ["#2196F3", "#4CAF50", "#FF9800", "#FFC107", "#E91E63", "#9E9E9E"]
labels_fig01     = ["3≤Mw<4", "4≤Mw<5", "5≤Mw<6", "6≤Mw<7", "7≤Mw<8", "Mw≥8"]

fig, axes = plt.subplots(2, 1, figsize=(11, 12))

ax1 = axes[0]
ax1.scatter(
    clean_df["Rjb_km"].clip(lower=0.01),
    clean_df["Earthquake_Magnitude"],
    alpha=0.30, s=20, facecolors="none", edgecolors="gray", linewidth=0.5,
)
ax1.set_xscale("log")
ax1.set_xlabel(r"Joyner-Boore distance, $R_{jb}$ (km)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Magnitude ($M_w$)", fontsize=12, fontweight="bold")
ax1.set_title("Fig 01: Data used", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3, which="both")
ax1.set_xlim(1e-2, 1.1e3)
ax1.set_ylim(3, 9.5)

ax2 = axes[1]
for (m_min, m_max), color, label in zip(magnitude_bins, colors_mag_fig01, labels_fig01):
    msk = (clean_df["Earthquake_Magnitude"] >= m_min) & (clean_df["Earthquake_Magnitude"] < m_max)
    ax2.scatter(
        clean_df.loc[msk, "Rjb_km"].clip(lower=0.01),
        clean_df.loc[msk, "PGA_g"],
        alpha=0.50, s=25, c=color, label=label, edgecolors="none",
    )
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Joyner-Boore distance, $R_{jb}$ (km)", fontsize=12, fontweight="bold")
ax2.set_ylabel("PGA (g)", fontsize=12, fontweight="bold")
ax2.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, which="both")
ax2.set_xlim(1e-2, 1.1e3)
ax2.set_ylim(1e-6, 2)
plt.tight_layout()
plt.savefig(f"{output_dir}/fig01_data_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# Fig 02 — Fault-mechanism breakdown (Fault_Type kept in clean_df for viz)
fault_labels = get_fault_labels(clean_df["Fault_Type"])
marker_map = {"Strike Slip": "x", "Normal": "*", "Reverse": "s", "Other": "o"}
color_map  = {"Strike Slip": "#b04ee0", "Normal": "#58b947", "Reverse": "#f1b722", "Other": "#2ca8ff"}

fig, axes = plt.subplots(2, 1, figsize=(11, 12))
for ax, title_text in zip(axes, [
    "Fig 02: Data used — Fault mechanism view (full set)",
    "Fig 02: Data used — Fault mechanism view (modeling set)",
]):
    for lbl in ["Strike Slip", "Normal", "Reverse", "Other"]:
        msk = fault_labels == lbl
        if not msk.any():
            continue
        ax.scatter(
            clean_df.loc[msk, "Rjb_km"].clip(lower=0.01),
            clean_df.loc[msk, "Earthquake_Magnitude"],
            alpha=0.45, s=28, marker=marker_map[lbl], c=color_map[lbl], label=lbl,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Joyner-Boore distance ($R_{jb}$), km", fontsize=12, fontweight="bold")
    ax.set_ylabel("Magnitude ($M_w$)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1e-2, 1.1e3)
    ax.set_ylim(4, 9.5)
    ax.set_title(title_text, fontsize=13, fontweight="bold")
axes[0].legend(loc="lower left", fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{output_dir}/fig02_data_used.png", dpi=300, bbox_inches="tight")
plt.close()

# Fig 03 — frequency histograms
fault_counts = fault_labels.value_counts()
fault_order  = ["Normal", "Reverse", "Strike Slip", "Other"]
plot_labels  = [l for l in fault_order if l in fault_counts.index]
plot_counts  = [int(fault_counts[l]) for l in plot_labels]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(clean_df["Earthquake_Magnitude"], bins=22, edgecolor="black",
             color="#F5B59E", alpha=0.85)
axes[0].set_xlabel("Magnitude ($M_w$)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("No. of records",     fontsize=12, fontweight="bold")
axes[0].set_title("Magnitude distribution", fontsize=13, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].hist(clean_df["Rjb_km"], bins=30, edgecolor="black", color="#F5B59E", alpha=0.85)
axes[1].set_xlabel(r"Joyner-Boore distance, $R_{jb}$ (km)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("No. of records",                         fontsize=12, fontweight="bold")
axes[1].set_title("Distance distribution", fontsize=13, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")

axes[2].barh(
    np.arange(len(plot_labels)), plot_counts,
    color=["#F8D8C8", "#F6B996", "#F08B66", "#DFA27A"][: len(plot_labels)],
    edgecolor="black", alpha=0.85,
)
axes[2].set_yticks(np.arange(len(plot_labels)))
axes[2].set_yticklabels(plot_labels, fontsize=11)
axes[2].set_xlabel("No. of records", fontsize=12, fontweight="bold")
axes[2].set_title("Fault mechanism distribution", fontsize=13, fontweight="bold")
axes[2].grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(f"{output_dir}/fig03_frequency_plots.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: Fig 01, Fig 02, Fig 03")

# ============================================================================
# 4. TRAIN / VAL / TEST SPLIT
# ============================================================================

print("\n[4/10] Splitting dataset (70/15/15)...")
splits  = split_data(X, y, random_state=RANDOM_STATE)
X_train = splits["X_train"]
X_val   = splits["X_val"]
X_test  = splits["X_test"]
y_train = splits["y_train"]
y_val   = splits["y_val"]
y_test  = splits["y_test"]
print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ============================================================================
# 5. SCALING
# ============================================================================

print("\n[5/10] Scaling features and targets...")
scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_val_sc   = scaler_X.transform(X_val)
X_test_sc  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_sc = scaler_y.fit_transform(y_train)
y_val_sc   = scaler_y.transform(y_val)
y_test_sc  = scaler_y.transform(y_test)

# ============================================================================
# 6. BUILD MODEL
# ============================================================================

print("\n[6/10] Building BiLSTM model...")
net = SeismicBiLSTM(
    n_inputs=N_INPUTS,
    n_periods=N_OUTPUTS,
    encoder_dims=(128, 256, 256),
    period_embed_dim=64,
    lstm_hidden=128,
    lstm_layers=2,
    dropout=0.20,
).to(DEVICE)

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"   Architecture : 8 -> [128->256->256]+residual -> BiLSTM(128x2) -> 30")
print(f"   Total params : {total_params:,}")
print(f"   Device       : {DEVICE}")

# ============================================================================
# 7. TRAINING
# ============================================================================

print("\n[7/10] Training BiLSTM (AdamW + Huber + OneCycleLR + early stopping)...")

BATCH_SIZE     = 256
MAX_EPOCHS     = 150
PATIENCE       = 30
LR             = 2e-3   # max_lr for OneCycleLR
WEIGHT_DECAY   = 1e-4
LAMBDA_SMOOTH  = 5e-4   # spectral 2nd-difference smoothness penalty weight


def make_loader(X_arr, y_arr, batch_size=128, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X_arr), torch.FloatTensor(y_arr))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


train_loader = make_loader(X_train_sc, y_train_sc, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = make_loader(X_val_sc,   y_val_sc,   batch_size=512, shuffle=False)

optimizer = torch.optim.AdamW(net.parameters(), lr=LR / 10, weight_decay=WEIGHT_DECAY)
# OneCycleLR: warms up LR to max_lr then anneals — proven to converge 3-5× faster
# than ReduceLROnPlateau. Called per batch, not per epoch.
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    epochs=MAX_EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.30,          # 30% warmup phase
    anneal_strategy="cos",
    div_factor=10.0,         # start LR = max_lr / 10
    final_div_factor=1e4,    # end LR = start_lr / 1e4
)
# Huber loss (delta=0.5): less sensitive to log-space outliers than MSE, preserves gradients
criterion = nn.HuberLoss(delta=0.5)

best_val_loss = float("inf")
best_state    = None
no_improve    = 0
train_losses  = []
val_losses    = []
best_epoch    = 0

for epoch in range(1, MAX_EPOCHS + 1):
    # -- Train --
    net.train()
    epoch_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = net(Xb)
        loss = criterion(pred, yb)
        # Penalise 2nd finite difference across the spectral axis for smoother curves
        diff2 = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
        loss = loss + LAMBDA_SMOOTH * diff2.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()   # OneCycleLR steps per batch, not per epoch
        epoch_loss += loss.item() * Xb.size(0)
    epoch_loss /= len(X_train_sc)

    # -- Validate --
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            val_loss += criterion(net(Xb), yb).item() * Xb.size(0)
    val_loss /= len(X_val_sc)

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_state    = {k: v.clone() for k, v in net.state_dict().items()}
        no_improve    = 0
        best_epoch    = epoch
    else:
        no_improve += 1

    current_lr = optimizer.param_groups[0]["lr"]
    if epoch % 10 == 0 or epoch == 1:
        print(
            f"   Epoch {epoch:4d}/{MAX_EPOCHS} | "
            f"train={epoch_loss:.5f} | val={val_loss:.5f} | "
            f"lr={current_lr:.2e} | "
            f"best_val={best_val_loss:.5f} @ ep{best_epoch}"
        )

    if no_improve >= PATIENCE:
        print(f"\n   Early stopping at epoch {epoch} (patience={PATIENCE})")
        break

net.load_state_dict(best_state)
print(f"\n   Best model: epoch {best_epoch}, val_loss={best_val_loss:.6f}")

# ============================================================================
# 8. EVALUATE
# ============================================================================

print("\n[8/10] Evaluating model on all splits...")
wrapper = BiLSTMWrapper(net, device=DEVICE)


def eval_split(X_sc, y_true):
    y_pred_sc = wrapper.predict(X_sc)
    y_pred    = scaler_y.inverse_transform(y_pred_sc)
    return y_pred, calculate_metrics(y_true.ravel(), y_pred.ravel())


y_train_pred, train_metrics = eval_split(X_train_sc, y_train)
y_val_pred,   val_metrics   = eval_split(X_val_sc,   y_val)
y_test_pred,  test_metrics  = eval_split(X_test_sc,  y_test)

print_metric_block("Train", train_metrics)
print_metric_block("Val",   val_metrics)
print_metric_block("Test",  test_metrics)

feature_metrics = []
for i, out_name in enumerate(OUTPUT_COLUMNS):
    y_t  = y_test[:, i]
    y_p  = y_test_pred[:, i]
    r    = float(np.corrcoef(y_t, y_p)[0, 1])
    r2   = float(r2_score(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae  = float(mean_absolute_error(y_t, y_p))
    feature_metrics.append({
        "feature": out_name,
        "label":   output_to_label(out_name),
        "r2": r2, "rmse": rmse, "mae": mae, "r": r,
    })

# ============================================================================
# 9. FIG 04 — Architecture diagram   |   FIG 05 — Regression scatter
# ============================================================================

print("\n[9/10] Generating Fig 04 (architecture) and Fig 05 (regression)...")

# --- Fig 04 ---
fig, ax = plt.subplots(figsize=(15, 7))
ax.axis("off")
ax.set_xlim(0, 15)
ax.set_ylim(0, 7)

layer_defs = [
    (1.0,  "Input\n(8)",          "#D5E8D4", "#82B366"),
    (3.2,  "Dense 128\n+LN+GELU", "#DAE8FC", "#6C8EBF"),
    (5.4,  "Dense 256\n+LN+GELU", "#DAE8FC", "#6C8EBF"),
    (7.6,  "Dense 256\n+LN+GELU", "#DAE8FC", "#6C8EBF"),
    (9.8,  "BiLSTM\n(128x2)",     "#FFE6CC", "#D79B00"),
    (12.0, "Output\n(30)",        "#F8CECC", "#B85450"),
]

box_w, box_h = 1.35, 1.05
for (x, label, fc, ec) in layer_defs:
    rect = plt.Rectangle(
        (x - box_w / 2, 3.0 - box_h / 2), box_w, box_h,
        facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, 3.0, label, ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=4)

for i in range(len(layer_defs) - 1):
    x0 = layer_defs[i][0] + box_w / 2
    x1 = layer_defs[i + 1][0] - box_w / 2
    ax.annotate("", xy=(x1, 3.0), xytext=(x0, 3.0),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#555555"))

input_labels = [r"$M_w$", r"$Z_{tor}$", r"$FFM$", r"$R_{jb}$",
                r"$\log(V_{s30})$", r"$\log(R_{jb})$",
                r"$M_w^2$", r"$M_w{\cdot}\log R$"]
for j, txt in enumerate(input_labels):
    ax.text(1.2, 4.2 + j * 0.30, txt, ha="center", va="bottom",
            fontsize=8.0, style="italic", color="#2D6A2D")

ax.text(12.2, 5.8, "Spectral outputs (ln-space)", ha="center",
        fontsize=9, fontstyle="italic", color="#8B0000")

ax.set_title(
    r"Fig 04: BiLSTM Architecture  "
    r"$[8 \rightarrow 128 \rightarrow 256 \rightarrow 256"
    r"\rightarrow \mathrm{BiLSTM}(128{\times}2) \rightarrow 30]$",
    fontsize=11, fontweight="bold", pad=14,
)
plt.tight_layout()
plt.savefig(f"{output_dir}/fig04_architecture.png", dpi=300, bbox_inches="tight")
plt.close()

# --- Fig 05 — Observed vs predicted, all three splits ---
# Matches PDF reference: Training (green ○), Validation (orange ○), Testing (gold ○),
# ideal-fit line (black), and Pearson R in the title.
fig05_targets = ["PGA_g", "T0pt100S", "T1pt000S", "PGV_cm_sec"]
fig05_labels  = ["PGA", r"PSA$_{0.1s}$", r"PSA$_{1.0s}$", "PGV"]

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

for ax, target, panel_label in zip(axes.ravel(), fig05_targets, fig05_labels):
    idx = OUTPUT_COLUMNS.index(target)

    # Extract per-split arrays in log-space
    tr_t = y_train[:, idx];    tr_p = y_train_pred[:, idx]
    va_t = y_val[:, idx];      va_p = y_val_pred[:, idx]
    te_t = y_test[:, idx];     te_p = y_test_pred[:, idx]

    r_te = float(np.corrcoef(te_t, te_p)[0, 1])

    # Subsample training scatter to avoid overplotting (show 1,500 points max)
    rng = np.random.default_rng(0)
    n_show = min(1500, len(tr_t))
    idx_show = rng.choice(len(tr_t), n_show, replace=False)

    # Filled markers matching reference PDF style:
    # Training = orange filled circles, Validation = yellow filled diamonds, Testing = gold filled squares
    ax.scatter(tr_t[idx_show], tr_p[idx_show],
               c="#E8853D", marker="o", s=14, alpha=0.35,
               edgecolors="none", label="Training data")
    ax.scatter(va_t, va_p,
               c="#F5D128", marker="D", s=16, alpha=0.60,
               edgecolors="none", label="Validation data")
    ax.scatter(te_t, te_p,
               c="#E8853D", marker="s", s=16, alpha=0.55,
               edgecolors="#5a3200", linewidths=0.4, label="Testing data")

    all_vals = np.concatenate([tr_t, tr_p, va_t, va_p, te_t, te_p])
    lims = [all_vals.min() - 0.3, all_vals.max() + 0.3]
    ax.plot(lims, lims, "k-", lw=1.6, label="Ideal fit")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Target (ln-space)",    fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted (ln-space)", fontsize=11, fontweight="bold")
    ax.set_title(f"{panel_label}   R = {r_te:.3f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.3)

fig.suptitle("Fig 05: Observed vs Predicted — Training / Validation / Testing (BiLSTM)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}/fig05_regression_plots.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: Fig 04, Fig 05")

# ============================================================================
# TABLE 01 — Statistical summary (PDF convention: −Rjb, −log(Rjb))
# ============================================================================

stat_series = {
    "Mw":          clean_df["Earthquake_Magnitude"],
    "-Rjb":       -clean_df["Rjb_km"],
    "-log(Rjb)":  -clean_df["log_Rjb"],
    "log(Vs30)":   clean_df["log_Vs30"],
    "Depth":       clean_df["Ztor_km"],
    "FFM":         clean_df["Finite_Fault_Model"].astype(float),
}
for col in OUTPUT_COLUMNS:
    stat_series[output_to_label(col)] = y_log_df[col]

rows_t1 = []
for name, s in stat_series.items():
    s_arr = np.asarray(s, dtype=float)
    s_arr = s_arr[np.isfinite(s_arr)]
    rows_t1.append({
        "Parameter": name,
        "Min":      f"{s_arr.min():.2f}",
        "Max":      f"{s_arr.max():.2f}",
        "Mean":     f"{s_arr.mean():.2f}",
        "Median":   f"{np.median(s_arr):.2f}",
        "STD":      f"{s_arr.std():.2f}",
        "Skewness": f"{float(pd.Series(s_arr).skew()):.2f}",
        "Kurtosis": f"{float(pd.Series(s_arr).kurtosis()):.2f}",
    })

t1_df = pd.DataFrame(rows_t1)
fig, ax = plt.subplots(figsize=(18, 16))
ax.axis("off")
tbl = ax.table(cellText=t1_df.values, colLabels=t1_df.columns,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.0)
tbl.scale(1, 1.30)
for j in range(len(t1_df.columns)):
    tbl[(0, j)].set_facecolor("#D0D0D0")
    tbl[(0, j)].set_text_props(weight="bold")
plt.title("Table 01: Statistical Summary of Dataset", fontsize=14, fontweight="bold", pad=14)
plt.savefig(f"{output_dir}/table01_statistics.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: Table 01")

# ============================================================================
# 10. SAVE MODEL ARTEFACTS + CONFIG
# ============================================================================

print("\n[10/10] Saving model artefacts...")
joblib.dump(wrapper,  f"{output_dir}/seismic_model_robust.pkl")
joblib.dump(scaler_X, f"{output_dir}/scaler_X.pkl")
joblib.dump(scaler_y, f"{output_dir}/scaler_y.pkl")
torch.save(net.state_dict(), f"{output_dir}/bilstm_weights.pt")
print("   Saved: seismic_model_robust.pkl (BiLSTMWrapper)")
print("   Saved: scaler_X.pkl, scaler_y.pkl")
print("   Saved: bilstm_weights.pt")

config = {
    "model_type":       "BiLSTM",
    "input_features":   INPUT_COLUMNS,
    "output_features":  OUTPUT_COLUMNS,
    "architecture":     "8 -> [128->256->256]+residual -> BiLSTM(128, 2-layer, bidirectional) -> 30",
    "encoder_dims":     [128, 256, 256],
    "lstm_hidden":      128,
    "lstm_layers":      2,
    "period_embed_dim": 64,
    "total_params":     total_params,
    "train_samples":    len(X_train),
    "val_samples":      len(X_val),
    "test_samples":     len(X_test),
    "best_epoch":       best_epoch,
    "best_val_loss":    best_val_loss,
    "train_metrics":    train_metrics,
    "val_metrics":      val_metrics,
    "test_metrics":     test_metrics,
    "feature_metrics":  feature_metrics,
    "fault_label_map":  {str(k): v for k, v in FAULT_LABEL_MAP.items()},
    "log_transformed_outputs": True,
    "hyperparameters": {
        "optimizer":    "AdamW",
        "lr_init":      LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size":   BATCH_SIZE,
        "max_epochs":   MAX_EPOCHS,
        "patience":     PATIENCE,
        "lr_schedule":  "OneCycleLR(max_lr=2e-3,pct_start=0.30,anneal=cos)",
        "loss":           "HuberLoss(delta=0.5) + spectral_smoothness(lambda=5e-4)",
        "lambda_smooth":  LAMBDA_SMOOTH,
        "dropout":        0.20,
        "activation":     "GELU",
        "norm":           "LayerNorm",
        "residual":       True,
        "period_encoding": "continuous_log_T_MLP",
    },
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

with open(f"{output_dir}/model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)
print("   Saved: model_config.json")

print("\n" + "=" * 80)
print("BIDIRECTIONAL LSTM TRAINING COMPLETE")
print("=" * 80)
print(f"End Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Best Epoch: {best_epoch}")
print(f"Train R²  : {train_metrics['r2']:.4f}")
print(f"Val R²    : {val_metrics['r2']:.4f}")
print(f"Test R²   : {test_metrics['r2']:.4f}")
print(f"Test RMSE : {test_metrics['rmse']:.4f}")
print("=" * 80)
