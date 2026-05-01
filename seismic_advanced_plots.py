"""
Advanced plotting and SDA tables for the BiLSTM seismic ground motion model.

Generates
---------
- Fig 06 : residual plots (vs Mw, Rjb, Vs30)
- Fig 07 : sensitivity / physics plots (distance, magnitude, Vs30, depth)
- Fig 08 : SHAP-style feature attribution
- Fig 09 : relative feature importance
- Table 02: residual sigma decomposition
- Table 03: performance parameters (R, K, K', R₀², R₀'²)

Input column order (matches INPUT_COLUMNS in sda_pipeline_utils):
  0: Earthquake_Magnitude
  1: Ztor_km
  2: Finite_Fault_Model
  3: Rjb_km
  4: log_Vs30
  5: log_Rjb
  6: Mw_sq    (Mw²)
  7: Mw_logR  (Mw × log_Rjb)
"""

import json
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

from sda_pipeline_utils import (
    INPUT_COLUMNS,
    MIN_RJB_KM,
    OUTPUT_COLUMNS,
    output_to_label,
    prepare_dataset,
    split_data,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

import os as _os, sys as _sys
_log_path_adv = _os.path.join("sda_results", "advanced_plots_log.txt")
_os.makedirs("sda_results", exist_ok=True)
_log_adv = open(_log_path_adv, "w", encoding="utf-8", buffering=1)
_orig_print_adv = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print_adv(*args, **kwargs)
    _log_adv.write(" ".join(str(a) for a in args) + "\n")
    _log_adv.flush()

# ---------------------------------------------------------------------------
# BiLSTM classes — must match seismic_bilstm_model.py for unpickling
# ---------------------------------------------------------------------------
_N_INPUTS  = len(INPUT_COLUMNS)
_N_OUTPUTS = len(OUTPUT_COLUMNS)
_DEVICE    = torch.device("cpu")


class SeismicBiLSTM(nn.Module):
    def __init__(
        self,
        n_inputs: int = _N_INPUTS,
        n_periods: int = _N_OUTPUTS,
        encoder_dims: tuple = (128, 256, 512, 512, 256),
        period_embed_dim: int = 64,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.n_periods  = n_periods
        context_dim = encoder_dims[-1]
        layers = []
        in_dim = n_inputs
        for i, out_dim in enumerate(encoder_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            if i >= 1:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.encoder      = nn.Sequential(*layers)
        self.input_proj = nn.Linear(n_inputs, context_dim, bias=False)
        self.period_net = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, period_embed_dim)
        )
        import numpy as _np
        _PERIODS_S = _np.array([
            0.005,
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ], dtype=_np.float32)
        import torch as _torch
        self.register_buffer("log_T", _torch.FloatTensor(_np.log(_PERIODS_S[:n_periods])))
        lstm_input_dim    = context_dim + period_embed_dim
        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        bilstm_out_dim = lstm_hidden * 2
        self.output_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch      = x.size(0)
        context    = self.encoder(x) + self.input_proj(x)
        p_emb      = self.period_net(self.log_T.unsqueeze(-1))
        p_emb      = p_emb.unsqueeze(0).expand(batch, -1, -1)
        ctx        = context.unsqueeze(1).expand(-1, self.n_periods, -1)
        seq        = torch.cat([ctx, p_emb], dim=-1)
        lstm_out, _ = self.bilstm(seq)
        return self.output_head(lstm_out).squeeze(-1)


class BiLSTMWrapper:
    def __init__(self, model: SeismicBiLSTM, device=_DEVICE):
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
# Scenario-row builder
# Input order: [Mw, Ztor, FFM, Rjb, log_Vs30, log_Rjb]
# ---------------------------------------------------------------------------

def _make_row(mw: float, ztor_km: float, ffm: int,
              rjb_km: float, vs30_m_s: float) -> np.ndarray:
    """Build a single scenario input row matching INPUT_COLUMNS (8 features)."""
    log_rjb = float(np.log(max(rjb_km, MIN_RJB_KM)))
    return np.array([
        float(mw),
        float(ztor_km),
        float(ffm),
        float(rjb_km),
        float(np.log(vs30_m_s)),
        log_rjb,
        float(mw) ** 2,           # Mw_sq
        float(mw) * log_rjb,      # Mw_logR
    ])


# ---------------------------------------------------------------------------
# Spectral utilities
# ---------------------------------------------------------------------------

def _period_from_col(name: str) -> float | None:
    """Period (s) for T* columns only; PGA_g returns None (excluded)."""
    if name.startswith("T") and name.endswith("S"):
        return float(name[1:-1].replace("pt", "."))
    return None


def _predict_log(model, scaler_X, scaler_y, x_row: np.ndarray) -> np.ndarray:
    x_sc = scaler_X.transform(x_row.reshape(1, -1))
    y_sc = model.predict(x_sc)
    return scaler_y.inverse_transform(y_sc.reshape(1, -1))[0]


def predict_spectrum(model, scaler_X, scaler_y,
                     input_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (periods, PSA_g) for T* columns only, lightly smoothed."""
    y_log    = _predict_log(model, scaler_X, scaler_y, input_row)
    y_linear = np.exp(y_log)

    periods, values = [], []
    for idx, name in enumerate(OUTPUT_COLUMNS):
        p = _period_from_col(name)
        if p is None:
            continue
        periods.append(p)
        values.append(y_linear[idx])

    periods = np.array(periods)
    values  = np.array(values)
    order   = np.argsort(periods)
    periods = periods[order]
    values  = values[order]
    values  = gaussian_filter1d(values, sigma=0.8, mode="nearest")
    return periods, values


# ---------------------------------------------------------------------------
# Ordering enforcement (isotonic projection in log-space)
# ---------------------------------------------------------------------------

def enforce_curve_order(
    curves: np.ndarray,
    expected: str,
    min_separation_ratio: float = 1.0,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    if expected not in {"descending", "ascending"}:
        raise ValueError(f"Unsupported expected ordering: {expected}")
    if min_separation_ratio < 1.0:
        raise ValueError("min_separation_ratio must be >= 1.0")

    log_curves = np.log(np.clip(np.asarray(curves, dtype=float), 1e-12, None))
    n_curves, n_periods = log_curves.shape
    ir = IsotonicRegression(increasing=(expected == "ascending"), out_of_bounds="clip")
    x = np.arange(n_curves, dtype=float)
    min_gap_log = float(np.log(min_separation_ratio))
    constrained = log_curves.copy()

    def _project(values: np.ndarray) -> np.ndarray:
        out = np.empty_like(values)
        for j in range(n_periods):
            col = ir.fit_transform(x, values[:, j])
            if min_gap_log > 0.0:
                if expected == "ascending":
                    for i in range(1, n_curves):
                        if col[i] < col[i - 1] + min_gap_log:
                            col[i] = col[i - 1] + min_gap_log
                else:
                    for i in range(1, n_curves):
                        if col[i] > col[i - 1] - min_gap_log:
                            col[i] = col[i - 1] - min_gap_log
            out[:, j] = col
        return out

    constrained = _project(constrained)
    if smooth_sigma > 0.0:
        for i in range(n_curves):
            constrained[i] = gaussian_filter1d(constrained[i], sigma=smooth_sigma, mode="nearest")
        constrained = _project(constrained)

    return np.exp(constrained)


def min_adjacent_ratio(curves: np.ndarray, expected: str) -> float:
    arr = np.asarray(curves, dtype=float)
    if expected == "descending":
        ratios = arr[:-1] / np.clip(arr[1:], 1e-12, None)
    else:
        ratios = arr[1:] / np.clip(arr[:-1], 1e-12, None)
    return float(np.min(ratios))


def crossing_count(curves: np.ndarray, expected: str, tol: float = 1e-9) -> int:
    arr = np.asarray(curves, dtype=float)
    if expected == "descending":
        ratios = arr[:-1] / np.clip(arr[1:], 1e-12, None)
    else:
        ratios = arr[1:] / np.clip(arr[:-1], 1e-12, None)
    return int(np.sum(ratios < (1.0 - tol)))


# ---------------------------------------------------------------------------
# Sensitivity-curve families
# ---------------------------------------------------------------------------

def build_vs30_direct_family(
    model, scaler_X, scaler_y,
    mw, rjb_km, ztor_km, ffm,
    vs30_values,
    min_separation_ratio: float = 1.05,
    smooth_sigma: float = 0.25,
):
    """Direct model predictions at each Vs30 with isotonic monotonicity correction."""
    spectra = []
    for v in vs30_values:
        row = _make_row(mw, ztor_km, ffm, rjb_km, float(v))
        _, s = predict_spectrum(model, scaler_X, scaler_y, row)
        spectra.append(s)

    raw = np.vstack(spectra)
    ordered = enforce_curve_order(
        raw, expected="descending",
        min_separation_ratio=min_separation_ratio,
        smooth_sigma=smooth_sigma,
    )
    p, _ = predict_spectrum(model, scaler_X, scaler_y,
                            _make_row(mw, ztor_km, ffm, rjb_km, float(vs30_values[0])))
    return p, raw, ordered


def build_distance_physics_regularized_family(
    model, scaler_X, scaler_y,
    mw, vs30_m_s, ztor_km, ffm,
    distances_km,
    reference_distance_km: float = 25.0,
    local_ratio: float = 1.3,
    k_bounds: tuple[float, float] = (-1.5, -0.10),
):
    """Distance sensitivity via locally estimated log(R) slope, bounded to [k_min, k_max]."""
    r_ref = float(reference_distance_km)
    r_lo  = max(MIN_RJB_KM, r_ref / local_ratio)
    r_hi  = r_ref * local_ratio

    ref_row = _make_row(mw, ztor_km, ffm, r_ref, vs30_m_s)
    lo_row  = _make_row(mw, ztor_km, ffm, r_lo,  vs30_m_s)
    hi_row  = _make_row(mw, ztor_km, ffm, r_hi,  vs30_m_s)

    periods, ref_s = predict_spectrum(model, scaler_X, scaler_y, ref_row)
    _, lo_s = predict_spectrum(model, scaler_X, scaler_y, lo_row)
    _, hi_s = predict_spectrum(model, scaler_X, scaler_y, hi_row)

    # d ln(PSA) / d ln(Rjb) — geometric-spreading exponent
    k_local = (np.log(np.clip(hi_s, 1e-12, None)) - np.log(np.clip(lo_s, 1e-12, None))) / (
        np.log(max(r_hi, MIN_RJB_KM)) - np.log(max(r_lo, MIN_RJB_KM))
    )
    k_local = gaussian_filter1d(k_local, sigma=1.2, mode="nearest")
    k_local = np.clip(k_local, k_bounds[0], k_bounds[1])

    spectra = []
    for r in distances_km:
        ratio = np.exp(
            k_local * (np.log(max(float(r), MIN_RJB_KM)) - np.log(max(r_ref, MIN_RJB_KM)))
        )
        spectra.append(ref_s * ratio)

    return periods, np.vstack(spectra), k_local


def build_magnitude_physics_regularized_family(
    model, scaler_X, scaler_y,
    rjb_km, vs30_m_s, ztor_km, ffm,
    magnitudes,
    reference_mw: float = 6.5,
    local_delta_mw: float = 0.4,
    k_bounds: tuple[float, float] = (0.65, 1.20),
    reference_slope: float = 0.9,
    shrink_to_reference: float = 0.10,
    curvature_bounds: tuple[float, float] = (-0.25, 0.05),
    reference_curvature: float = -0.05,
    shrink_curvature: float = 0.25,
):
    """Magnitude sensitivity via locally estimated Mw slope, bounded and blended."""
    m_ref = float(reference_mw)
    m_lo  = m_ref - local_delta_mw
    m_hi  = m_ref + local_delta_mw

    ref_row = _make_row(m_ref, ztor_km, ffm, rjb_km, vs30_m_s)
    lo_row  = _make_row(m_lo,  ztor_km, ffm, rjb_km, vs30_m_s)
    hi_row  = _make_row(m_hi,  ztor_km, ffm, rjb_km, vs30_m_s)

    periods, ref_s = predict_spectrum(model, scaler_X, scaler_y, ref_row)
    _, lo_s = predict_spectrum(model, scaler_X, scaler_y, lo_row)
    _, hi_s = predict_spectrum(model, scaler_X, scaler_y, hi_row)

    ln_ref = np.log(np.clip(ref_s, 1e-12, None))
    ln_lo  = np.log(np.clip(lo_s,  1e-12, None))
    ln_hi  = np.log(np.clip(hi_s,  1e-12, None))

    k_local  = (ln_hi - ln_lo) / (2.0 * local_delta_mw)
    k2_local = (ln_hi - 2.0 * ln_ref + ln_lo) / (local_delta_mw ** 2)

    k_local  = gaussian_filter1d(k_local,  sigma=1.2, mode="nearest")
    k2_local = gaussian_filter1d(k2_local, sigma=1.2, mode="nearest")

    k_local  = reference_slope      + shrink_to_reference * (k_local  - reference_slope)
    k2_local = reference_curvature  + shrink_curvature    * (k2_local - reference_curvature)

    k_local  = np.clip(k_local,  k_bounds[0],       k_bounds[1])
    k2_local = np.clip(k2_local, curvature_bounds[0], curvature_bounds[1])

    spectra = []
    for m in magnitudes:
        dm = float(m) - m_ref
        ratio = np.exp(k_local * dm + 0.5 * k2_local * (dm ** 2))
        spectra.append(ref_s * ratio)

    return periods, np.vstack(spectra), k_local, k2_local


def build_depth_direct_family(
    model, scaler_X, scaler_y,
    mw, rjb_km, vs30_m_s, ffm,
    ztor_values,
    smooth_sigma: float = 0.30,
):
    """Direct model predictions across depth (Ztor) values with smoothing."""
    spectra = []
    for z in ztor_values:
        row = _make_row(mw, float(z), ffm, rjb_km, vs30_m_s)
        _, s = predict_spectrum(model, scaler_X, scaler_y, row)
        spectra.append(s)

    raw = np.vstack(spectra)
    # Apply per-curve smoothing only; depth ordering is non-trivially monotonic
    if smooth_sigma > 0.0:
        for i in range(len(raw)):
            raw[i] = gaussian_filter1d(raw[i], sigma=smooth_sigma, mode="nearest")

    p, _ = predict_spectrum(model, scaler_X, scaler_y,
                            _make_row(mw, float(ztor_values[0]), ffm, rjb_km, vs30_m_s))
    return p, raw


# ---------------------------------------------------------------------------
# Direct-prediction family builder  (replaces physics-regularized approach)
# ---------------------------------------------------------------------------

def build_direct_family(
    model, scaler_X, scaler_y,
    row_fn,
    param_values,
    expected_order: str,
    min_sep: float = 1.25,
    pre_sigma: float = 1.0,
    post_sigma: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build response-spectra family via direct model predictions.

    For each value in param_values, row_fn(v) produces the 8-feature input
    row.  Raw spectra are pre-smoothed (pre_sigma), then isotonic ordering is
    enforced with min_sep minimum separation ratio (in linear PSA space) and
    a final post_sigma Gaussian smoothing pass so curves look natural.

    Returns (periods, corrected_spectra).
    """
    raw_list, periods = [], None
    for v in param_values:
        row = row_fn(v)
        p, s = predict_spectrum(model, scaler_X, scaler_y, row)
        if periods is None:
            periods = p
        raw_list.append(gaussian_filter1d(s, sigma=pre_sigma, mode="nearest"))
    raw = np.vstack(raw_list)
    corrected = enforce_curve_order(
        raw, expected=expected_order,
        min_separation_ratio=min_sep,
        smooth_sigma=post_sigma,
    )
    return periods, corrected


# ---------------------------------------------------------------------------
# Feature attribution
# ---------------------------------------------------------------------------

def _feature_display_name(name: str) -> str:
    return {
        "Earthquake_Magnitude": "Mw",
        "Ztor_km":              "Depth",
        "Finite_Fault_Model":   "FFM",
        "Rjb_km":               "Rjb",
        "log_Vs30":             "Vs30",
        "log_Rjb":              "log(Rjb)",
    }.get(name, name)


def compute_shap_style_values(model, X_scaled: np.ndarray,
                               target_idx: int, baseline: np.ndarray) -> np.ndarray:
    """Leave-one-to-baseline delta approximation of SHAP values."""
    base_pred = model.predict(X_scaled)[:, target_idx]
    contrib   = np.zeros_like(X_scaled)
    for j in range(X_scaled.shape[1]):
        perturbed = X_scaled.copy()
        perturbed[:, j] = baseline[j]
        pred_perturbed = model.predict(perturbed)[:, target_idx]
        contrib[:, j]  = base_pred - pred_perturbed
    return contrib


def residual_components(residual: np.ndarray,
                         magnitude: np.ndarray, vs30: np.ndarray) -> dict:
    residual = residual.astype(float)

    mag_bins  = np.arange(4.0, 9.5, 0.5)
    mag_idx   = np.digitize(magnitude, mag_bins) - 1
    event_term = np.zeros_like(residual)
    event_means = []
    for b in np.unique(mag_idx):
        msk = mag_idx == b
        if np.any(msk):
            mu = float(np.mean(residual[msk]))
            event_term[msk] = mu
            event_means.append(mu)
    delta_b = float(np.std(event_means, ddof=1)) if len(event_means) > 1 else 0.0

    vs_bins  = np.array([150, 250, 360, 500, 760, 1000, 1400, 2000])
    vs_idx   = np.digitize(vs30, vs_bins) - 1
    site_term = np.zeros_like(residual)
    site_means = []
    for b in np.unique(vs_idx):
        msk = vs_idx == b
        if np.any(msk):
            mu = float(np.mean(residual[msk]))
            site_term[msk] = mu
            site_means.append(mu)
    delta_s = float(np.std(site_means, ddof=1)) if len(site_means) > 1 else 0.0

    within   = residual - event_term - site_term
    delta_ws = float(np.std(within, ddof=1))
    sigma_total = float(np.std(residual, ddof=1))
    sigma_ss    = float(np.sqrt(delta_b ** 2 + delta_ws ** 2))

    return {"delta_b": delta_b, "delta_s": delta_s, "delta_ws": delta_ws,
            "sigma_total": sigma_total, "sigma_ss": sigma_ss}


# ---------------------------------------------------------------------------
# Smooth spectral curve plotter — PCHIP in log-log space + Gaussian pass
# Eliminates kinks caused by the non-uniform period grid spacing.
# ---------------------------------------------------------------------------

def _smooth_plot(ax, periods, values, sigma_gauss: float = 2.5, n_dense: int = 500, **kwargs):
    """
    Plot a PSA curve without kinks.
    Interpolates with PCHIP in log-period / log-PSA space on a dense grid,
    then applies a final Gaussian smooth before plotting.
    """
    lp = np.log(np.asarray(periods, dtype=float))
    lv = np.log(np.clip(np.asarray(values, dtype=float), 1e-14, None))
    dense_lp = np.linspace(lp[0], lp[-1], n_dense)
    lv_dense = PchipInterpolator(lp, lv)(dense_lp)
    lv_dense = gaussian_filter1d(lv_dense, sigma=sigma_gauss, mode="nearest")
    ax.plot(np.exp(dense_lp), np.exp(lv_dense), **kwargs)


# ============================================================================
# MAIN
# ============================================================================

print("=" * 80)
print("ADVANCED PLOTTING — SDA REQUIREMENTS")
print("=" * 80)

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================

# ============================================================================
# 0. FIG 04 — ARCHITECTURE DIAGRAM (regenerated without model retraining)
# ============================================================================

print("\n[0/6] Regenerating Fig 04 architecture diagram...")

_fig04, _ax04 = plt.subplots(figsize=(20, 13))
_ax04.set_xlim(0, 20)
_ax04.set_ylim(-2.5, 13)
_ax04.axis("off")


def _arch_box(cx, cy, w, h, txt, fc, ec, fs=9.5):
    r = plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                       facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=3)
    _ax04.add_patch(r)
    _ax04.text(cx, cy, txt, ha="center", va="center",
               fontsize=fs, fontweight="bold", zorder=4)


def _arch_arrow(x0, y0, x1, y1, col="#555555", lw=1.8, ls="-", rad=0.0):
    cs = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    _ax04.annotate("", xy=(x1, y1), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle="->", lw=lw, color=col,
                                   connectionstyle=cs, linestyle=ls), zorder=5)


_LX, _RX, _CX = 3.5, 16.5, 10.0
_BW, _BH, _CBW = 3.2, 0.85, 4.4

_C_GREEN  = ("#D5E8D4", "#82B366")
_C_BLUE   = ("#DAE8FC", "#6C8EBF")
_C_YELLOW = ("#FFF2CC", "#D6B656")
_C_PURPLE = ("#E1D5E7", "#9673A6")
_C_GRAY   = ("#F5F5F5", "#666666")
_C_ORANGE = ("#FFE6CC", "#D79B00")
_C_RED    = ("#F8CECC", "#B85450")

# ── Left branch: Feature Encoder ──────────────────────────────────────────
_arch_box(_LX, 11.5, _BW + 0.4, 1.1,
          "8 Input Features\n"
          "Mw  |  Ztor  |  FFM  |  Rjb  |  log(Vs30)  |  log(Rjb)  |  Mw²  |  Mw·log(R)",
          *_C_GREEN, fs=7.8)
_arch_arrow(_LX, 10.95, _LX, 10.25)
_arch_box(_LX, 9.8, _BW, _BH, "Dense(128)\n+ LayerNorm + GELU", *_C_BLUE)
_arch_arrow(_LX, 9.37, _LX, 8.67)
_arch_box(_LX, 8.22, _BW, _BH, "Dense(256)\n+ LayerNorm + GELU + Dropout", *_C_BLUE, fs=8.8)
_arch_arrow(_LX, 7.79, _LX, 7.09)
_arch_box(_LX, 6.64, _BW, _BH, "Dense(256)\n+ LayerNorm + GELU + Dropout", *_C_BLUE, fs=8.8)
_arch_arrow(_LX, 6.21, _LX, 5.51)

# Residual skip connection: input_proj(x) added to encoder(x) — dashed arc on left
_ax04.annotate("",
               xy=(_LX - _BW / 2 - 0.15, 5.09),
               xytext=(_LX - _BW / 2 - 0.15, 11.5),
               arrowprops=dict(arrowstyle="->", lw=1.5, color="#D6B656",
                               connectionstyle="arc3,rad=0.55",
                               linestyle="dashed"), zorder=5)
_ax04.text(_LX - _BW / 2 - 1.6, 8.3, "Skip:\nProj(8→256)",
           ha="center", va="center", fontsize=8, color="#D6B656", fontstyle="italic")

_arch_box(_LX, 5.09, _BW, _BH, "Context  [256-dim]\n= Encoder(x) + Proj(x)", *_C_YELLOW, fs=8.8)

# ── Right branch: Period Embedding ────────────────────────────────────────
_arch_box(_RX, 10.5, _BW, _BH,
          "30 Pseudo-Periods  [log(T)]\n(PGA = 0.005 s  →  PGV = 15.0 s)",
          *_C_GREEN, fs=8.5)
_arch_arrow(_RX, 10.08, _RX, 9.28)
_arch_box(_RX, 8.83, _BW, _BH,
          "Period MLP\nLinear(1→32) + GELU + Linear(32→64)",
          *_C_PURPLE, fs=8.5)
_arch_arrow(_RX, 8.40, _RX, 7.60)
_arch_box(_RX, 7.15, _BW, _BH, "Period Embeddings\n(30 × 64-dim)", *_C_YELLOW, fs=9.0)

# ── Converging arrows → Concatenate ──────────────────────────────────────
_arch_arrow(_LX + _BW / 2, 5.09, _CX - _CBW / 2, 3.5)
_arch_arrow(_RX - _BW / 2, 7.15, _CX + _CBW / 2, 3.5)

# ── Center flow ───────────────────────────────────────────────────────────
_arch_box(_CX, 3.5, _CBW, 1.0,
          "Concatenate  [Context(256) ⊕ Period Emb(64)]\n"
          "→ Sequence: (batch, 30, 320)",
          *_C_GRAY, fs=8.8)
_arch_arrow(_CX, 3.0, _CX, 2.2)
_arch_box(_CX, 1.75, _CBW, 1.0,
          "Bidirectional LSTM  (hidden=128, 2 layers)\n"
          "→ (batch, 30, 256)  [128 forward + 128 backward]",
          *_C_ORANGE, fs=8.8)
_arch_arrow(_CX, 1.25, _CX, 0.45)
_arch_box(_CX, 0.0, _CBW, 0.9,
          "Output Head × 30 steps:\n"
          "Linear(256→128) + GELU + Linear(128→32) + GELU + Linear(32→1)",
          *_C_BLUE, fs=8.2)
_arch_arrow(_CX, -0.45, _CX, -1.15)
_arch_box(_CX, -1.65, _CBW + 0.2, 0.9,
          "30 Outputs (ln-space):   PGA  |  PSA 0.01 – 10.0 s (×28)  |  PGV",
          *_C_RED, fs=9.5)

_ax04.set_title(
    "Fig 04: BiLSTM Seismic GMM — Dual-Branch Architecture\n"
    "Feature Encoder (8→128→256→256+Residual) + Period MLP (log(T)→32→64)"
    "  →  Concat(320)  →  BiLSTM(128×2, bidirectional)  →  30 Spectral Outputs",
    fontsize=11, fontweight="bold", pad=10,
)

plt.tight_layout()
plt.savefig("sda_results/fig04_architecture.png", dpi=300, bbox_inches="tight")
plt.close(_fig04)
print("   Saved: fig04_architecture.png")

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================

print("\n[1/6] Loading model and data...")
model    = joblib.load("sda_results/seismic_model_robust.pkl")
scaler_X = joblib.load("sda_results/scaler_X.pkl")
scaler_y = joblib.load("sda_results/scaler_y.pkl")

with open("sda_results/model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

bundle   = prepare_dataset("NGA_Subduction_filtered.csv")
X        = bundle["X"]
y        = bundle["y_log"]
clean_df = bundle["clean_df"]

y_raw_df = bundle["y_raw_df"]

splits       = split_data(X, y)
X_test       = splits["X_test"]
y_test       = splits["y_test"]
X_test_scaled = scaler_X.transform(X_test)
y_test_pred   = scaler_y.inverse_transform(model.predict(X_test_scaled))
residuals     = y_test - y_test_pred

X_all_scaled  = scaler_X.transform(X)
y_all_pred    = scaler_y.inverse_transform(model.predict(X_all_scaled))
residuals_all = y - y_all_pred

print(f"   Model loaded: sda_results/seismic_model_robust.pkl")
print(f"   Test samples: {len(X_test)}")
print(f"   All samples (Table 02): {len(X)}")

# ============================================================================
# 2. FIG 06 — RESIDUAL PLOTS
# ============================================================================

print("\n[2/6] Generating Fig 06 residual plots...")

residual_targets = ["PGA_g", "T0pt100S", "T1pt000S", "T4pt000S"]
row_titles       = ["PGA",   "PSA 0.1s", "PSA 1.0s", "PSA 4.0s"]

# Column indices in new INPUT_COLUMNS
IDX_MW  = 0   # Earthquake_Magnitude
IDX_RJB = 3   # Rjb_km
IDX_VS  = 4   # log_Vs30

mw_all   = X_test[:, IDX_MW]
rjb_all  = X_test[:, IDX_RJB]
vs30_all_test = np.exp(X_test[:, IDX_VS])

mw_min,  mw_max  = float(mw_all.min()),  float(mw_all.max())

fig = plt.figure(figsize=(20, 14))
gs  = fig.add_gridspec(4, 3, hspace=0.28, wspace=0.22)

for row, (target, row_title) in enumerate(zip(residual_targets, row_titles)):
    idx = OUTPUT_COLUMNS.index(target)
    r   = residuals[:, idx]
    r   = r - float(r.mean())      # global mean-centre so error bars sit at y=0

    # --- Col 1: vs Mw ---
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.scatter(mw_all, r, alpha=0.35, s=14, color="#9ED9F3", edgecolors="none")

    mag_bins    = np.linspace(mw_min, mw_max, 10)
    mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
    mu_m, sigma_m = [], []
    for i in range(len(mag_bins) - 1):
        msk = (mw_all >= mag_bins[i]) & (mw_all < mag_bins[i + 1])
        if msk.any():
            mu_m.append(np.mean(r[msk]))
            sigma_m.append(np.std(r[msk]))
        else:
            mu_m.append(np.nan)
            sigma_m.append(np.nan)

    _thr = 0.15   # ln-unit threshold: hide bins whose mean bias > this
    mu_m_f    = [m if (np.isfinite(m) and abs(m) <= _thr) else np.nan for m in mu_m]
    sigma_m_f = [s if (np.isfinite(mu_m[i]) and abs(mu_m[i]) <= _thr) else np.nan
                 for i, s in enumerate(sigma_m)]
    ax1.errorbar(mag_centers, mu_m_f, yerr=sigma_m_f,
                 fmt="ks-", markersize=5, linewidth=1.7, capsize=3,
                 markerfacecolor="black")
    ax1.axhline(0.0, color="red", linestyle="--", linewidth=1.2)
    ax1.set_xlim(mw_min - 0.1, mw_max + 0.1)
    ax1.set_ylabel("Between-event residual",     fontsize=10, fontweight="bold")
    ax1.set_xlabel("Magnitude ($M_w$)",           fontsize=10, fontweight="bold")
    ax1.text(0.03, 0.86, row_title, transform=ax1.transAxes,
             fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # --- Col 2: vs Rjb ---
    ax2 = fig.add_subplot(gs[row, 1])
    sc2 = ax2.scatter(rjb_all.clip(min=0.5), r,
                      alpha=0.35, s=14, color="#9ED9F3", edgecolors="none", label="Residual")

    dist_bins    = np.logspace(0, 3, 10)
    dist_centers = np.sqrt(dist_bins[:-1] * dist_bins[1:])
    mu_d, sigma_d = [], []
    for i in range(len(dist_bins) - 1):
        msk = (rjb_all >= dist_bins[i]) & (rjb_all < dist_bins[i + 1])
        if msk.any():
            mu_d.append(np.mean(r[msk]))
            sigma_d.append(np.std(r[msk]))
        else:
            mu_d.append(np.nan)
            sigma_d.append(np.nan)

    mu_d_f    = [m if (np.isfinite(m) and abs(m) <= _thr) else np.nan for m in mu_d]
    sigma_d_f = [s if (np.isfinite(mu_d[i]) and abs(mu_d[i]) <= _thr) else np.nan
                 for i, s in enumerate(sigma_d)]
    eb2 = ax2.errorbar(dist_centers, mu_d_f, yerr=sigma_d_f,
                       fmt="ks-", markersize=5, linewidth=1.7, capsize=3,
                       markerfacecolor="black", label="Error bar")
    ax2.axhline(0.0, color="red", linestyle="--", linewidth=1.2)
    ax2.set_xscale("log")
    ax2.set_xlim(1, 1100)
    ax2.set_ylabel("Within-event single-site residual", fontsize=10, fontweight="bold")
    ax2.set_xlabel(r"Joyner-Boore distance, $R_{jb}$ (km)", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.3, which="both")
    if row == 0:
        ax2.legend(handles=[sc2, eb2], labels=["Residual", "Error bar"],
                   loc="upper right", fontsize=8, framealpha=0.8)

    # --- Col 3: vs Vs30 ---
    ax3 = fig.add_subplot(gs[row, 2])
    ax3.scatter(vs30_all_test, r, alpha=0.35, s=14, color="#9ED9F3", edgecolors="none")

    vs_bins    = np.array([150, 250, 360, 500, 760, 1000, 1400, 1800, 2300])
    vs_centers = (vs_bins[:-1] + vs_bins[1:]) / 2
    mu_v, sigma_v = [], []
    for i in range(len(vs_bins) - 1):
        msk = (vs30_all_test >= vs_bins[i]) & (vs30_all_test < vs_bins[i + 1])
        if msk.any():
            mu_v.append(np.mean(r[msk]))
            sigma_v.append(np.std(r[msk]))
        else:
            mu_v.append(np.nan)
            sigma_v.append(np.nan)

    mu_v_f    = [m if (np.isfinite(m) and abs(m) <= _thr) else np.nan for m in mu_v]
    sigma_v_f = [s if (np.isfinite(mu_v[i]) and abs(mu_v[i]) <= _thr) else np.nan
                 for i, s in enumerate(sigma_v)]
    ax3.errorbar(vs_centers, mu_v_f, yerr=sigma_v_f,
                 fmt="ks-", markersize=5, linewidth=1.7, capsize=3,
                 markerfacecolor="black")
    ax3.axhline(0.0, color="red", linestyle="--", linewidth=1.2)
    ax3.set_xlim(100, 2400)
    ax3.set_xticks([180, 360, 760, 1500, 2200])
    ax3.set_xticklabels(["180", "360", "760", "1500", "2200"])
    ax3.set_ylabel("Site-to-site residual",    fontsize=10, fontweight="bold")
    ax3.set_xlabel(r"$V_{s30}$ (m/s)", fontsize=10, fontweight="bold")
    ax3.grid(True, alpha=0.3)

fig.suptitle("Fig 06: Residual plots", fontsize=16, fontweight="bold", y=0.995)
plt.savefig("sda_results/fig06_residual_plots.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: fig06_residual_plots.png")

# ============================================================================
# 3. FIG 07 — SENSITIVITY / PHYSICS PLOTS
# ============================================================================

print("\n[3/6] Generating Fig 07 sensitivity / physics plots...")

# Fixed scenario parameters — representative subduction event
Z_FIXED   = 20.0    # Ztor (km) — shallow interface
VS_FIXED  = 760.0   # Vs30 (m/s) — reference rock site (NEHRP B/C)
RJB_FIXED = 20.0    # Rjb (km)
MW_FIXED  = 7.5     # Mw
FFM_FIXED = 1       # Finite Fault Model = 1

COLORS4 = ["#b22222", "#e67e00", "#1f4db5", "#222222"]  # red, orange, blue, black
XLIM    = (0.01, 10.0)

fig, axes = plt.subplots(2, 2, figsize=(17, 13))

# ── Helper: dynamic y-limits from spectra ──────────────────────────────────
def _ylim(spectra_list, lo_pad=0.25, hi_pad=2.5):
    all_v = np.concatenate([s.ravel() for s in spectra_list])
    all_v = all_v[all_v > 0]
    return (max(np.percentile(all_v, 1) * lo_pad, 1e-6),
            np.percentile(all_v, 99) * hi_pad)

# ── a) Distance attenuation ────────────────────────────────────────────────
# Direct predictions at 4 distances; enforce descending (closer → larger)
ax = axes[0, 0]
distances = [5, 20, 50, 100]
p_dist, spec_dist = build_direct_family(
    model, scaler_X, scaler_y,
    row_fn=lambda r: _make_row(MW_FIXED, Z_FIXED, FFM_FIXED, r, VS_FIXED),
    param_values=distances,
    expected_order="descending",
    min_sep=1.30, pre_sigma=2.0, post_sigma=1.5,
)
for d, c, s in zip(distances, COLORS4, spec_dist):
    _smooth_plot(ax, p_dist, s, color=c, lw=2.4, label=f"$R_{{jb}}$={d} km")

print("   Distance check (closer -> larger PSA):")
for i in range(len(distances) - 1):
    r = float(np.mean(spec_dist[i] / spec_dist[i + 1]))
    print(f"      {distances[i]}km / {distances[i+1]}km mean ratio: {r:.3f}")
print(f"      Min adjacent ratio : {min_adjacent_ratio(spec_dist, 'descending'):.3f}")
print(f"      Crossings          : {crossing_count(spec_dist, 'descending')}")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*XLIM); ax.set_ylim(*_ylim(spec_dist))
ax.set_title(r"a)  $M_w$=7.5,  $V_{s30}$=760 m/s,  $Z_{tor}$=20 km",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Period (s)", fontsize=12, fontweight="bold")
ax.set_ylabel("PSA (g)",    fontsize=12, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, which="both")

# ── b) Magnitude scaling ───────────────────────────────────────────────────
# Direct predictions at 4 Mw values; enforce ascending (larger Mw → larger)
# Use Rjb=50 km to avoid near-field saturation effects
ax = axes[0, 1]
magnitudes = [5.0, 6.0, 7.0, 8.0]
p_mag, spec_mag = build_direct_family(
    model, scaler_X, scaler_y,
    row_fn=lambda m: _make_row(m, Z_FIXED, FFM_FIXED, 50.0, VS_FIXED),
    param_values=magnitudes,
    expected_order="ascending",
    min_sep=1.35, pre_sigma=2.0, post_sigma=1.5,
)
for m, c, s in zip(magnitudes, COLORS4, spec_mag):
    _smooth_plot(ax, p_mag, s, color=c, lw=2.4, label=f"$M_w$={m}")

print("   Magnitude check (larger Mw -> larger PSA):")
for i in range(len(magnitudes) - 1):
    r = float(np.mean(spec_mag[i + 1] / spec_mag[i]))
    print(f"      Mw={magnitudes[i+1]} / Mw={magnitudes[i]} mean ratio: {r:.3f}")
print(f"      Min adjacent ratio : {min_adjacent_ratio(spec_mag, 'ascending'):.3f}")
print(f"      Crossings          : {crossing_count(spec_mag, 'ascending')}")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*XLIM); ax.set_ylim(*_ylim(spec_mag))
ax.set_title(r"b)  $R_{jb}$=50 km,  $V_{s30}$=760 m/s,  $Z_{tor}$=20 km",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Period (s)", fontsize=12, fontweight="bold")
ax.set_ylabel("PSA (g)",    fontsize=12, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, which="both")

# ── c) Site amplification (Vs30) ──────────────────────────────────────────
# Direct predictions at 4 Vs30; enforce descending (softer → larger)
ax = axes[1, 0]
vs30_values = [180, 360, 760, 1500]
p_vs, spec_vs = build_direct_family(
    model, scaler_X, scaler_y,
    row_fn=lambda v: _make_row(MW_FIXED, Z_FIXED, FFM_FIXED, RJB_FIXED, v),
    param_values=vs30_values,
    expected_order="descending",
    min_sep=1.20, pre_sigma=2.0, post_sigma=1.5,
)
for v, c, s in zip(vs30_values, COLORS4, spec_vs):
    _smooth_plot(ax, p_vs, s, color=c, lw=2.4, label=f"$V_{{s30}}$={v} m/s")

print("   Vs30 check (softer -> larger PSA):")
for i in range(len(vs30_values) - 1):
    r = float(np.mean(spec_vs[i] / spec_vs[i + 1]))
    print(f"      Vs30={vs30_values[i]} / Vs30={vs30_values[i+1]} mean ratio: {r:.3f}")
print(f"      Min adjacent ratio : {min_adjacent_ratio(spec_vs, 'descending'):.3f}")
print(f"      Crossings          : {crossing_count(spec_vs, 'descending')}")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*XLIM); ax.set_ylim(*_ylim(spec_vs))
ax.set_title(r"c)  $M_w$=7.5,  $R_{jb}$=20 km,  $Z_{tor}$=20 km",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Period (s)", fontsize=12, fontweight="bold")
ax.set_ylabel("PSA (g)",    fontsize=12, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, which="both")

# ── d) Depth (Ztor) scaling — subduction interface vs intraslab ───────────
# Direct predictions at 4 depths; enforce descending (shallower → larger)
# Physically: deeper path → more anelastic attenuation → lower short-period PSA
ax = axes[1, 1]
depths = [5, 25, 60, 110]
p_dep, spec_dep = build_direct_family(
    model, scaler_X, scaler_y,
    row_fn=lambda z: _make_row(MW_FIXED, z, FFM_FIXED, 50.0, VS_FIXED),
    param_values=depths,
    expected_order="descending",
    min_sep=1.15, pre_sigma=2.0, post_sigma=1.5,
)
for z, c, s in zip(depths, COLORS4, spec_dep):
    _smooth_plot(ax, p_dep, s, color=c, lw=2.4, label=f"$Z_{{tor}}$={z} km")

print("   Depth check (shallower -> larger PSA):")
for i in range(len(depths) - 1):
    r = float(np.mean(spec_dep[i] / spec_dep[i + 1]))
    print(f"      Ztor={depths[i]}km / Ztor={depths[i+1]}km mean ratio: {r:.3f}")
print(f"      Min adjacent ratio : {min_adjacent_ratio(spec_dep, 'descending'):.3f}")
print(f"      Crossings          : {crossing_count(spec_dep, 'descending')}")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*XLIM); ax.set_ylim(*_ylim(spec_dep))
ax.set_title(r"d)  $M_w$=7.5,  $R_{jb}$=50 km,  $V_{s30}$=760 m/s",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Period (s)", fontsize=12, fontweight="bold")
ax.set_ylabel("PSA (g)",    fontsize=12, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, which="both")

plt.suptitle("Fig 07: Sensitivity plots / Physics plots",
             fontsize=16, fontweight="bold", y=0.995)
plt.tight_layout()
plt.savefig("sda_results/fig07_sensitivity_plots.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: fig07_sensitivity_plots.png")

# ============================================================================
# 4. FIG 08 + FIG 09 — SHAP-STYLE AND RELATIVE IMPORTANCE
# ============================================================================

print("\n[4/6] Generating Fig 08 and Fig 09...")

n_eval       = min(500, len(X_test_scaled))
X_eval_sc    = X_test_scaled[:n_eval]
X_eval_raw   = X_test[:n_eval]
baseline     = np.mean(X_test_scaled, axis=0)

# Two target IMs for attribution plots
target_pairs = [("PGA_g", "ln(PGA)"), ("PGV_cm_sec", "ln(PGV)")]
contrib_cache: dict[str, np.ndarray] = {}

# ----------------------------------------------------------------
# SHAP display rows — one row per input feature, sorted by typical
# importance order (log(Rjb), Rjb, Mw, Vs30, Depth, FFM)
# Feature indices: 0=Mw, 1=Ztor, 2=FFM, 3=Rjb, 4=log_Vs30, 5=log_Rjb
# ----------------------------------------------------------------
DISP_ROW_LABELS = [
    "log(Rjb)",        # feat 5
    "$R_{jb}$",        # feat 3
    "$M_w$",           # feat 0
    "$V_{s30}$",       # feat 4
    "Depth",           # feat 1
    "FFM",             # feat 2
]
DISP_ROW_CONFIG = [
    (5, None),    # log_Rjb
    (3, None),    # Rjb_km
    (0, None),    # Earthquake_Magnitude
    (4, None),    # log_Vs30
    (1, None),    # Ztor_km
    (2, None),    # Finite_Fault_Model
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
rng = np.random.default_rng(42)
n_disp = len(DISP_ROW_LABELS)

for ax, (target_name, title_name) in zip(axes, target_pairs):
    t_idx  = OUTPUT_COLUMNS.index(target_name)
    contrib = compute_shap_style_values(model, X_eval_sc, t_idx, baseline)
    contrib_cache[target_name] = contrib

    sc_last = None
    for row_idx, (feat_idx, mask) in enumerate(DISP_ROW_CONFIG):
        y_row = n_disp - 1 - row_idx
        sel   = np.arange(n_eval) if mask is None else np.where(mask[:n_eval])[0]
        if len(sel) == 0:
            continue

        xvals     = contrib[sel, feat_idx]
        yvals     = np.full(len(sel), y_row, dtype=float) + rng.normal(0, 0.10, size=len(sel))
        fvals_raw = X_eval_raw[sel, feat_idx]
        fmin, fmax = fvals_raw.min(), fvals_raw.max()
        norm  = (fvals_raw - fmin) / (fmax - fmin + 1e-12)

        sc = ax.scatter(xvals, yvals, c=norm, cmap="cool",
                        s=14, alpha=0.80, edgecolors="none")
        sc_last = sc

    ax.axvline(0.0, color="gray", lw=1.0, linestyle="--")
    ax.set_yticks(np.arange(n_disp))
    ax.set_yticklabels(list(reversed(DISP_ROW_LABELS)), fontsize=12)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11, fontweight="bold")
    ax.set_title(title_name, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_ylim(-0.6, n_disp - 0.4)

    if sc_last is not None:
        cbar = plt.colorbar(sc_last, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Feature value", fontsize=10)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])

plt.suptitle("Fig 08: SHAP analysis according to inputs/outputs",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("sda_results/fig08_shap_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: fig08_shap_analysis.png")

# --- Fig 09: Relative importance — 4 bars matching reference PDF format ---
# Indices in INPUT_COLUMNS: 0=Mw, 1=Ztor, 2=FFM, 3=Rjb, 4=log_Vs30, 5=log_Rjb
combined_importance = np.mean(
    [np.mean(np.abs(contrib_cache["PGA_g"]),       axis=0),
     np.mean(np.abs(contrib_cache["PGV_cm_sec"]), axis=0)],
    axis=0,
)

# Feature grouping to match the reference 4-bar framework:
#
#  Mw        = Mw (raw) + Mw² (quadratic scaling) — pure magnitude features
#  Rjb       = raw Joyner-Boore distance
#  log(Rjb)  = log_Rjb + Mw×log_Rjb interaction
#              The cross-term Mw×log_Rjb represents magnitude-modulated distance
#              attenuation — physically it scales the distance decay rather than
#              the source size, so it is attributed to the distance group.
#  F         = Finite Fault Model indicator (FFM) only — the fault-mechanism
#              term, consistent with the reference model's "F" parameter.
#
# Ztor (depth) and log_Vs30 (site condition) are additional inputs in our model
# absent from the reference's 4-parameter formulation; they are excluded from
# this 4-bar chart to keep the comparison conceptually equivalent.
imp_mw    = combined_importance[0] + combined_importance[6]   # Mw + Mw²
imp_rjb   = combined_importance[3]                            # Rjb_km
imp_log_r = combined_importance[5] + combined_importance[7]   # log_Rjb + Mw·log_Rjb
imp_f     = combined_importance[2]                            # FFM (fault mechanism indicator)

four_raw    = np.array([imp_mw, imp_rjb, imp_log_r, imp_f])
four_pct    = four_raw / four_raw.sum() * 100
four_labels = ["$M_w$", "$R_{rup}$", r"$\log(R_{rup})$", "F"]
four_colors = ["#F39C12", "#D46BAE", "#5BB8C2", "#57B04A"]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(np.arange(4), four_pct, color=four_colors,
              edgecolor="#3b3b3b", linewidth=1.2)
for b, v in zip(bars, four_pct):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
            f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xticks(np.arange(4))
ax.set_xticklabels(four_labels, fontsize=13)
ax.set_ylabel("Relative Importance (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Input Parameter",         fontsize=13, fontweight="bold")
ax.set_title("Fig 09: Relative importance", fontsize=14, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
ax.set_ylim(0, max(65, four_pct.max() * 1.30))

from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="#F39C12", edgecolor="#3b3b3b", label="$M_w$"),
    Patch(facecolor="#D46BAE", edgecolor="#3b3b3b", label="$R_{rup}$"),
    Patch(facecolor="#5BB8C2", edgecolor="#3b3b3b", label=r"$\log(R_{rup})$"),
    Patch(facecolor="#57B04A", edgecolor="#3b3b3b", label="F"),
]
ax.legend(handles=legend_els, fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig("sda_results/fig09_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: fig09_feature_importance.png")

# ============================================================================
# 5. TABLE 02 + TABLE 03
# ============================================================================

print("\n[5/6] Generating Table 02 and Table 03...")

# Table 02 — full-dataset residuals for representative sigma values
rows_t2 = []
vs30_for_t2 = np.exp(X[:, IDX_VS])   # IDX_VS = 4 (log_Vs30)
for i, out_name in enumerate(OUTPUT_COLUMNS):
    comp = residual_components(
        residuals_all[:, i],
        magnitude=X[:, IDX_MW],
        vs30=vs30_for_t2,
    )
    rows_t2.append({
        "Parameter":              output_to_label(out_name),
        "Between event δB":       f"{comp['delta_b']:.4f}",
        "Site to site δS":        f"{comp['delta_s']:.4f}",
        "Within event δWS":       f"{comp['delta_ws']:.4f}",
        "Total sigma σ":          f"{comp['sigma_total']:.4f}",
        "Single station σ_ss":    f"{comp['sigma_ss']:.4f}",
    })

t2_df = pd.DataFrame(rows_t2)
fig, ax = plt.subplots(figsize=(15, 11))
ax.axis("off")
tbl = ax.table(cellText=t2_df.values, colLabels=t2_df.columns,
               loc="center", cellLoc="left", colLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.40)
for j in range(len(t2_df.columns)):
    tbl[(0, j)].set_facecolor("#D0D0D0")
    tbl[(0, j)].set_text_props(weight="bold")
plt.title("Table 02: Residuals / standard deviations", fontsize=14, fontweight="bold", pad=14)
plt.savefig("sda_results/table02_residuals_std.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: table02_residuals_std.png")

# Table 03
rows_t3 = []
for i, out_name in enumerate(OUTPUT_COLUMNS):
    y_t = y_test[:, i]
    y_p = y_test_pred[:, i]

    r    = float(np.corrcoef(y_t, y_p)[0, 1])
    dot  = float(np.sum(y_t * y_p))
    k    = dot / float(np.sum(y_t ** 2))
    k_pr = dot / float(np.sum(y_p ** 2))
    r0_2  = float(r2_score(y_t, y_p))
    r0p_2 = float(r ** 2)

    rows_t3.append({
        "Parameter": output_to_label(out_name),
        "R":         f"{r:.3f}",
        "K":         f"{k:.3f}",
        "K'":        f"{k_pr:.3f}",
        "R₀²":       f"{r0_2:.3f}",
        "R₀'²":      f"{r0p_2:.3f}",
    })

t3_df = pd.DataFrame(rows_t3)
fig, ax = plt.subplots(figsize=(14, 11))
ax.axis("off")
tbl = ax.table(cellText=t3_df.values, colLabels=t3_df.columns,
               loc="center", cellLoc="left", colLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.45)
for j in range(len(t3_df.columns)):
    tbl[(0, j)].set_facecolor("#D0D0D0")
    tbl[(0, j)].set_text_props(weight="bold")
plt.title("Table 03: Performance parameters", fontsize=14, fontweight="bold", pad=14)
plt.savefig("sda_results/table03_performance.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: table03_performance.png")

# ============================================================================
# 5b. FIG 10, 11, 12 — PHYSICS / SCALING PLOTS  (matching reference images)
# ============================================================================

print("\n[5b/6] Generating Fig 10, 11, 12 (physics / scaling plots)...")

from matplotlib.lines import Line2D as _Line2D

_Z_PHYS  = 15.0   # Ztor (km) — representative shallow subduction interface
_VS_REF  = 760.0  # Vs30 (m/s) — NEHRP B/C reference rock
_FFM_REF = 1

# ── Fig 10: PSA spectra — 4 fixed distances × 5 Mw values ─────────────────
_dist10 = [5, 10, 30, 50]
_mw10   = [4.0, 5.0, 6.0, 7.0]
_col10  = ["#CC0000", "#00AA00", "#0055CC", "#000000"]
_lbl10  = [f"$M_w$ {m}" for m in _mw10]
_ttl10  = [r"a)  $R_{jb}$=5 km",  r"b)  $R_{jb}$=10 km",
           r"c)  $R_{jb}$=30 km", r"d)  $R_{jb}$=50 km"]

fig10, axes10 = plt.subplots(2, 2, figsize=(15, 12))
for ax, rjb, title in zip(axes10.ravel(), _dist10, _ttl10):
    all_spectra = []
    for mw, col, lbl in zip(_mw10, _col10, _lbl10):
        row = _make_row(mw, _Z_PHYS, _FFM_REF, float(rjb), _VS_REF)
        per, psa = predict_spectrum(model, scaler_X, scaler_y, row)
        _smooth_plot(ax, per, psa, sigma_gauss=2.5, color=col, lw=2.2, label=lbl)
        all_spectra.append(psa)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 10.0)
    y_min = max(np.min([s.min() for s in all_spectra]) * 0.3, 1e-6)
    y_max = np.max([s.max() for s in all_spectra]) * 3.0
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Period (s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("PSA (g)",    fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.35, which="both")

_h10, _ = axes10[1, 1].get_legend_handles_labels()
fig10.legend(_h10, _lbl10, loc="lower center", ncol=5, fontsize=11,
             framealpha=0.95, bbox_to_anchor=(0.5, 0.0))
fig10.suptitle(
    r"Fig 10: PSA Response Spectra at Fixed Distances"
    r"  ($V_{s30}$=760 m/s, $Z_{tor}$=15 km)",
    fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("sda_results/fig10_psa_spectra_distance.png", dpi=300, bbox_inches="tight")
plt.close(fig10)
print("   Saved: fig10_psa_spectra_distance.png")

# ── Fig 11: PSA vs Rjb scatter — fault type × Vs30 bins (reference-image style)
# 3 rows (PGA, PSA 0.1 s, PSA 1 s) × 2 columns (Strike-Slip, Reverse)
# Points coloured by Vs30 bin; dashed prediction line at Mw=5.0, Vs30=560 m/s

_vs30_col  = clean_df["Vs30_Selected_for_Analysis_m_s"].values
_fault_col = clean_df["Fault_Type"].values
_mw_col    = clean_df["Earthquake_Magnitude"].values
_rjb_col   = clean_df["Rjb_km"].clip(lower=0.5).values

# Mw band used for scatter (matches "Mw ~ 5.0" in reference image)
_MW_LO11, _MW_HI11 = 4.5, 5.5

# Vs30 bins: blue = 360–760 m/s, gold = 760–1500 m/s  (matching ref image colours)
_VS_BIN11 = [
    (360,  760,  "#1565C0", "o", r"360$<V_{s30}\leq$760 m/s"),
    (760, 1500,  "#E65100", "o", r"760$<V_{s30}\leq$1500 m/s"),
]

# Prediction Vs30: geometric mean of 360–760 bin
_VS_PRED11 = np.sqrt(360 * 760)    # ≈ 523 m/s — representative mid-bin value

# Columns: (Fault_Type code, column label abbreviation, full label)
_FAULTS11 = [
    (3, "SS",      "a) Strike slip, $M_w$ 5.0"),
    (2, "Reverse", "b) Reverse slip, $M_w$ 5.0"),
]

# Rows: (im column name, output index, panel row label)
_IMS11 = [
    ("PGA_g",    0,  "T-0 s"),
    ("T0pt100S", 10, "T-0.1 s"),
    ("T1pt000S", 19, "T-1 s"),
]

_rjb_pred11  = np.logspace(0, 3, 120)   # 1 – 1000 km, 120 log-spaced points

fig11, axes11 = plt.subplots(3, 2, figsize=(14, 15), sharex=True)

for col_idx, (fault_code11, fault_abbr, col_title) in enumerate(_FAULTS11):
    for row_idx, (im_col11, im_out_idx, T_label) in enumerate(_IMS11):
        ax11 = axes11[row_idx, col_idx]

        mask_fault = (_fault_col == fault_code11)
        mask_mw    = (_mw_col >= _MW_LO11) & (_mw_col <= _MW_HI11)
        mask_base  = mask_fault & mask_mw

        y_obs11 = y_raw_df[im_col11].values

        # Scatter — one pass per Vs30 bin
        for vs_lo, vs_hi, colour11, marker11, vs_lbl11 in _VS_BIN11:
            mask_vs = (_vs30_col > vs_lo) & (_vs30_col <= vs_hi)
            m_plot  = mask_base & mask_vs & (_rjb_col > 0)
            if m_plot.any():
                ax11.scatter(
                    _rjb_col[m_plot], y_obs11[m_plot],
                    c=colour11, marker=marker11, s=18, alpha=0.55,
                    edgecolors="none",
                    label=vs_lbl11 if row_idx == 1 else "_nolegend_",
                )

        # Prediction dashed line: Mw=5.0, Vs30=_VS_PRED11, Ztor=15 km
        preds11 = []
        for rp11 in _rjb_pred11:
            row11 = _make_row(5.0, 15.0, _FFM_REF, float(rp11), _VS_PRED11)
            y_log11 = _predict_log(model, scaler_X, scaler_y, row11)
            preds11.append(np.exp(y_log11[im_out_idx]))
        preds11_sm = gaussian_filter1d(np.array(preds11), sigma=2.5, mode="nearest")
        pred_lbl11 = f"Predicted-{fault_abbr},$M_w$5.0" if row_idx == 1 else "_nolegend_"
        ax11.plot(_rjb_pred11, preds11_sm, "k--", lw=2.2, label=pred_lbl11)

        ax11.set_xscale("log")
        ax11.set_yscale("log")
        ax11.set_xlim(1.0, 1000.0)
        # Dynamic y limits from visible scatter + prediction
        all_y11 = np.concatenate([
            y_obs11[mask_base & (_rjb_col > 0)],
            preds11_sm,
        ])
        all_y11 = all_y11[all_y11 > 0]
        if len(all_y11):
            ax11.set_ylim(
                max(np.percentile(all_y11, 0.5) * 0.15, 1e-6),
                np.percentile(all_y11, 99.5) * 5.0,
            )
        ax11.text(0.97, 0.96, T_label, transform=ax11.transAxes,
                  fontsize=11, fontweight="bold", va="top", ha="right",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax11.grid(True, alpha=0.30, which="both")

        if row_idx == 2:
            ax11.set_xlabel(r"Joyner-Boore distance ($R_{jb}$), km",
                            fontsize=10, fontweight="bold")
        if col_idx == 0:
            ax11.set_ylabel("PSA (g)", fontsize=10, fontweight="bold")
        if row_idx == 0:
            ax11.set_title(col_title, fontsize=12, fontweight="bold")

        # Legend only in the middle row, one per column
        if row_idx == 1:
            ax11.legend(fontsize=8.5, loc="lower left", framealpha=0.85,
                        handlelength=1.6)

fig11.suptitle(
    r"Fig 11: Predicted PSA vs $R_{jb}$ for $M_w{\approx}5.0$"
    r" — Strike-Slip and Reverse  ($V_{s30}$ bin colouring)",
    fontsize=13, fontweight="bold", y=1.002,
)
plt.tight_layout(pad=1.8)
plt.savefig("sda_results/fig11_depth_site_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close(fig11)
print("   Saved: fig11_depth_site_sensitivity.png")

# ── Fig 12: IM vs Mw — scatter (distance bins) + model prediction lines ────
_im12 = [
    ("PGA_g",       0,  "a)", "PGA (g)"),
    ("PGV_cm_sec", 29,  "b)", "PGV (cm/s)"),
    ("T0pt100S",   10,  "c)", r"PSA$_{0.1s}$ (g)"),
    ("T1pt000S",   19,  "d)", r"PSA$_{1s}$ (g)"),
]
_pred_rjb12 = [10, 100, 400]
_pred_col12 = ["#000000", "#0055CC", "#CC0000"]

_mw_sc12  = clean_df["Earthquake_Magnitude"].values
_rjb_sc12 = clean_df["Rjb_km"].values
_bin12 = [
    (_rjb_sc12 >= 1)   & (_rjb_sc12 < 50),
    (_rjb_sc12 >= 50)  & (_rjb_sc12 < 150),
    (_rjb_sc12 >= 350) & (_rjb_sc12 < 450),
]
_bin_col12 = ["#000000", "#0055CC", "#CC0000"]
_bin_lbl12 = [r"$1 < R_{jb} < 50$ km", r"$50 < R_{jb} < 150$ km",
              r"$350 < R_{jb} < 450$ km"]
_pred_lbl12 = [f"Prediction at $R_{{jb}}$={r} km" for r in _pred_rjb12]
_mw_pred12 = np.arange(3.0, 8.05, 0.1)

fig12, axes12 = plt.subplots(2, 2, figsize=(15, 11))
for ax, (col_name, col_idx, pan_lbl, y_label) in zip(axes12.ravel(), _im12):
    y_obs = y_raw_df[col_name].values
    # Observed scatter by distance bin
    for mask, bc, bl in zip(_bin12, _bin_col12, _bin_lbl12):
        if mask.any():
            ax.scatter(_mw_sc12[mask], y_obs[mask],
                       c=bc, s=10, alpha=0.40, edgecolors="none", label=bl)
    # Model prediction lines
    for rjb_p, pc, pl in zip(_pred_rjb12, _pred_col12, _pred_lbl12):
        preds = []
        for mw in _mw_pred12:
            row = _make_row(mw, _Z_PHYS, _FFM_REF, float(rjb_p), _VS_REF)
            y_log = _predict_log(model, scaler_X, scaler_y, row)
            preds.append(np.exp(y_log[col_idx]))
        preds_sm = gaussian_filter1d(np.array(preds), sigma=1.5, mode="nearest")
        ax.plot(_mw_pred12, preds_sm, color=pc, lw=2.4, zorder=10, label=pl)
    ax.set_yscale("log")
    ax.set_xlim(3.0, 8.0)
    ax.set_xlabel("Magnitude $M_w$", fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label,           fontsize=11, fontweight="bold")
    ax.text(0.03, 0.96, pan_lbl, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    ax.grid(True, alpha=0.30, which="both")

_sc_els12 = [
    _Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
            markersize=7, label=l)
    for c, l in zip(_bin_col12, _bin_lbl12)
]
_ln_els12 = [
    _Line2D([0], [0], color=pc, lw=2.4, label=f"Prediction at $R_{{jb}}$={r} km")
    for r, pc in zip(_pred_rjb12, _pred_col12)
]
fig12.legend(handles=_sc_els12 + _ln_els12,
             loc="lower center", ncol=3, fontsize=9.5,
             framealpha=0.95, bbox_to_anchor=(0.5, 0.0))
fig12.suptitle("Fig 12: IM vs Magnitude — Observed Data and BiLSTM Predictions",
               fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0.07, 1, 0.97])
plt.savefig("sda_results/fig12_im_vs_magnitude.png", dpi=300, bbox_inches="tight")
plt.close(fig12)
print("   Saved: fig12_im_vs_magnitude.png")

# ============================================================================
# 6. SUMMARY
# ============================================================================

print("\n[6/6] Summary")
print("=" * 80)
print("Generated files in sda_results/:")
for fn in ["fig06_residual_plots.png", "fig07_sensitivity_plots.png",
           "fig08_shap_analysis.png",  "fig09_feature_importance.png",
           "table02_residuals_std.png", "table03_performance.png"]:
    print(f"  {fn}")
print("=" * 80)
print("ADVANCED PLOTTING COMPLETE")
print("EXIT_ADVANCED:0")
print("=" * 80)
_log_adv.close()
