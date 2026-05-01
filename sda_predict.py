"""
SDA BiLSTM Ground Motion Predictor — Interactive Pipeline
==========================================================
Run this script and enter your scenario parameters when prompted.
No code editing is required.

Usage:   python sda_predict.py
Outputs: Console table  +  PNG spectrum  +  CSV (saved automatically)
"""

from __future__ import annotations
import json, os, sys, warnings
import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
import joblib, torch, torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.family": "sans-serif"})

# ---------------------------------------------------------------------------
# Paths — change only if your folder structure differs
# ---------------------------------------------------------------------------
MODEL_DIR  = "sda_results"
OUTPUT_DIR = "sda_results/user_predictions"

# ---------------------------------------------------------------------------
# Constants (must match sda_pipeline_utils.py)
# ---------------------------------------------------------------------------
MIN_RJB_KM = 0.1
OUTPUT_COLUMNS = [
    "PGA_g",
    "T0pt010S","T0pt020S","T0pt030S","T0pt040S","T0pt050S",
    "T0pt060S","T0pt070S","T0pt080S","T0pt090S","T0pt100S",
    "T0pt200S","T0pt300S","T0pt400S","T0pt500S","T0pt600S",
    "T0pt700S","T0pt800S","T0pt900S","T1pt000S",
    "T2pt000S","T3pt000S","T4pt000S","T5pt000S",
    "T6pt000S","T7pt000S","T8pt000S","T9pt000S","T10pt000S",
    "PGV_cm_sec",
]
_PSEUDO_T = [
    0.005,
    0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
    0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,
    1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,15.0,
]
_SPEC_IDX = [i for i,c in enumerate(OUTPUT_COLUMNS) if c not in ("PGA_g","PGV_cm_sec")]
_SPEC_T   = np.array([_PSEUDO_T[i] for i in _SPEC_IDX])
_IDX_PGA  = OUTPUT_COLUMNS.index("PGA_g")
_IDX_PGV  = OUTPUT_COLUMNS.index("PGV_cm_sec")

_TABLE = [
    ("PSA 0.10 s", "T0pt100S"), ("PSA 0.20 s", "T0pt200S"),
    ("PSA 0.30 s", "T0pt300S"), ("PSA 0.50 s", "T0pt500S"),
    ("PSA 1.00 s", "T1pt000S"), ("PSA 2.00 s", "T2pt000S"),
    ("PSA 3.00 s", "T3pt000S"), ("PSA 5.00 s", "T5pt000S"),
    ("PSA 10.0 s", "T10pt000S"),
]

# Parameter definitions: (display name, dtype, default, lo, hi, unit)
_PARAMS = [
    ("Earthquake Magnitude  Mw",          float, 7.0,  4.0,  8.5,  ""),
    ("Joyner-Boore Distance Rjb",         float, 50.0, 0.1, 500.0, "km"),
    ("Site Condition Vs30",               float, 760.0,100.0,2000.0,"m/s"),
    ("Depth to Top of Rupture  Ztor",     float, 15.0, 0.0, 120.0, "km"),
    ("Finite-Fault Model flag  FFM",      int,   1,    0,    1,     "(0=point, 1=finite)"),
    ("Scenario label (optional, Enter=skip)", str, "", None, None, ""),
]

# ---------------------------------------------------------------------------
# BiLSTM class definitions — must match seismic_bilstm_model.py
# ---------------------------------------------------------------------------
class SeismicBiLSTM(nn.Module):
    def __init__(self, n_inputs=8, n_periods=30, encoder_dims=(128,256,256),
                 period_embed_dim=64, lstm_hidden=128, lstm_layers=2, dropout=0.20):
        super().__init__()
        self.n_periods = n_periods
        context_dim = encoder_dims[-1]
        layers, in_dim = [], n_inputs
        for i, out_dim in enumerate(encoder_dims):
            layers += [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()]
            if i >= 1:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.encoder    = nn.Sequential(*layers)
        self.input_proj = nn.Linear(n_inputs, context_dim, bias=False)
        self.period_net = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, period_embed_dim)
        )
        pts = np.array([
            0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
            0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,
            1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,15.0,
        ], dtype=np.float32)
        self.register_buffer("log_T", torch.FloatTensor(np.log(pts[:n_periods])))
        lstm_in = context_dim + period_embed_dim
        self.bilstm = nn.LSTM(lstm_in, lstm_hidden, lstm_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if lstm_layers > 1 else 0.0)
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden*2, 128), nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        batch   = x.size(0)
        context = self.encoder(x) + self.input_proj(x)
        p_emb   = self.period_net(self.log_T.unsqueeze(-1))
        p_emb   = p_emb.unsqueeze(0).expand(batch, -1, -1)
        ctx     = context.unsqueeze(1).expand(-1, self.n_periods, -1)
        out, _  = self.bilstm(torch.cat([ctx, p_emb], dim=-1))
        return self.output_head(out).squeeze(-1)


class BiLSTMWrapper:
    def __init__(self, model, device=torch.device("cpu")):
        self.model = model; self.device = device
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(X).to(self.device)).cpu().numpy()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _div(ch="─", w=68): print(ch * w)


def _prompt_value(label, dtype, default, lo, hi, unit):
    """Prompt the user for a single parameter; validate range; return value."""
    while True:
        suffix = f"  [{lo} – {hi}] {unit}" if lo is not None else f"  {unit}"
        if dtype == str:
            raw = input(f"  {label}{suffix}: ").strip()
            return raw if raw else default
        default_str = str(int(default)) if dtype == int else str(default)
        raw = input(f"  {label}{suffix}  [default {default_str}]: ").strip()
        if raw == "":
            return dtype(default)
        try:
            val = dtype(raw)
        except ValueError:
            print(f"    [!] Please enter a valid {dtype.__name__} value.")
            continue
        if lo is not None and not (lo <= float(val) <= hi):
            print(f"    [WARNING] {val} is outside the recommended training range "
                  f"[{lo}, {hi}]. Predictions may extrapolate.")
            confirm = input("    Continue anyway? (y/n): ").strip().lower()
            if confirm != "y":
                continue
        return val


def _build_row(mw, ztor, ffm, rjb, vs30):
    log_rjb = float(np.log(max(float(rjb), MIN_RJB_KM)))
    return np.array([
        float(mw), float(ztor), float(ffm), float(rjb),
        float(np.log(float(vs30))), log_rjb,
        float(mw)**2, float(mw)*log_rjb,
    ])


def _predict_all(model, sx, sy, row):
    x_sc  = sx.transform(row.reshape(1, -1))
    y_sc  = model.predict(x_sc)
    y_log = sy.inverse_transform(y_sc.reshape(1, -1))[0]
    return np.exp(y_log)


def _smooth_spec(periods, values):
    lp = np.log(periods); lv = np.log(np.clip(values, 1e-14, None))
    dp = np.linspace(lp[0], lp[-1], 500)
    lv2 = gaussian_filter1d(PchipInterpolator(lp, lv)(dp), sigma=2.0, mode="nearest")
    return np.exp(dp), np.exp(lv2)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _print_results(sc, y_all, sigma_all):
    _div()
    print(f"  PREDICTION RESULTS — {sc['name'] or 'Unnamed scenario'}")
    _div()
    print(f"  Input parameters:")
    print(f"    Mw = {sc['Mw']}   Rjb = {sc['Rjb_km']} km   "
          f"Vs30 = {sc['Vs30']} m/s   Ztor = {sc['Ztor_km']} km   FFM = {sc['FFM']}")
    _div("·")
    print(f"  {'Intensity Measure':<22} {'Median':>12}  {'Unit'}")
    _div("·")
    print(f"  {'PGA':<22} {y_all[_IDX_PGA]:>12.5f}  g")
    print(f"  {'PGV':<22} {y_all[_IDX_PGV]:>12.4f}  cm/s")
    _div("·")
    for lbl, col in _TABLE:
        idx = OUTPUT_COLUMNS.index(col)
        sig = sigma_all[idx]
        lo  = y_all[idx] * np.exp(-sig)
        hi  = y_all[idx] * np.exp( sig)
        print(f"  {lbl:<22} {y_all[idx]:>12.5f}  g"
              f"   [+/-1s: {lo:.5f} – {hi:.5f}]")
    _div()


def _plot_spectrum(scenarios, sigma_all, save_path=None):
    palette = ["#1f4db5","#b22222","#228B22","#8B008B",
               "#D79B00","#2E8B57","#CC6600","#333333"]
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, sc in enumerate(scenarios):
        col    = palette[i % len(palette)]
        raw    = sc["y_all"][_SPEC_IDX]
        sigs   = sigma_all[_SPEC_IDX]
        dp, sv = _smooth_spec(_SPEC_T, raw)

        # Re-interpolate sigma envelope on same dense grid
        lp_orig = np.log(_SPEC_T)
        dp_lp   = np.log(dp)
        sig_dense = PchipInterpolator(lp_orig, sigs)(dp_lp)

        ax.plot(dp, sv, color=col, lw=2.3,
                label=sc["name"] or f"Scenario {i+1}")
        ax.fill_between(dp, sv*np.exp(-sig_dense), sv*np.exp(sig_dense),
                        alpha=0.13, color=col)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.01, 10.0)
    ax.set_xlabel("Period (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("PSA (g)",    fontsize=12, fontweight="bold")
    ax.set_title("BiLSTM-GMM — Predicted PSA Response Spectra\n"
                 "(Shaded band: +/-1 sigma test uncertainty)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(fontsize=9.5, framealpha=0.92)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\n  Plot saved: {save_path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------
def main():
    # ── Load model ─────────────────────────────────────────────────────────
    required = {
        "Model":    os.path.join(MODEL_DIR, "seismic_model_robust.pkl"),
        "Scaler X": os.path.join(MODEL_DIR, "scaler_X.pkl"),
        "Scaler y": os.path.join(MODEL_DIR, "scaler_y.pkl"),
        "Config":   os.path.join(MODEL_DIR, "model_config.json"),
    }
    for label, path in required.items():
        if not os.path.exists(path):
            sys.exit(
                f"\n[ERROR] {label} not found: {path}\n"
                "Run seismic_bilstm_model.py first to train the model.\n"
            )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _div("═")
    print("  SDA BiLSTM Ground Motion Predictor — Interactive Mode")
    _div("═")
    print("  Loading model weights and scalers ...", end=" ", flush=True)

    model    = joblib.load(required["Model"])
    scaler_X = joblib.load(required["Scaler X"])
    scaler_y = joblib.load(required["Scaler y"])

    with open(required["Config"], encoding="utf-8") as fh:
        config = json.load(fh)

    rmse_map  = {m["feature"]: m["rmse"] for m in config.get("feature_metrics", [])}
    sigma_all = np.array([rmse_map.get(c, 0.0) for c in OUTPUT_COLUMNS])

    r2   = config.get("test_metrics", {}).get("r2", float("nan"))
    arch = config.get("architecture", "BiLSTM")
    ep   = config.get("best_epoch", "?")
    print("done.")
    print(f"\n  Architecture : {arch}")
    print(f"  Parameters   : {config.get('total_params', '?'):,}")
    print(f"  Best epoch   : {ep}    |    Test R2 = {r2:.4f}")
    _div()
    print("  Model is ready. Enter your scenario parameters below.")
    print("  Recommended input ranges are shown in brackets.")
    print("  Press Enter to accept the default value shown.")

    session_results = []
    run_again = True

    while run_again:
        _div("─")
        print("  SCENARIO INPUT")
        _div("─")

        values = {}
        for label, dtype, default, lo, hi, unit in _PARAMS:
            values[label] = _prompt_value(label, dtype, default, lo, hi, unit)

        mw    = values["Earthquake Magnitude  Mw"]
        rjb   = values["Joyner-Boore Distance Rjb"]
        vs30  = values["Site Condition Vs30"]
        ztor  = values["Depth to Top of Rupture  Ztor"]
        ffm   = values["Finite-Fault Model flag  FFM"]
        name  = values["Scenario label (optional, Enter=skip)"]

        print("\n  Computing prediction ...", end=" ", flush=True)
        row   = _build_row(mw, ztor, ffm, rjb, vs30)
        y_all = _predict_all(model, scaler_X, scaler_y, row)
        print("done.\n")

        sc = {"name": name, "Mw": mw, "Rjb_km": rjb,
              "Vs30": vs30, "Ztor_km": ztor, "FFM": ffm,
              "y_all": y_all}
        session_results.append(sc)

        _print_results(sc, y_all, sigma_all)

        # ── Save CSV ──────────────────────────────────────────────────────
        save_csv = input("  Save full output (all 30 IMs) to CSV? (y/n): ").strip().lower()
        if save_csv == "y":
            fname = (name.replace(" ", "_")[:40] or f"scenario_{len(session_results)}")
            csv_path = os.path.join(OUTPUT_DIR, f"{fname}.csv")
            row_dict = {"Scenario": name, "Mw": mw, "Rjb_km": rjb,
                        "Vs30_m_s": vs30, "Ztor_km": ztor, "FFM": ffm}
            for j, col in enumerate(OUTPUT_COLUMNS):
                row_dict[col] = float(y_all[j])
            pd.DataFrame([row_dict]).to_csv(csv_path, index=False, float_format="%.6e")
            print(f"  CSV saved: {csv_path}")

        # ── Plot ──────────────────────────────────────────────────────────
        show_plot = input("  Show / save spectrum plot? (y/n): ").strip().lower()
        if show_plot == "y":
            fname2 = (name.replace(" ", "_")[:40] or f"scenario_{len(session_results)}")
            png_path = os.path.join(OUTPUT_DIR, f"{fname2}_spectrum.png")
            _plot_spectrum([sc], sigma_all, save_path=png_path)

        # ── Loop control ──────────────────────────────────────────────────
        _div("─")
        again = input("  Predict another scenario? (y/n): ").strip().lower()
        run_again = (again == "y")

    # ── End-of-session: compare all scenarios on one plot ─────────────────
    if len(session_results) > 1:
        _div("═")
        compare = input(
            f"  {len(session_results)} scenarios computed. "
            "Plot all on one comparison chart? (y/n): "
        ).strip().lower()
        if compare == "y":
            comp_path = os.path.join(OUTPUT_DIR, "comparison_spectra.png")
            _plot_spectrum(session_results, sigma_all, save_path=comp_path)

    _div("═")
    print("  Session complete. Thank you for using SDA BiLSTM-GMM Predictor.")
    _div("═")


if __name__ == "__main__":
    main()
