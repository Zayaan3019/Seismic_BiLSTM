"""
Robust Seismic Ground Motion Prediction Model (SDA-aligned)
===========================================================

This script trains a robust MLP model in log space and generates SDA-style
core figures plus Table 01 statistics.
"""

import json
import os
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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
np.random.seed(RANDOM_STATE)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def print_metric_block(name: str, metrics: dict) -> None:
    print(f"\n   {name} set metrics:")
    print("   " + "-" * 52)
    print(f"   MSE : {metrics['mse']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   MAE : {metrics['mae']:.6f}")
    print(f"   R2  : {metrics['r2']:.6f}")


print("=" * 80)
print("ROBUST SEISMIC GROUND MOTION PREDICTION MODEL")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[1/10] Loading and preprocessing data...")
bundle = prepare_dataset("NGA_Subduction_filtered.csv")
clean_df = bundle["clean_df"]
X_df = bundle["X_df"]
y_raw_df = bundle["y_raw_df"]
y_log_df = bundle["y_log_df"]
X = bundle["X"]
y = bundle["y_log"]

print(f"   Dataset loaded: {len(clean_df)} clean records")
print(f"   Input features: {len(INPUT_COLUMNS)}")
print(f"   Output features: {len(OUTPUT_COLUMNS)}")
print("   Output domain: natural-log transformed")

# ============================================================================
# 2. OUTPUT DIRECTORY
# ============================================================================

output_dir = "sda_results"
os.makedirs(output_dir, exist_ok=True)
print(f"\n[2/10] Output directory ready: {output_dir}/")

# ============================================================================
# 3. FIG 01 / FIG 02 / FIG 03
# ============================================================================

print("\n[3/10] Generating data distribution and frequency plots...")

# Fig 01
fig, axes = plt.subplots(2, 1, figsize=(11, 12))

ax1 = axes[0]
ax1.scatter(
    clean_df["Rjb_km"],
    clean_df["Earthquake_Magnitude"],
    alpha=0.35,
    s=22,
    color="lightgray",
    edgecolors="gray",
    linewidth=0.45,
)
ax1.set_xscale("log")
ax1.set_xlabel("Joyner-Boore distance, Rjb (km)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Magnitude (Mw)", fontsize=12, fontweight="bold")
ax1.set_title("Fig 01: Data used", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1e-2, 3e2)
ax1.set_ylim(3, 8)

ax2 = axes[1]
magnitude_bins = [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
colors_mag = ["blue", "limegreen", "red", "gold", "magenta"]
labels_mag = ["3<Mw<4", "4<Mw<5", "5<Mw<6", "6<=Mw<7", "7<=Mw<8"]
for (m_min, m_max), color, label in zip(magnitude_bins, colors_mag, labels_mag):
    mask_mag = (clean_df["Earthquake_Magnitude"] >= m_min) & (clean_df["Earthquake_Magnitude"] < m_max)
    ax2.scatter(
        clean_df.loc[mask_mag, "Rjb_km"],
        clean_df.loc[mask_mag, "PGA_g"],
        alpha=0.55,
        s=28,
        c=color,
        label=label,
        edgecolors="none",
    )

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Joyner-Boore distance, Rjb (km)", fontsize=12, fontweight="bold")
ax2.set_ylabel("PGA(g)", fontsize=12, fontweight="bold")
ax2.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, which="both")
ax2.set_xlim(1e-2, 3e2)
ax2.set_ylim(1e-6, 2)

plt.tight_layout()
plt.savefig(f"{output_dir}/fig01_data_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# Fig 02 - fault mechanism distributions in Mw-Rjb space
fault_labels = get_fault_labels(clean_df["Fault_Type"])
marker_map = {
    "Strike Slip": "x",
    "Normal": "*",
    "Reverse": "s",
    "Other": "o",
}
color_map = {
    "Strike Slip": "#b04ee0",
    "Normal": "#58b947",
    "Reverse": "#f1b722",
    "Other": "#2ca8ff",
}

fig, axes = plt.subplots(2, 1, figsize=(11, 12))

for ax, title_text in zip(
    axes,
    [
        "Fig 02: Data used (Fault mechanism view - full set)",
        "Fault mechanism view (cleaned modeling set)",
    ],
):
    for label in ["Strike Slip", "Normal", "Reverse", "Other"]:
        mask_fault = fault_labels == label
        if mask_fault.sum() == 0:
            continue
        ax.scatter(
            clean_df.loc[mask_fault, "Rjb_km"],
            clean_df.loc[mask_fault, "Earthquake_Magnitude"],
            alpha=0.45,
            s=30,
            marker=marker_map[label],
            c=color_map[label],
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Joyner-Boore distance (Rjb), km", fontsize=12, fontweight="bold")
    ax.set_ylabel("Magnitude (Mw)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1e-2, 3e2)
    ax.set_ylim(3, 8)
    ax.set_title(title_text, fontsize=13, fontweight="bold")

axes[0].legend(loc="lower left", fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{output_dir}/fig02_data_used.png", dpi=300, bbox_inches="tight")
plt.close()

# Fig 03
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(clean_df["Earthquake_Magnitude"], bins=20, edgecolor="black", color="#F5B59E", alpha=0.8)
axes[0].set_xlabel("Magnitude (Mw)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("No. of records", fontsize=12, fontweight="bold")
axes[0].set_title("Magnitude distribution", fontsize=13, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].hist(clean_df["Rjb_km"], bins=25, edgecolor="black", color="#F5B59E", alpha=0.8)
axes[1].set_xlabel("Joyner-Boore distance, Rjb (km)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("No. of records", fontsize=12, fontweight="bold")
axes[1].set_title("Distance distribution", fontsize=13, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")

fault_counts = fault_labels.value_counts()
fault_order = ["Normal", "Reverse", "Strike Slip", "Other"]
plot_labels = [label for label in fault_order if label in fault_counts.index]
plot_counts = [int(fault_counts[label]) for label in plot_labels]
axes[2].barh(
    np.arange(len(plot_labels)),
    plot_counts,
    color=["#F8D8C8", "#F6B996", "#F08B66", "#DFA27A"][: len(plot_labels)],
    edgecolor="black",
    alpha=0.85,
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
# 4. TRAIN/VAL/TEST SPLIT
# ============================================================================

print("\n[4/10] Splitting dataset...")
splits = split_data(X, y, random_state=RANDOM_STATE)
X_train = splits["X_train"]
X_val = splits["X_val"]
X_test = splits["X_test"]
y_train = splits["y_train"]
y_val = splits["y_val"]
y_test = splits["y_test"]

print(f"   Train: {len(X_train)} samples ({len(X_train) / len(X) * 100:.1f}%)")
print(f"   Val  : {len(X_val)} samples ({len(X_val) / len(X) * 100:.1f}%)")
print(f"   Test : {len(X_test)} samples ({len(X_test) / len(X) * 100:.1f}%)")

# ============================================================================
# 5. SCALING
# ============================================================================

print("\n[5/10] Scaling features...")
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

print("   Input scaler : StandardScaler")
print("   Output scaler: StandardScaler")

# ============================================================================
# 6. MODEL
# ============================================================================

print("\n[6/10] Building optimized MLP...")
model = MLPRegressor(
    hidden_layer_sizes=(128, 256, 256, 128),
    activation="relu",
    solver="adam",
    alpha=0.0005,
    batch_size=64,
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=800,
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=80,
    random_state=RANDOM_STATE,
    verbose=True,
)
print("   Architecture: 6 -> 128 -> 256 -> 256 -> 128 -> 22")
print("   Training objective: multi-output regression in log space")

# ============================================================================
# 7. TRAINING / EVALUATION
# ============================================================================

print("\n[7/10] Training model...")
print("   " + "-" * 70)
model.fit(X_train_scaled, y_train_scaled)
print("   " + "-" * 70)
print(f"   Training finished after {model.n_iter_} iterations")

print("\n[8/10] Evaluating model...")
y_train_pred = scaler_y.inverse_transform(model.predict(X_train_scaled))
y_val_pred = scaler_y.inverse_transform(model.predict(X_val_scaled))
y_test_pred = scaler_y.inverse_transform(model.predict(X_test_scaled))

train_metrics = calculate_metrics(y_train, y_train_pred)
val_metrics = calculate_metrics(y_val, y_val_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

print_metric_block("Training", train_metrics)
print_metric_block("Validation", val_metrics)
print_metric_block("Test", test_metrics)

print("\n   Per-output metrics (test set):")
print("   " + "-" * 90)
print(f"   {'Parameter':<16}{'R2':>10}{'RMSE':>14}{'MAE':>14}{'R':>12}")
print("   " + "-" * 90)

feature_metrics = []
for i, feature_name in enumerate(OUTPUT_COLUMNS):
    r2_i = float(r2_score(y_test[:, i], y_test_pred[:, i]))
    rmse_i = float(np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i])))
    mae_i = float(mean_absolute_error(y_test[:, i], y_test_pred[:, i]))
    r_i = float(np.corrcoef(y_test[:, i], y_test_pred[:, i])[0, 1])

    feature_metrics.append(
        {
            "feature": feature_name,
            "label": output_to_label(feature_name),
            "r2": r2_i,
            "rmse": rmse_i,
            "mae": mae_i,
            "r": r_i,
        }
    )
    print(f"   {output_to_label(feature_name):<16}{r2_i:>10.4f}{rmse_i:>14.4f}{mae_i:>14.4f}{r_i:>12.4f}")

# ============================================================================
# 8. SAVE ARTIFACTS
# ============================================================================

print("\n[9/10] Saving artifacts...")
joblib.dump(model, f"{output_dir}/seismic_model_robust.pkl")
joblib.dump(scaler_X, f"{output_dir}/scaler_X.pkl")
joblib.dump(scaler_y, f"{output_dir}/scaler_y.pkl")

model_config = {
    "input_features": INPUT_COLUMNS,
    "output_features": OUTPUT_COLUMNS,
    "architecture": "6 -> 128 -> 256 -> 256 -> 128 -> 22",
    "train_samples": int(len(X_train)),
    "val_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "iterations": int(model.n_iter_),
    "train_metrics": train_metrics,
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
    "feature_metrics": feature_metrics,
    "fault_label_map": {str(k): v for k, v in FAULT_LABEL_MAP.items()},
    "log_transformed_outputs": True,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open(f"{output_dir}/model_config.json", "w", encoding="utf-8") as f:
    json.dump(model_config, f, indent=4)
print("   Model, scalers, and config saved")

# ============================================================================
# 9. FIG 04 / FIG 05
# ============================================================================

print("\n[10/10] Generating architecture, regression, and Table 01 plots...")

# Fig 04
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis("off")

layer_positions = [0.08, 0.27, 0.46, 0.64, 0.82, 0.96]
layer_sizes = [6, 128, 256, 256, 128, len(OUTPUT_COLUMNS)]
layer_colors = ["#9EA6FF", "#F7D7DD", "#F3EA8D", "#F3EA8D", "#F7D7DD", "#57F257"]
layer_labels = ["Input\n(6)", "Hidden 1\n(128)", "Hidden 2\n(256)", "Hidden 3\n(256)", "Hidden 4\n(128)", f"Output\n({len(OUTPUT_COLUMNS)})"]

nodes = []
for x_pos, size, color, label in zip(layer_positions, layer_sizes, layer_colors, layer_labels):
    n_draw = min(size, 11)
    ys = np.linspace(0.10, 0.90, n_draw)
    layer_nodes = []
    for y_pos in ys:
        circ = plt.Circle((x_pos, y_pos), 0.016, color=color, ec="gray", lw=0.9)
        ax.add_patch(circ)
        layer_nodes.append((x_pos, y_pos))
    nodes.append(layer_nodes)
    ax.text(x_pos, 0.03, label, ha="center", va="top", fontsize=9, fontweight="bold")

for i in range(len(nodes) - 1):
    for n1 in nodes[i][::2]:
        for n2 in nodes[i + 1][::2]:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color="gray", alpha=0.30, lw=0.6)

# Input labels (Mw at bottom, Focalmech at top — matching expected PDF order)
input_labels_arch = ["Focalmech", "Depth", r"log($V_s$30)", r"log($R_{jb}$+50)", "$R_{jb}$", "$M_w$"]
for idx, label in enumerate(input_labels_arch):
    if idx < len(nodes[0]):
        ax.annotate("", xy=(nodes[0][idx][0] - 0.016, nodes[0][idx][1]),
                    xytext=(0.005, nodes[0][idx][1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
        ax.text(0.003, nodes[0][idx][1], label, ha="right", va="center", fontsize=10)

# Output labels (show top and bottom of output layer)
out_labels_map = {0: "ln(PGA)", 1: "ln(PGV)", len(nodes[-1]) - 2: r"ln($PSA_{1.5s}$)", len(nodes[-1]) - 1: r"ln($PSA_{4s}$)"}
for idx, label in out_labels_map.items():
    if idx < len(nodes[-1]):
        ax.annotate("", xy=(nodes[-1][idx][0] + 0.035, nodes[-1][idx][1]),
                    xytext=(nodes[-1][idx][0] + 0.016, nodes[-1][idx][1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
        ax.text(nodes[-1][idx][0] + 0.037, nodes[-1][idx][1], label, ha="left", va="center", fontsize=10)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, 1)
ax.set_title("Fig 04: Architecture for ANN", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}/fig04_architecture.png", dpi=300, bbox_inches="tight")
plt.close()

# Fig 05
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

regression_targets = ["PGA_g", "PGV_cm_sec", "T0pt100S", "T1pt000S"]
subplot_titles = ["a) PGA", "b) PGV", "c) PSA at 0.1s", "d) PSA at 1.0s"]

for ax, out_name, sub_title in zip(axes, regression_targets, subplot_titles):
    idx = OUTPUT_COLUMNS.index(out_name)

    ax.scatter(
        y_train[:, idx],
        y_train_pred[:, idx],
        alpha=0.5,
        s=22,
        color="#6AAA2A",
        edgecolors="none",
        label="Training data",
    )
    ax.scatter(
        y_val[:, idx],
        y_val_pred[:, idx],
        alpha=0.55,
        s=25,
        color="#F07B3B",
        edgecolors="none",
        label="Validation data",
    )
    ax.scatter(
        y_test[:, idx],
        y_test_pred[:, idx],
        alpha=0.65,
        s=28,
        color="#F2D046",
        edgecolors="goldenrod",
        linewidth=0.4,
        label="Testing data",
    )

    all_vals = np.concatenate([y_test[:, idx], y_test_pred[:, idx]])
    min_v = float(np.min(all_vals))
    max_v = float(np.max(all_vals))
    ax.plot([min_v, max_v], [min_v, max_v], "k-", lw=2.0, label="Ideal fit")

    r_val = float(np.corrcoef(y_test[:, idx], y_test_pred[:, idx])[0, 1])
    ax.set_title(sub_title, fontsize=13, fontweight="bold")
    ax.text(0.04, 0.92, f"R-{r_val:.3f}", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.set_xlabel("Target", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, framealpha=0.95, bbox_to_anchor=(0.5, 0.03))
fig.suptitle("Fig 05: Regression plots", fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.savefig(f"{output_dir}/fig05_regression_plots.png", dpi=300, bbox_inches="tight")
plt.close()

# Table 01
stat_series = {
    "Mw": clean_df["Earthquake_Magnitude"],
    "-Rjb": -clean_df["Rjb_km"],
    "-log(Rjb)": -np.log(clean_df["Rjb_km"].clip(lower=1e-6)),
    "log(Vs30)": clean_df["log_Vs30"],
    "Depth": clean_df["Ztor_km"],
}
for col in OUTPUT_COLUMNS:
    stat_series[output_to_label(col)] = y_log_df[col]

rows = []
for name, series in stat_series.items():
    s = pd.to_numeric(series, errors="coerce").dropna()
    rows.append(
        {
            "Parameter": name,
            "Min": f"{s.min():.2f}",
            "Max": f"{s.max():.2f}",
            "Mean": f"{s.mean():.2f}",
            "Median": f"{s.median():.2f}",
            "STD": f"{s.std(ddof=1):.2f}",
            "Skewness": f"{s.skew():.2f}",
            "Kurtosis": f"{s.kurtosis():.2f}",
        }
    )

stats_df = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(17, 10))
ax.axis("off")

table = ax.table(
    cellText=stats_df.values,
    colLabels=stats_df.columns,
    loc="center",
    cellLoc="left",
    colLoc="left",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for col_idx in range(len(stats_df.columns)):
    table[(0, col_idx)].set_facecolor("#D0D0D0")
    table[(0, col_idx)].set_text_props(weight="bold")

plt.title("Table 01: Statistics of the data", fontsize=15, fontweight="bold", pad=16)
plt.savefig(f"{output_dir}/table01_statistics.png", dpi=300, bbox_inches="tight")
plt.close()

print("   Saved: Fig 04, Fig 05, Table 01")

print("\n" + "=" * 80)
print("MODEL TRAINING AND CORE PLOTS COMPLETE")
print("=" * 80)
print(f"Test R2 (overall): {test_metrics['r2']:.6f}")
print(f"Artifacts written to: {output_dir}/")
print("=" * 80)
