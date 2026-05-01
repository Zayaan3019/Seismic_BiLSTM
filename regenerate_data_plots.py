"""Regenerate Fig 01, Fig 02, Fig 03 from the existing dataset without retraining."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sda_pipeline_utils import get_fault_labels, prepare_dataset

sns.set_style("whitegrid")

OUTPUT_DIR = "sda_results"

bundle   = prepare_dataset("NGA_Subduction_filtered.csv")
clean_df = bundle["clean_df"]
y_log_df = bundle["y_log_df"]

# ============================================================================
# Fig 01 — Mw–Rjb scatter + PGA–Rjb by magnitude class
# ============================================================================
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
    if not msk.any():
        continue
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
plt.savefig(f"{OUTPUT_DIR}/fig01_data_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: fig01_data_distribution.png")

# ============================================================================
# Fig 02 — Fault-mechanism breakdown
# ============================================================================
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
    ax.set_ylim(3, 9.5)
    ax.set_title(title_text, fontsize=13, fontweight="bold")
axes[0].legend(loc="lower left", fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig02_data_used.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: fig02_data_used.png")

# ============================================================================
# Fig 03 — Frequency histograms
# ============================================================================
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
plt.savefig(f"{OUTPUT_DIR}/fig03_frequency_plots.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: fig03_frequency_plots.png")

print("\nDone — Fig 01, Fig 02, Fig 03 regenerated.")
