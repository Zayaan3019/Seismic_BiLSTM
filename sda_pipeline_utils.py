"""Shared data-loading and preprocessing utilities for the SDA seismic pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EPSILON = 1e-10
RANDOM_STATE = 42

# Rjb floor applied before log(Rjb) – standard practice for on-fault (Rjb=0) records.
# 0.1 km avoids −∞ without introducing a meaningful physical distortion.
MIN_RJB_KM = 0.1

# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

INPUT_COLUMNS = [
    "Earthquake_Magnitude",   # Mw
    "Ztor_km",                # depth to top of rupture (km)
    "Finite_Fault_Model",     # binary: 1 = finite fault model available
    "Rjb_km",                 # Joyner-Boore distance (km), raw
    "log_Vs30",               # ln(Vs30)
    "log_Rjb",                # ln(Rjb), floored at MIN_RJB_KM for on-fault records
    "Mw_sq",                  # Mw² — quadratic magnitude scaling (physics-informed)
    "Mw_logR",                # Mw × ln(Rjb) — magnitude-attenuation interaction
]

# 30 outputs: PGA + 28 oscillator periods spanning 0.01 s – 10.0 s + PGV
OUTPUT_COLUMNS = [
    "PGA_g",
    "T0pt010S",   # 0.01 s
    "T0pt020S",   # 0.02 s
    "T0pt030S",   # 0.03 s
    "T0pt040S",   # 0.04 s
    "T0pt050S",   # 0.05 s
    "T0pt060S",   # 0.06 s
    "T0pt070S",   # 0.07 s
    "T0pt080S",   # 0.08 s
    "T0pt090S",   # 0.09 s
    "T0pt100S",   # 0.10 s
    "T0pt200S",   # 0.20 s
    "T0pt300S",   # 0.30 s
    "T0pt400S",   # 0.40 s
    "T0pt500S",   # 0.50 s
    "T0pt600S",   # 0.60 s
    "T0pt700S",   # 0.70 s
    "T0pt800S",   # 0.80 s
    "T0pt900S",   # 0.90 s
    "T1pt000S",   # 1.0 s
    "T2pt000S",   # 2.0 s
    "T3pt000S",   # 3.0 s
    "T4pt000S",   # 4.0 s
    "T5pt000S",   # 5.0 s
    "T6pt000S",   # 6.0 s
    "T7pt000S",   # 7.0 s
    "T8pt000S",   # 8.0 s
    "T9pt000S",   # 9.0 s
    "T10pt000S",  # 10.0 s
    "PGV_cm_sec", # Peak ground velocity (cm/s); pseudo-period 15 s in sequence
]

# Focal-mechanism labels — kept for visualization (Fig 02, residuals) even
# though Fault_Type is no longer a model input.
FAULT_LABEL_MAP = {
    3: "Strike Slip",
    1: "Normal",
    2: "Reverse",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(csv_path: str = "NGA_Subduction_filtered.csv") -> dict:
    """
    Load the NGA-Subduction flatfile, clean sentinel values, engineer features,
    and return model-ready arrays.

    Sentinel handling
    -----------------
    * Rjb_km = −999  → NaN (5 records): non-physical placeholder.
    * Vs30 ≤ 0        → NaN (65 records at −999): non-physical placeholder.
    * Rjb_km = 0      → kept, but floored to MIN_RJB_KM before log().

    Feature engineering
    -------------------
    * log_Vs30 = ln(Vs30)
    * log_Rjb  = ln(max(Rjb_km, MIN_RJB_KM))

    All records with any non-finite model input or log-output are dropped.
    Fault_Type is retained in clean_df for downstream visualization only.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Columns required for feature engineering and outputs
    required_raw = [
        "Earthquake_Magnitude",
        "Ztor_km",
        "Finite_Fault_Model",
        "Fault_Type",                       # visualization only, not a model input
        "Rjb_km",
        "Vs30_Selected_for_Analysis_m_s",
        *OUTPUT_COLUMNS,
    ]
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    work = df.copy()
    for col in required_raw:
        work[col] = _to_numeric(work[col])

    # --- Sentinel removal ---
    work.loc[work["Rjb_km"] < 0.0, "Rjb_km"] = np.nan                      # −999 sentinel
    work.loc[work["Vs30_Selected_for_Analysis_m_s"] <= 0.0,
             "Vs30_Selected_for_Analysis_m_s"] = np.nan                      # −999 sentinel

    # --- Categorical fill ---
    for col in ("Fault_Type", "Finite_Fault_Model"):
        mode_val = work[col].dropna().mode()
        if len(mode_val):
            work[col] = work[col].fillna(mode_val.iloc[0])
        work[col] = work[col].astype(int)

    # --- Feature engineering ---
    work["log_Vs30"] = np.log(work["Vs30_Selected_for_Analysis_m_s"])
    work["log_Rjb"]  = np.log(work["Rjb_km"].clip(lower=MIN_RJB_KM))
    work["Mw_sq"]    = work["Earthquake_Magnitude"] ** 2
    work["Mw_logR"]  = work["Earthquake_Magnitude"] * work["log_Rjb"]

    # --- Build X / y ---
    X_df     = work[INPUT_COLUMNS].copy()
    y_raw_df = work[OUTPUT_COLUMNS].copy()
    y_log_df = np.log(y_raw_df.clip(lower=EPSILON))

    # Drop any record where a model input or output is non-finite
    mask = (
        np.isfinite(X_df.values).all(axis=1)
        & np.isfinite(y_log_df.values).all(axis=1)
    )

    clean_df = work.loc[mask].reset_index(drop=True)
    X_df     = X_df.loc[mask].reset_index(drop=True)
    y_raw_df = y_raw_df.loc[mask].reset_index(drop=True)
    y_log_df = y_log_df.loc[mask].reset_index(drop=True)

    return {
        "raw_df":   df,
        "clean_df": clean_df,
        "X_df":     X_df,
        "y_raw_df": y_raw_df,
        "y_log_df": y_log_df,
        "X":        X_df.values,
        "y_log":    y_log_df.values,
        "y_raw":    y_raw_df.values,
    }


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

def split_data(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE) -> dict:
    """Stratified 70 / 15 / 15 split."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state
    )
    # 15% of the remaining 85% ≈ 17.65% → gives 15% of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=random_state
    )
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def output_to_label(output_name: str) -> str:
    """Human-readable label for figures and tables."""
    if output_name == "PGA_g":
        return "PGA"
    if output_name == "PGV_cm_sec":
        return "PGV"
    if output_name.startswith("T") and output_name.endswith("S"):
        period = output_name[1:-1].replace("pt", ".")
        return f"PSA{float(period):.3g}s"
    return output_name


def get_fault_labels(series: pd.Series) -> pd.Series:
    mapped = series.map(FAULT_LABEL_MAP)
    return mapped.fillna("Other")


def fault_code(label: str) -> int:
    """Reverse-lookup focal-mechanism label → integer code."""
    inverse = {v: k for k, v in FAULT_LABEL_MAP.items()}
    return inverse[label]
