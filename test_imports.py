"""Quick smoke test for all seismic pipeline dependencies."""
import sys
sys.path.insert(0, '.')

try:
    import json, os, warnings
    from datetime import datetime
    import joblib
    import matplotlib
    matplotlib.use('Agg')
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
        FAULT_LABEL_MAP, INPUT_COLUMNS, OUTPUT_COLUMNS, RANDOM_STATE,
        get_fault_labels, output_to_label, prepare_dataset, split_data,
    )
    print("All imports OK")

    bundle = prepare_dataset("NGA_Subduction_filtered.csv")
    print(f"Dataset OK, records: {len(bundle['clean_df'])}")
    print(f"X shape: {bundle['X'].shape}")
    print(f"y_log shape: {bundle['y_log'].shape}")

    splits = split_data(bundle["X"], bundle["y_log"])
    print(f"Train: {splits['X_train'].shape[0]} | Val: {splits['X_val'].shape[0]} | Test: {splits['X_test'].shape[0]}")
    print("All checks PASSED")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback; traceback.print_exc()
