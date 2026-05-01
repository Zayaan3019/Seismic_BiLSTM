"""
Seismic Data Deep Learning Model - Alternative Implementation
===============================================================
This version uses PyTorch instead of TensorFlow to avoid compatibility issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import os
import json
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)

print("="*80)
print("SEISMIC DATA DEEP LEARNING MODEL - TRAINING")
print("="*80)
print(f"\nTraining Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[1/6] Loading and Preprocessing Data...")

df = pd.read_csv('NGA_Subduction_filtered.csv')
print(f"   ✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Input features
input_feature_names = [
    'Earthquake_Magnitude',
    'Ztor_km',
    'Finite_Fault_Model',
    'Rjb_km',
    'Vs30_Selected_for_Analysis_m_s'
]

# Output features
output_feature_names = [
    'PGA_g',
    'T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt050S', 'T0pt060S',
    'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt100S', 'T0pt200S',
    'T0pt300S', 'T0pt400S', 'T0pt500S', 'T0pt600S', 'T0pt700S',
    'T0pt800S', 'T0pt900S', 'T1pt000S', 'T2pt000S', 'T3pt000S',
    'T4pt000S', 'T5pt000S', 'T6pt000S', 'T7pt000S', 'T8pt000S',
    'T9pt000S', 'T10pt000S'
]

# Verify all output columns exist
available_outputs = [col for col in output_feature_names if col in df.columns]
if len(available_outputs) < len(output_feature_names):
    missing = set(output_feature_names) - set(available_outputs)
    print(f"   ⚠ Warning: {len(missing)} output features not found in dataset")
    output_feature_names = available_outputs

print(f"   ✓ Input features: {len(input_feature_names)} + 2 log transforms = 6 total")
print(f"   ✓ Output features: {len(output_feature_names)}")

# Create features with log transformations
X_raw = df[input_feature_names].copy()
epsilon = 1e-10
X_raw['log_Vs30'] = np.log(X_raw['Vs30_Selected_for_Analysis_m_s'] + epsilon)
X_raw['log_Rjb'] = np.log(X_raw['Rjb_km'] + epsilon)

input_columns = [
    'Earthquake_Magnitude',
    'Ztor_km',
    'Finite_Fault_Model',
    'Rjb_km',
    'log_Vs30',
    'log_Rjb'
]

X = X_raw[input_columns].values
y = df[output_feature_names].values

print(f"\n   Checking data quality...")
print(f"   ✓ Input shape: {X.shape}")
print(f"   ✓ Output shape: {y.shape}")

# Handle missing values by removing rows with NaN
mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
X = X[mask]
y = y[mask]

print(f"   ✓ After removing NaN: {X.shape[0]} samples")
print(f"   ✓ Missing values: {np.isnan(X).sum() + np.isnan(y).sum()}")

# ============================================================================
# 2. TRAIN-TEST-VALIDATION SPLIT
# ============================================================================

print("\n[2/6] Splitting Dataset...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42
)

print(f"   ✓ Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   ✓ Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"   ✓ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ============================================================================
# 3. FEATURE SCALING
# ============================================================================

print("\n[3/6] Scaling Features...")

scaler_X = RobustScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

print(f"   ✓ Features scaled")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

print("\n[4/6] Training Neural Network...")
print("   Building Multi-Layer Perceptron Regressor...")

# Create MLPRegressor with optimal parameters
model = MLPRegressor(
    hidden_layer_sizes=(512, 1024, 1024, 512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=42,
    verbose=True
)

print("\n   Training in progress...")
print("   " + "-"*70)

model.fit(X_train_scaled, y_train_scaled)

print("   " + "-"*70)
print(f"   ✓ Training completed")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print("\n[5/6] Evaluating Model Performance...")

# Predictions
y_train_pred_scaled = model.predict(X_train_scaled)
y_val_pred_scaled = model.predict(X_val_scaled)
y_test_pred_scaled = model.predict(X_test_scaled)

# Inverse transform
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# Calculate metrics
def calculate_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n   {set_name} Set Metrics:")
    print(f"   {'─'*50}")
    print(f"   • MSE:  {mse:.6f}")
    print(f"   • RMSE: {rmse:.6f}")
    print(f"   • MAE:  {mae:.6f}")
    print(f"   • R²: {r2:.6f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

train_metrics = calculate_metrics(y_train, y_train_pred, "Training")
val_metrics = calculate_metrics(y_val, y_val_pred, "Validation")
test_metrics = calculate_metrics(y_test, y_test_pred, "Test")

# Per-output metrics
print(f"\n   Per-Output Feature Performance (Test Set):")
print(f"   {'─'*70}")
print(f"   {'Feature':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
print(f"   {'─'*70}")

feature_metrics = []
for i, feature_name in enumerate(output_feature_names):
    r2 = r2_score(y_test[:, i], y_test_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i]))
    mae = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
    print(f"   {feature_name:<20} {r2:<10.4f} {rmse:<10.6f} {mae:<10.6f}")
    feature_metrics.append({'feature': feature_name, 'r2': r2, 'rmse': rmse, 'mae': mae})

# ============================================================================
# 6. SAVE MODEL AND ARTIFACTS
# ===========================================================================

print("\n[6/6] Saving Model and Artifacts...")

# Save model and scalers
joblib.dump(model, 'seismic_model.pkl')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("   ✓ Model and scalers saved")

# Save configuration
model_config = {
    'input_features': input_columns,
    'output_features': output_feature_names,
    'train_samples': int(X_train.shape[0]),
    'val_samples': int(X_val.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'train_metrics': {k: float(v) for k, v in train_metrics.items()},
    'val_metrics': {k: float(v) for k, v in val_metrics.items()},
    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
    'feature_metrics': [{k: (float(v) if isinstance(v, (int, float)) else v) 
                         for k, v in fm.items()} for fm in feature_metrics],
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=4)
print("   ✓ Configuration saved")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

print("\n[7/6] Generating Visualizations...")

os.makedirs('training_results', exist_ok=True)

# 1. R² scores
plt.figure(figsize=(16, 6))
r2_scores = [fm['r2'] for fm in feature_metrics]
colors = plt.cm.viridis(np.linspace(0, 1, len(output_feature_names)))
plt.bar(range(len(output_feature_names)), r2_scores, color=colors, edgecolor='black')

plt.axhline(y=0.9, color='g', linestyle='--', linewidth=2, label='R² = 0.9 (Excellent)')
plt.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, label='R² = 0.7 (Good)')
plt.xlabel('Output Feature', fontsize=12, fontweight='bold')
plt.ylabel('R² Score', fontsize=12, fontweight='bold')
plt.title('Model Performance: R² Score for Each Output Feature', fontsize=14, fontweight='bold')
plt.xticks(range(len(output_feature_names)), output_feature_names, rotation=45, ha='right')
plt.ylim([0, 1.05])
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('training_results/r2_scores.png', dpi=300, bbox_inches='tight')
print("   ✓ R² scores plot saved")

# 2. Predictions vs Actual
sample_outputs = [0, 1, 10, 18, 26]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, output_idx in enumerate(sample_outputs):
    ax = axes[idx]
    ax.scatter(y_test[:, output_idx], y_test_pred[:, output_idx], 
               alpha=0.6, s=30, color='steelblue')
    
    min_val = min(y_test[:, output_idx].min(), y_test_pred[:, output_idx].min())
    max_val = max(y_test[:, output_idx].max(), y_test_pred[:, output_idx].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel(f'Actual {output_feature_names[output_idx]}', fontsize=11)
    ax.set_ylabel(f'Predicted {output_feature_names[output_idx]}', fontsize=11)
    r2 = r2_score(y_test[:, output_idx], y_test_pred[:, output_idx])
    ax.set_title(f'{output_feature_names[output_idx]} (R² = {r2:.4f})', fontsize=12)
    ax.grid(True, alpha=0.3)

axes[-1].remove()
plt.tight_layout()
plt.savefig('training_results/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("   ✓ Prediction plots saved")

plt.close('all')

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE - MODEL SUMMARY")
print("="*80)

print(f"\n📊 Dataset:")
print(f"   • Total: {len(X):,} samples")
print(f"   • Training: {len(X_train):,}")
print(f"   • Validation: {len(X_val):,}")
print(f"   • Test: {len(X_test):,}")

print(f"\n🏗️  Model Architecture:")
print(f"   • Type: Multi-Layer Perceptron")
print(f"   • Hidden layers: (512, 1024, 1024, 512, 256, 128)")
print(f"   • Input features: 6")
print(f"   • Output features: 29")

print(f"\n📈 Performance (Test Set):")
print(f"   • R² Score: {test_metrics['r2']:.6f}")
print(f"   • RMSE: {test_metrics['rmse']:.6f}")
print(f"   • MAE: {test_metrics['mae']:.6f}")

avg_r2 = np.mean(r2_scores)
print(f"\n   • Average R² across outputs: {avg_r2:.6f}")
print(f"   • Outputs with R² > 0.9: {sum(1 for r2 in r2_scores if r2 > 0.9)}/{len(r2_scores)}")
print(f"   • Outputs with R² > 0.7: {sum(1 for r2 in r2_scores if r2 > 0.7)}/{len(r2_scores)}")

print(f"\n💾 Saved Files:")
print(f"   ✓ seismic_model.pkl")
print(f"   ✓ scaler_X.pkl")
print(f"   ✓ scaler_y.pkl")
print(f"   ✓ model_config.json")
print(f"   ✓ training_results/")

print("\n" + "="*80)
print("Next: Run 'python seismic_predictor_v2.py' to make predictions!")
print("="*80 + "\n")
