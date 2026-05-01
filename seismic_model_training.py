"""
Seismic Data Deep Learning Model
==================================
This script builds a robust deep learning model to predict spectral accelerations
from seismic parameters using the NGA Subduction dataset.

Input Features (6):
- Earthquake_Magnitude
- Ztor_km
- Finite_Fault_Model
- Rjb_km
- log(Vs30_Selected_for_Analysis_m_s)
- log(Rjb_km)

Output Features (29):
- PGA_g and spectral accelerations (T0pt010S through T10pt000S)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure TensorFlow for optimal performance on your CPU
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

print("="*80)
print("SEISMIC DATA DEEP LEARNING MODEL - TRAINING")
print("="*80)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Training Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[1/7] Loading and Preprocessing Data...")

# Load the dataset
df = pd.read_csv('NGA_Subduction_filtered.csv')
print(f"   ✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Define input features (with log transformations)
input_feature_names = [
    'Earthquake_Magnitude',
    'Ztor_km',
    'Finite_Fault_Model',
    'Rjb_km',
    'Vs30_Selected_for_Analysis_m_s'  # Will be log-transformed
]

# Define output features (spectral accelerations)
output_feature_names = [
    'PGA_g',
    'T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt050S', 'T0pt060S',
    'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt100S', 'T0pt200S',
    'T0pt300S', 'T0pt400S', 'T0pt500S', 'T0pt600S', 'T0pt700S',
    'T0pt800S', 'T0pt900S', 'T1pt000S', 'T2pt000S', 'T3pt000S',
    'T4pt000S', 'T5pt000S', 'T6pt000S', 'T7pt000S', 'T8pt000S',
    'T9pt000S', 'T10pt000S'
]

print(f"   ✓ Input features: {len(input_feature_names)} + 2 log transforms = 6 total")
print(f"   ✓ Output features: {len(output_feature_names)}")

# Create feature matrix with transformations
X_raw = df[input_feature_names].copy()

# Add log transformations
# Add small epsilon to avoid log(0)
epsilon = 1e-10
X_raw['log_Vs30'] = np.log(X_raw['Vs30_Selected_for_Analysis_m_s'] + epsilon)
X_raw['log_Rjb'] = np.log(X_raw['Rjb_km'] + epsilon)

# Final input features (6 features)
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

# Check for any missing or invalid values
print(f"\n   Checking data quality...")
print(f"   ✓ Input shape: {X.shape}")
print(f"   ✓ Output shape: {y.shape}")
print(f"   ✓ Missing values in X: {np.isnan(X).sum()}")
print(f"   ✓ Missing values in y: {np.isnan(y).sum()}")
print(f"   ✓ Infinite values in X: {np.isinf(X).sum()}")
print(f"   ✓ Infinite values in y: {np.isinf(y).sum()}")

# ============================================================================
# 2. TRAIN-TEST-VALIDATION SPLIT
# ============================================================================

print("\n[2/7] Splitting Dataset...")

# Split: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
)

print(f"   ✓ Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   ✓ Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"   ✓ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ============================================================================
# 3. FEATURE SCALING
# ============================================================================

print("\n[3/7] Scaling Features...")

# Use RobustScaler for input (robust to outliers in seismic data)
scaler_X = RobustScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Use StandardScaler for output
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

print(f"   ✓ Input scaled using RobustScaler")
print(f"   ✓ Output scaled using StandardScaler")

# ============================================================================
# 4. MODEL ARCHITECTURE
# ============================================================================

print("\n[4/7] Building Neural Network Architecture...")

def create_model(input_dim, output_dim):
    """
    Create a deep neural network optimized for seismic data prediction.
    
    Architecture:
    - Multiple hidden layers with increasing then decreasing neurons
    - Batch normalization for stable training
    - Dropout for regularization
    - L2 regularization
    - ReLU activation for hidden layers
    """
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # First block - Expansion
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Second block
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Third block - Peak capacity
        layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Fourth block
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Fifth block - Compression
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Sixth block
        layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Output layer (linear activation for regression)
        layers.Dense(output_dim, activation='linear')
    ])
    
    return model

# Create the model
model = create_model(X_train_scaled.shape[1], y_train_scaled.shape[1])

# Compile with custom loss and optimizer
model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss='mse',
    metrics=['mae', 'mse']
)

# Print model summary
print("\n   Model Architecture:")
print("   " + "="*70)
model.summary(print_fn=lambda x: print("   " + x))
print("   " + "="*70)

total_params = model.count_params()
print(f"\n   ✓ Total parameters: {total_params:,}")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

print("\n[5/7] Training Model...")

# Create callbacks
checkpoint_cb = callbacks.ModelCheckpoint(
    'best_seismic_model',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=0,
    save_format='tf'  # Use TensorFlow SavedModel format
)

early_stopping_cb = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

# Custom callback for progress
class TrainingProgress(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}: "
                  f"loss={logs['loss']:.6f}, "
                  f"val_loss={logs['val_loss']:.6f}, "
                  f"mae={logs['mae']:.6f}, "
                  f"val_mae={logs['val_mae']:.6f}")

progress_cb = TrainingProgress()

# Train the model
print("\n   Starting training process...")
print("   " + "-"*70)

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=200,
    batch_size=32,
    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, progress_cb],
    verbose=0
)

print("   " + "-"*70)
print(f"   ✓ Training completed after {len(history.history['loss'])} epochs")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================

print("\n[6/7] Evaluating Model Performance...")

# Load best model
model = keras.models.load_model('best_seismic_model')

# Predictions on all sets
y_train_pred_scaled = model.predict(X_train_scaled, verbose=0)
y_val_pred_scaled = model.predict(X_val_scaled, verbose=0)
y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)

# Inverse transform to original scale
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
    print(f"   • R² Score: {r2:.6f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

train_metrics = calculate_metrics(y_train, y_train_pred, "Training")
val_metrics = calculate_metrics(y_val, y_val_pred, "Validation")
test_metrics = calculate_metrics(y_test, y_test_pred, "Test")

# Per-output feature metrics
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
# 7. SAVE MODEL AND ARTIFACTS
# ============================================================================

print("\n[7/7] Saving Model and Artifacts...")

# Save scalers
import joblib
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("   ✓ Scalers saved: scaler_X.pkl, scaler_y.pkl")

# Save model configuration
model_config = {
    'input_features': input_columns,
    'output_features': output_feature_names,
    'train_samples': int(X_train.shape[0]),
    'val_samples': int(X_val.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'total_parameters': int(total_params),
    'training_epochs': len(history.history['loss']),
    'train_metrics': {k: float(v) for k, v in train_metrics.items()},
    'val_metrics': {k: float(v) for k, v in val_metrics.items()},
    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
    'feature_metrics': [{k: (float(v) if isinstance(v, (int, float)) else v) 
                         for k, v in fm.items()} for fm in feature_metrics],
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=4)
print("   ✓ Model configuration saved: model_config.json")

# ============================================================================
# 8. VISUALIZATION
# ============================================================================

print("\n[8/7] Generating Visualizations...")

# Create output directory
os.makedirs('training_results', exist_ok=True)

# 1. Training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('Mean Absolute Error During Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
if 'lr' in history.history:
    plt.plot(history.history['lr'], linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results/training_history.png', dpi=300, bbox_inches='tight')
print("   ✓ Training history plot saved")

# 2. Predicted vs Actual (Sample outputs)
sample_outputs = [0, 1, 10, 18, 26]  # PGA, T0.01, T0.2, T1.0, T10.0
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, output_idx in enumerate(sample_outputs):
    ax = axes[idx]
    
    # Plot test set
    ax.scatter(y_test[:, output_idx], y_test_pred[:, output_idx], 
               alpha=0.6, s=30, label='Test Data', color='steelblue')
    
    # Perfect prediction line
    min_val = min(y_test[:, output_idx].min(), y_test_pred[:, output_idx].min())
    max_val = max(y_test[:, output_idx].max(), y_test_pred[:, output_idx].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Labels and formatting
    ax.set_xlabel(f'Actual {output_feature_names[output_idx]}', fontsize=11)
    ax.set_ylabel(f'Predicted {output_feature_names[output_idx]}', fontsize=11)
    r2 = r2_score(y_test[:, output_idx], y_test_pred[:, output_idx])
    ax.set_title(f'{output_feature_names[output_idx]} (R² = {r2:.4f})', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Remove extra subplot
axes[-1].remove()

plt.tight_layout()
plt.savefig('training_results/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("   ✓ Prediction scatter plots saved")

# 3. R² scores for all outputs
plt.figure(figsize=(16, 6))
r2_scores = [r2_score(y_test[:, i], y_test_pred[:, i]) for i in range(len(output_feature_names))]
colors = plt.cm.viridis(np.linspace(0, 1, len(output_feature_names)))
bars = plt.bar(range(len(output_feature_names)), r2_scores, color=colors, edgecolor='black', linewidth=1.5)

plt.axhline(y=0.9, color='g', linestyle='--', linewidth=2, label='R² = 0.9 (Excellent)', alpha=0.7)
plt.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, label='R² = 0.7 (Good)', alpha=0.7)
plt.xlabel('Output Feature', fontsize=12, fontweight='bold')
plt.ylabel('R² Score', fontsize=12, fontweight='bold')
plt.title('Model Performance: R² Score for Each Output Feature', fontsize=14, fontweight='bold')
plt.xticks(range(len(output_feature_names)), output_feature_names, rotation=45, ha='right')
plt.ylim([0, 1.05])
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('training_results/r2_scores_all_outputs.png', dpi=300, bbox_inches='tight')
print("   ✓ R² scores plot saved")

plt.close('all')

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE - MODEL SUMMARY")
print("="*80)

print(f"\n📊 Dataset Information:")
print(f"   • Total samples: {len(X):,}")
print(f"   • Training: {len(X_train):,} samples")
print(f"   • Validation: {len(X_val):,} samples")
print(f"   • Test: {len(X_test):,} samples")

print(f"\n🏗️  Model Architecture:")
print(f"   • Input features: 6")
print(f"   • Output features: 29")
print(f"   • Hidden layers: 6")
print(f"   • Total parameters: {total_params:,}")
print(f"   • Training epochs: {len(history.history['loss'])}")

print(f"\n📈 Performance Metrics (Test Set):")
print(f"   • R² Score: {test_metrics['r2']:.6f}")
print(f"   • RMSE: {test_metrics['rmse']:.6f}")
print(f"   • MAE: {test_metrics['mae']:.6f}")

avg_r2 = np.mean(r2_scores)
print(f"\n   • Average R² across all outputs: {avg_r2:.6f}")
print(f"   • Outputs with R² > 0.9: {sum(1 for r2 in r2_scores if r2 > 0.9)}/{len(r2_scores)}")
print(f"   • Outputs with R² > 0.7: {sum(1 for r2 in r2_scores if r2 > 0.7)}/{len(r2_scores)}")

print(f"\n💾 Saved Files:")
print(f"   ✓ best_seismic_model/ (Best model - TensorFlow SavedModel format)")
print(f"   ✓ scaler_X.pkl (Input feature scaler)")
print(f"   ✓ scaler_y.pkl (Output feature scaler)")
print(f"   ✓ model_config.json (Model configuration)")
print(f"   ✓ training_results/ (Visualization plots)")

print("\n" + "="*80)
print("Next step: Run 'seismic_predictor.py' to use the model for predictions!")
print("="*80 + "\n")
