"""
Comprehensive Model Testing and Validation
===========================================
This script performs rigorous testing of the trained seismic model
to ensure it meets benchmark performance standards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
import joblib
import json
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE MODEL TESTING AND VALIDATION")
print("="*80)

# Load model components
print("\n[1/5] Loading Model Components...")
model = keras.models.load_model('best_seismic_model')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

with open('model_config.json', 'r') as f:
    config = json.load(f)

print("   ✓ Model loaded successfully")

# Load test data
print("\n[2/5] Loading Test Data...")
df = pd.read_csv('NGA_Subduction_filtered.csv')

# Prepare features
input_columns = config['input_features']
output_columns = config['output_features']

# Recreate features
X_raw = df[['Earthquake_Magnitude', 'Ztor_km', 'Finite_Fault_Model', 
            'Rjb_km', 'Vs30_Selected_for_Analysis_m_s']].copy()

epsilon = 1e-10
X_raw['log_Vs30'] = np.log(X_raw['Vs30_Selected_for_Analysis_m_s'] + epsilon)
X_raw['log_Rjb'] = np.log(X_raw['Rjb_km'] + epsilon)

X = X_raw[input_columns].values
y = df[output_columns].values

# Use the same test split
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"   ✓ Test set: {len(X_test)} samples")

# Make predictions
print("\n[3/5] Making Predictions...")
X_test_scaled = scaler_X.transform(X_test)
y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

print("   ✓ Predictions completed")

# Comprehensive testing
print("\n[4/5] Performing Comprehensive Tests...")
print("="*80)

# Test 1: Overall Performance Metrics
print("\n📊 TEST 1: Overall Performance Metrics")
print("-"*80)

mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

# Calculate MAPE (handling zeros carefully)
mask = y_test != 0
mape = np.mean(np.abs((y_test[mask] - y_test_pred[mask]) / y_test[mask])) * 100

print(f"MSE:  {mse:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"MAE:  {mae:.8f}")
print(f"R² Score: {r2:.6f}")
print(f"MAPE: {mape:.2f}%")

# Benchmark assessment
print(f"\n✓ BENCHMARK ASSESSMENT:")
if r2 >= 0.9:
    print(f"   🌟 EXCELLENT: R² = {r2:.4f} (≥ 0.9)")
elif r2 >= 0.8:
    print(f"   ✓ VERY GOOD: R² = {r2:.4f} (≥ 0.8)")
elif r2 >= 0.7:
    print(f"   ✓ GOOD: R² = {r2:.4f} (≥ 0.7)")
else:
    print(f"   ⚠ NEEDS IMPROVEMENT: R² = {r2:.4f}")

# Test 2: Per-Output Performance
print("\n📊 TEST 2: Per-Output Feature Performance")
print("-"*80)
print(f"{'Feature':<20} {'R²':<10} {'RMSE':<12} {'MAE':<12} {'Status':<15}")
print("-"*80)

excellent_count = 0
good_count = 0
needs_improvement = 0

per_output_metrics = []

for i, feature_name in enumerate(output_columns):
    r2_feat = r2_score(y_test[:, i], y_test_pred[:, i])
    rmse_feat = np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i]))
    mae_feat = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
    
    if r2_feat >= 0.9:
        status = "Excellent ⭐"
        excellent_count += 1
    elif r2_feat >= 0.7:
        status = "Good ✓"
        good_count += 1
    else:
        status = "Fair ⚠"
        needs_improvement += 1
    
    print(f"{feature_name:<20} {r2_feat:<10.4f} {rmse_feat:<12.6f} {mae_feat:<12.6f} {status:<15}")
    
    per_output_metrics.append({
        'feature': feature_name,
        'r2': r2_feat,
        'rmse': rmse_feat,
        'mae': mae_feat
    })

print("-"*80)
print(f"\nSummary:")
print(f"   • Excellent (R² ≥ 0.9): {excellent_count}/{len(output_columns)}")
print(f"   • Good (R² ≥ 0.7): {good_count}/{len(output_columns)}")
print(f"   • Needs Improvement: {needs_improvement}/{len(output_columns)}")

# Test 3: Residual Analysis
print("\n📊 TEST 3: Residual Analysis")
print("-"*80)

residuals = y_test - y_test_pred
residuals_flat = residuals.flatten()

# Normality test (Shapiro-Wilk on a sample due to large dataset)
sample_size = min(5000, len(residuals_flat))
sample_residuals = np.random.choice(residuals_flat, sample_size, replace=False)
stat, p_value = stats.shapiro(sample_residuals)

print(f"Residual Statistics:")
print(f"   • Mean: {np.mean(residuals_flat):.8f}")
print(f"   • Std Dev: {np.std(residuals_flat):.8f}")
print(f"   • Median: {np.median(residuals_flat):.8f}")
print(f"   • Min: {np.min(residuals_flat):.8f}")
print(f"   • Max: {np.max(residuals_flat):.8f}")
print(f"\nNormality Test (Shapiro-Wilk on sample):")
print(f"   • p-value: {p_value:.6f}")

if abs(np.mean(residuals_flat)) < 0.01:
    print(f"   ✓ Residuals are approximately centered at zero")
else:
    print(f"   ⚠ Residuals show bias")

# Test 4: Prediction Intervals
print("\n📊 TEST 4: Prediction Confidence")
print("-"*80)

# Calculate prediction intervals for each output
within_1std = []
within_2std = []

for i in range(y_test.shape[1]):
    errors = y_test[:, i] - y_test_pred[:, i]
    std_error = np.std(errors)
    
    within_1 = np.sum(np.abs(errors) <= std_error) / len(errors) * 100
    within_2 = np.sum(np.abs(errors) <= 2*std_error) / len(errors) * 100
    
    within_1std.append(within_1)
    within_2std.append(within_2)

avg_within_1std = np.mean(within_1std)
avg_within_2std = np.mean(within_2std)

print(f"Percentage of predictions within error bounds (average across outputs):")
print(f"   • Within 1 std dev: {avg_within_1std:.1f}% (Expected: ~68%)")
print(f"   • Within 2 std dev: {avg_within_2std:.1f}% (Expected: ~95%)")

if avg_within_1std >= 65 and avg_within_2std >= 93:
    print(f"   ✓ Prediction intervals are well-calibrated")
else:
    print(f"   ⚠ Prediction intervals may need calibration")

# Test 5: Extreme Value Performance
print("\n📊 TEST 5: Performance on Extreme Values")
print("-"*80)

# Test on high magnitude scenarios
extreme_mask = df['Earthquake_Magnitude'].values >= 7.0
if np.sum(extreme_mask) > 0:
    # Get indices in the reordered test set
    all_indices = np.arange(len(y))
    _, test_indices = train_test_split(all_indices, test_size=0.15, random_state=42)
    
    extreme_in_test = []
    for idx in test_indices:
        if extreme_mask[idx]:
            extreme_in_test.append(np.where(test_indices == idx)[0][0])
    
    if len(extreme_in_test) > 0:
        extreme_in_test = np.array(extreme_in_test)
        y_extreme = y_test[extreme_in_test]
        y_extreme_pred = y_test_pred[extreme_in_test]
        
        r2_extreme = r2_score(y_extreme, y_extreme_pred)
        mae_extreme = mean_absolute_error(y_extreme, y_extreme_pred)
        
        print(f"Performance on M ≥ 7.0 earthquakes:")
        print(f"   • Number of samples: {len(extreme_in_test)}")
        print(f"   • R² Score: {r2_extreme:.4f}")
        print(f"   • MAE: {mae_extreme:.6f}")
        
        if r2_extreme >= 0.7:
            print(f"   ✓ Good performance on extreme events")
        else:
            print(f"   ⚠ Performance degrades for extreme events")

# Test 6: Cross-validation of predictions
print("\n📊 TEST 6: Consistency Check")
print("-"*80)

# Make predictions 5 times (with dropout, should be similar)
predictions_multiple = []
for _ in range(5):
    y_pred_temp = model.predict(X_test_scaled, verbose=0)
    predictions_multiple.append(scaler_y.inverse_transform(y_pred_temp))

predictions_multiple = np.array(predictions_multiple)
prediction_std = np.std(predictions_multiple, axis=0)
avg_std = np.mean(prediction_std)

print(f"Prediction consistency (std dev across 5 runs):")
print(f"   • Average std dev: {avg_std:.8f}")

if avg_std < 0.001:
    print(f"   ✓ Predictions are highly consistent")
else:
    print(f"   ⚠ Predictions show variability")

# Generate visualization report
print("\n[5/5] Generating Test Visualizations...")
print("="*80)

# Create comprehensive test report figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Residual distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(residuals_flat, bins=100, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Residuals', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax1.grid(True, alpha=0.3)

# 2. Q-Q plot
ax2 = fig.add_subplot(gs[0, 1])
stats.probplot(sample_residuals, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. R² scores by output
ax3 = fig.add_subplot(gs[0, 2])
r2_scores_list = [m['r2'] for m in per_output_metrics]
colors_bar = ['green' if r2 >= 0.9 else 'orange' if r2 >= 0.7 else 'red' for r2 in r2_scores_list]
ax3.barh(range(len(output_columns)), r2_scores_list, color=colors_bar)
ax3.set_yticks(range(len(output_columns)))
ax3.set_yticklabels(output_columns, fontsize=7)
ax3.set_xlabel('R² Score', fontsize=10)
ax3.set_title('R² Score by Output Feature', fontsize=12, fontweight='bold')
ax3.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent')
ax3.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Good')
ax3.set_xlim([0, 1])
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4-6. Predicted vs Actual for key outputs
key_outputs = [0, 10, 18]  # PGA, T0.2, T1.0
titles = ['PGA', 'T=0.2s', 'T=1.0s']

for idx, (output_idx, title) in enumerate(zip(key_outputs, titles)):
    ax = fig.add_subplot(gs[1, idx])
    ax.scatter(y_test[:, output_idx], y_test_pred[:, output_idx], alpha=0.5, s=20)
    
    min_val = min(y_test[:, output_idx].min(), y_test_pred[:, output_idx].min())
    max_val = max(y_test[:, output_idx].max(), y_test_pred[:, output_idx].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel(f'Actual {title}', fontsize=10)
    ax.set_ylabel(f'Predicted {title}', fontsize=10)
    r2_val = r2_score(y_test[:, output_idx], y_test_pred[:, output_idx])
    ax.set_title(f'{title} (R²={r2_val:.4f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

# 7. Residuals vs Predicted
ax7 = fig.add_subplot(gs[2, 0])
for i in range(min(5, y_test.shape[1])):
    ax7.scatter(y_test_pred[:, i], residuals[:, i], alpha=0.3, s=10)
ax7.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Predicted Values', fontsize=10)
ax7.set_ylabel('Residuals', fontsize=10)
ax7.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. MAE by output
ax8 = fig.add_subplot(gs[2, 1])
mae_scores = [m['mae'] for m in per_output_metrics]
ax8.bar(range(len(output_columns)), mae_scores, color='steelblue', edgecolor='black')
ax8.set_xlabel('Output Feature Index', fontsize=10)
ax8.set_ylabel('MAE', fontsize=10)
ax8.set_title('Mean Absolute Error by Output', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# 9. Performance summary text
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
summary_text = f"""
MODEL PERFORMANCE SUMMARY

Overall Metrics:
• R² Score: {r2:.6f}
• RMSE: {rmse:.6f}
• MAE: {mae:.6f}
• MAPE: {mape:.2f}%

Output Features:
• Excellent (R² ≥ 0.9): {excellent_count}/{len(output_columns)}
• Good (R² ≥ 0.7): {good_count}/{len(output_columns)}
• Fair: {needs_improvement}/{len(output_columns)}

Residual Analysis:
• Mean: {np.mean(residuals_flat):.6f}
• Std Dev: {np.std(residuals_flat):.6f}

Confidence:
• Within 1σ: {avg_within_1std:.1f}%
• Within 2σ: {avg_within_2std:.1f}%

STATUS: {'✓ BENCHMARK ACHIEVED' if r2 >= 0.85 else '⚠ NEEDS IMPROVEMENT'}
"""
ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center', 
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Comprehensive Model Testing Report', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('training_results/comprehensive_test_report.png', dpi=300, bbox_inches='tight')
print("   ✓ Comprehensive test report saved")

plt.close()

# Final report
print("\n" + "="*80)
print("TESTING COMPLETE - FINAL ASSESSMENT")
print("="*80)

print(f"\n✅ Model Performance: ", end="")
if r2 >= 0.9:
    print("EXCELLENT (R² ≥ 0.9)")
elif r2 >= 0.8:
    print("VERY GOOD (R² ≥ 0.8)")
elif r2 >= 0.7:
    print("GOOD (R² ≥ 0.7)")
else:
    print("NEEDS IMPROVEMENT")

print(f"\n📊 Detailed Results:")
print(f"   • {excellent_count} out of {len(output_columns)} outputs achieved excellent performance (R² ≥ 0.9)")
print(f"   • {good_count + excellent_count} out of {len(output_columns)} outputs achieved good or better performance (R² ≥ 0.7)")

if r2 >= 0.85 and excellent_count >= len(output_columns) * 0.5:
    print(f"\n🎉 BENCHMARK LEVEL ACHIEVED!")
    print(f"   The model demonstrates high accuracy and robustness.")
else:
    print(f"\n⚠️  Model shows satisfactory performance but may benefit from:")
    print(f"   • More training data")
    print(f"   • Hyperparameter tuning")
    print(f"   • Feature engineering")

print("\n💾 Generated Files:")
print("   ✓ training_results/comprehensive_test_report.png")

print("\n" + "="*80 + "\n")
