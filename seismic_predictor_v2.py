"""
Seismic Data Prediction Interface - Alternative Version
========================================================
Works with scikit-learn MLPRegressor model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime

class SeismicPredictor:
    """Prediction interface for scikit-learn based seismic model."""
    
    def __init__(self):
        print("="*80)
        print("SEISMIC DATA PREDICTION SYSTEM")
        print("="*80)
        print("\n[Loading Model...]")
        
        self.model = joblib.load('seismic_model.pkl')
        self.scaler_X = joblib.load('scaler_X.pkl')
        self.scaler_y = joblib.load('scaler_y.pkl')
        
        with open('model_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.input_features = self.config['input_features']
        self.output_features = self.config['output_features']
        
        print(f"   ✓ Model loaded successfully")
        print(f"\n   Performance (Test Set):")
        print(f"   • R² Score: {self.config['test_metrics']['r2']:.6f}")
        print(f"   • RMSE: {self.config['test_metrics']['rmse']:.6f}")
        print(f"   • MAE: {self.config['test_metrics']['mae']:.6f}")
        print("\n" + "="*80)
    
    def predict(self, earthquake_magnitude, ztor_km, finite_fault_model, 
                rjb_km, vs30_m_s, verbose=True):
        """Make prediction for given seismic parameters."""
        
        epsilon = 1e-10
        log_vs30 = np.log(vs30_m_s + epsilon)
        log_rjb = np.log(rjb_km + epsilon)
        
        X_input = np.array([[
            earthquake_magnitude,
            ztor_km,
            finite_fault_model,
            rjb_km,
            log_vs30,
            log_rjb
        ]])
        
        X_scaled = self.scaler_X.transform(X_input)
        y_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_scaled.reshape(1, -1))
        
        results = {}
        for i, feature_name in enumerate(self.output_features):
            results[feature_name] = float(y_pred[0, i])
        
        if verbose:
            print("\n" + "="*80)
            print("PREDICTION RESULTS")
            print("="*80)
            print(f"\nInput Parameters:")
            print(f"   • Earthquake Magnitude: {earthquake_magnitude}")
            print(f"   • Depth to Top of Rupture (Ztor): {ztor_km} km")
            print(f"   • Finite Fault Model: {finite_fault_model}")
            print(f"   • Joyner-Boore Distance (Rjb): {rjb_km} km")
            print(f"   • Shear Wave Velocity (Vs30): {vs30_m_s} m/s")
            print(f"\nPredicted Spectral Accelerations:")
            print(f"   {'-'*70}")
            print(f"   {'Period':<20} {'Acceleration (g)':<20}")
            print(f"   {'-'*70}")
            
            for feature_name, value in results.items():
                print(f"   {feature_name:<20} {value:.8f}")
            
            print(f"   {'-'*70}")
            print("="*80 + "\n")
        
        return results
    
    def predict_batch(self, input_df):
        """Batch prediction from DataFrame."""
        
        epsilon = 1e-10
        X_input = input_df.copy()
        X_input['log_Vs30'] = np.log(X_input['Vs30_Selected_for_Analysis_m_s'] + epsilon)
        X_input['log_Rjb'] = np.log(X_input['Rjb_km'] + epsilon)
        
        X_input = X_input[self.input_features].values
        X_scaled = self.scaler_X.transform(X_input)
        y_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_scaled)
        
        results_df = pd.DataFrame(y_pred, columns=self.output_features)
        print(f"\n✓ Batch prediction completed for {len(input_df)} samples")
        
        return results_df
    
    def plot_response_spectrum(self, predictions, save_path=None, show_plot=True):
        """Plot response spectrum."""
        
        spectral_features = [f for f in self.output_features if f.startswith('T')]
        
        periods = []
        accelerations = []
        
        for feature in spectral_features:
            period_str = feature.replace('T', '').replace('S', '').replace('pt', '.')
            period = float(period_str)
            periods.append(period)
            accelerations.append(predictions[feature])
        
        plt.figure(figsize=(12, 7))
        plt.plot(periods, accelerations, 'b-o', linewidth=2, markersize=6)
        plt.scatter([0.01], [predictions['PGA_g']], s=150, c='red', marker='*', label='PGA')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Period (seconds)', fontsize=14, fontweight='bold')
        plt.ylabel('Spectral Acceleration (g)', fontsize=14, fontweight='bold')
        plt.title('Predicted Response Spectrum', fontsize=16, fontweight='bold')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Plot saved: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def compare_scenarios(self, scenarios_df, output_path='scenario_comparison.png'):
        """Compare multiple scenarios."""
        
        predictions_df = self.predict_batch(scenarios_df)
        spectral_features = [f for f in self.output_features if f.startswith('T')]
        
        periods = []
        for feature in spectral_features:
            period_str = feature.replace('T', '').replace('S', '').replace('pt', '.')
            periods.append(float(period_str))
        
        plt.figure(figsize=(14, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios_df)))
        
        for idx, row in scenarios_df.iterrows():
            accelerations = [predictions_df.loc[idx, f] for f in spectral_features]
            label = f"M={row['Earthquake_Magnitude']}, R={row['Rjb_km']:.0f}km"
            plt.plot(periods, accelerations, '-o', linewidth=2, color=colors[idx], label=label)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Period (seconds)', fontsize=14)
        plt.ylabel('Spectral Acceleration (g)', fontsize=14)
        plt.title('Scenario Comparison', fontsize=16, fontweight='bold')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"   ✓ Comparison saved: {output_path}")
        plt.close()

def example_predictions():
    """Run example predictions."""
    
    print("\n" + "="*80)
    print("RUNNING EXAMPLE PREDICTIONS")
    print("="*80)
    
    predictor = SeismicPredictor()
    
    # Example 1
    print("\n📍 Example 1: Moderate Earthquake at Close Distance")
    results1 = predictor.predict(6.5, 15.0, 0, 10.0, 500.0)
    predictor.plot_response_spectrum(results1, 'example1_spectrum.png', show_plot=False)
    
    # Example 2
    print("\n📍 Example 2: Large Earthquake at Moderate Distance")
    results2 = predictor.predict(7.5, 20.0, 0, 50.0, 450.0)
    predictor.plot_response_spectrum(results2, 'example2_spectrum.png', show_plot=False)
    
    # Example 3
    print("\n📍 Example 3: Smaller Earthquake at Far Distance")
    results3 = predictor.predict(6.0, 25.0, 0, 100.0, 600.0)
    predictor.plot_response_spectrum(results3, 'example3_spectrum.png', show_plot=False)
    
    # Compare scenarios
    print("\n📊 Comparing Multiple Scenarios...")
    scenarios = pd.DataFrame({
        'Earthquake_Magnitude': [6.0, 6.5, 7.0, 7.5],
        'Ztor_km': [15.0, 15.0, 20.0, 25.0],
        'Finite_Fault_Model': [0, 0, 0, 0],
        'Rjb_km': [20.0, 30.0, 50.0, 100.0],
        'Vs30_Selected_for_Analysis_m_s': [500.0, 450.0, 400.0, 600.0]
    })
    
    predictor.compare_scenarios(scenarios)
    
    print("\n" + "="*80)
    print("✓ Examples completed!")
    print("="*80 + "\n")

def interactive_mode():
    """Interactive prediction mode."""
    
    print("\n" + "="*80)
    print("INTERACTIVE PREDICTION MODE")
    print("="*80)
    
    predictor = SeismicPredictor()
    
    while True:
        print("\n" + "-"*80)
        print("Enter earthquake parameters (or 'q' to quit):")
        print("-"*80)
        
        try:
            mag = input("Earthquake Magnitude: ")
            if mag.lower() == 'q':
                break
            mag = float(mag)
            
            ztor = float(input("Ztor (km): "))
            ffm = int(input("Finite Fault Model (0/1): "))
            rjb = float(input("Rjb (km): "))
            vs30 = float(input("Vs30 (m/s): "))
            
            results = predictor.predict(mag, ztor, ffm, rjb, vs30)
            
            plot_choice = input("\nPlot spectrum? (y/n): ")
            if plot_choice.lower() == 'y':
                predictor.plot_response_spectrum(results, 
                    save_path=f'spectrum_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            
        except ValueError:
            print("\n❌ Invalid input")
            continue
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue
    
    print("\n✓ Exiting...")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SEISMIC PREDICTION SYSTEM")
    print("="*80)
    print("\nOptions:")
    print("  1. Run example predictions")
    print("  2. Interactive mode")
    print("  3. Exit")
    print("="*80)
    
    choice = input("\nSelect (1-3): ")
    
    if choice == '1':
        example_predictions()
    elif choice == '2':
        interactive_mode()
    elif choice == '3':
        print("\n✓ Exiting...")
    else:
        print("\n❌ Invalid choice. Running examples...")
        example_predictions()
