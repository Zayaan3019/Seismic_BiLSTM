"""
Seismic Data Prediction Interface
===================================
This script provides an interactive interface to use the trained deep learning model
for predicting spectral accelerations from seismic parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class SeismicPredictor:
    """
    A class for making predictions using the trained seismic model.
    """
    
    def __init__(self, model_path='best_seismic_model',
                 scaler_X_path='scaler_X.pkl',
                 scaler_y_path='scaler_y.pkl',
                 config_path='model_config.json'):
        """
        Initialize the predictor by loading the model and scalers.
        """
        print("="*80)
        print("SEISMIC DATA PREDICTION SYSTEM")
        print("="*80)
        print("\n[Loading Model Components...]")
        
        # Load model
        self.model = keras.models.load_model(model_path)
        print(f"   ✓ Model loaded from {model_path}")
        
        # Load scalers
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        print(f"   ✓ Scalers loaded")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        print(f"   ✓ Configuration loaded")
        
        # Extract feature names
        self.input_features = self.config['input_features']
        self.output_features = self.config['output_features']
        
        print(f"\n   Model Performance (Test Set):")
        print(f"   • R² Score: {self.config['test_metrics']['r2']:.6f}")
        print(f"   • RMSE: {self.config['test_metrics']['rmse']:.6f}")
        print(f"   • MAE: {self.config['test_metrics']['mae']:.6f}")
        print("\n" + "="*80)
    
    def predict(self, earthquake_magnitude, ztor_km, finite_fault_model, 
                rjb_km, vs30_m_s, verbose=True):
        """
        Predict spectral accelerations for given seismic parameters.
        
        Parameters:
        -----------
        earthquake_magnitude : float
            Earthquake magnitude (e.g., 6.5, 7.0)
        ztor_km : float
            Depth to top of rupture in km
        finite_fault_model : int
            Finite fault model indicator (0 or 1)
        rjb_km : float
            Joyner-Boore distance in km
        vs30_m_s : float
            Shear wave velocity in m/s
        verbose : bool
            Print prediction details
        
        Returns:
        --------
        dict : Dictionary containing predictions for all output features
        """
        
        # Prepare input with log transformations
        epsilon = 1e-10
        log_vs30 = np.log(vs30_m_s + epsilon)
        log_rjb = np.log(rjb_km + epsilon)
        
        # Create input array
        X_input = np.array([[
            earthquake_magnitude,
            ztor_km,
            finite_fault_model,
            rjb_km,
            log_vs30,
            log_rjb
        ]])
        
        # Scale input
        X_scaled = self.scaler_X.transform(X_input)
        
        # Make prediction
        y_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_scaled)
        
        # Create results dictionary
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
        """
        Predict for a batch of inputs from a DataFrame.
        
        Parameters:
        -----------
        input_df : pandas.DataFrame
            DataFrame with columns: Earthquake_Magnitude, Ztor_km, 
            Finite_Fault_Model, Rjb_km, Vs30_Selected_for_Analysis_m_s
        
        Returns:
        --------
        pandas.DataFrame : Predictions for all output features
        """
        
        # Check required columns
        required_cols = ['Earthquake_Magnitude', 'Ztor_km', 'Finite_Fault_Model', 
                        'Rjb_km', 'Vs30_Selected_for_Analysis_m_s']
        
        for col in required_cols:
            if col not in input_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Prepare features
        epsilon = 1e-10
        X_input = input_df.copy()
        X_input['log_Vs30'] = np.log(X_input['Vs30_Selected_for_Analysis_m_s'] + epsilon)
        X_input['log_Rjb'] = np.log(X_input['Rjb_km'] + epsilon)
        
        # Reorder columns to match training
        X_input = X_input[self.input_features].values
        
        # Scale and predict
        X_scaled = self.scaler_X.transform(X_input)
        y_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_scaled)
        
        # Create results DataFrame
        results_df = pd.DataFrame(y_pred, columns=self.output_features)
        
        print(f"\n✓ Batch prediction completed for {len(input_df)} samples")
        
        return results_df
    
    def plot_response_spectrum(self, predictions, save_path=None, show_plot=True):
        """
        Plot the predicted response spectrum.
        
        Parameters:
        -----------
        predictions : dict
            Dictionary of predictions from predict() method
        save_path : str, optional
            Path to save the plot
        show_plot : bool
            Whether to display the plot
        """
        
        # Extract periods and accelerations (excluding PGA)
        spectral_features = [f for f in self.output_features if f.startswith('T')]
        
        # Parse periods from feature names
        periods = []
        accelerations = []
        
        for feature in spectral_features:
            # Extract numeric value from feature name (e.g., 'T0pt010S' -> 0.010)
            period_str = feature.replace('T', '').replace('S', '').replace('pt', '.')
            period = float(period_str)
            periods.append(period)
            accelerations.append(predictions[feature])
        
        # Create plot
        plt.figure(figsize=(12, 7))
        plt.plot(periods, accelerations, 'b-o', linewidth=2, markersize=6, 
                markerfacecolor='steelblue', markeredgecolor='darkblue', 
                markeredgewidth=1.5, label='Predicted Spectrum')
        
        # Add PGA as a separate point
        plt.scatter([0.01], [predictions['PGA_g']], s=150, c='red', 
                   marker='*', edgecolors='darkred', linewidths=2, 
                   label='PGA', zorder=5)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Period (seconds)', fontsize=14, fontweight='bold')
        plt.ylabel('Spectral Acceleration (g)', fontsize=14, fontweight='bold')
        plt.title('Predicted Response Spectrum', fontsize=16, fontweight='bold')
        plt.grid(True, which='both', alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Response spectrum plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def compare_scenarios(self, scenarios_df, output_path='scenario_comparison.png'):
        """
        Compare multiple earthquake scenarios.
        
        Parameters:
        -----------
        scenarios_df : pandas.DataFrame
            DataFrame with multiple scenarios (rows) and input parameters (columns)
        output_path : str
            Path to save comparison plot
        """
        
        # Get predictions
        predictions_df = self.predict_batch(scenarios_df)
        
        # Extract spectral features
        spectral_features = [f for f in self.output_features if f.startswith('T')]
        
        # Parse periods
        periods = []
        for feature in spectral_features:
            period_str = feature.replace('T', '').replace('S', '').replace('pt', '.')
            periods.append(float(period_str))
        
        # Plot
        plt.figure(figsize=(14, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios_df)))
        
        for idx, row in scenarios_df.iterrows():
            accelerations = [predictions_df.loc[idx, f] for f in spectral_features]
            label = f"Scenario {idx+1}: M={row['Earthquake_Magnitude']}, R={row['Rjb_km']:.1f}km, Vs30={row['Vs30_Selected_for_Analysis_m_s']:.0f}m/s"
            plt.plot(periods, accelerations, '-o', linewidth=2, markersize=5, 
                    color=colors[idx], label=label)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Period (seconds)', fontsize=14, fontweight='bold')
        plt.ylabel('Spectral Acceleration (g)', fontsize=14, fontweight='bold')
        plt.title('Comparison of Earthquake Scenarios', fontsize=16, fontweight='bold')
        plt.grid(True, which='both', alpha=0.3, linestyle='--')
        plt.legend(fontsize=9, loc='best')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Scenario comparison plot saved to {output_path}")
        plt.close()

def interactive_mode():
    """
    Interactive mode for making predictions.
    """
    print("\n" + "="*80)
    print("INTERACTIVE PREDICTION MODE")
    print("="*80)
    
    try:
        predictor = SeismicPredictor()
        
        while True:
            print("\n" + "-"*80)
            print("Enter earthquake parameters (or 'q' to quit):")
            print("-"*80)
            
            try:
                # Get user inputs
                mag = input("Earthquake Magnitude (e.g., 6.5, 7.0): ")
                if mag.lower() == 'q':
                    break
                mag = float(mag)
                
                ztor = float(input("Depth to Top of Rupture - Ztor (km): "))
                ffm = int(input("Finite Fault Model (0 or 1): "))
                rjb = float(input("Joyner-Boore Distance - Rjb (km): "))
                vs30 = float(input("Shear Wave Velocity - Vs30 (m/s): "))
                
                # Make prediction
                results = predictor.predict(mag, ztor, ffm, rjb, vs30)
                
                # Ask if user wants to plot
                plot_choice = input("\nPlot response spectrum? (y/n): ")
                if plot_choice.lower() == 'y':
                    predictor.plot_response_spectrum(results, 
                                                    save_path=f'response_spectrum_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                
            except ValueError as e:
                print(f"\n❌ Error: Invalid input. Please enter numeric values.")
                continue
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue
        
        print("\n✓ Exiting interactive mode...")
        
    except Exception as e:
        print(f"\n❌ Error loading model: {str(e)}")
        print("Please ensure the model has been trained first by running 'seismic_model_training.py'")

def example_predictions():
    """
    Run example predictions to demonstrate the model.
    """
    print("\n" + "="*80)
    print("RUNNING EXAMPLE PREDICTIONS")
    print("="*80)
    
    try:
        predictor = SeismicPredictor()
        
        # Example 1: Moderate earthquake, close distance
        print("\n📍 Example 1: Moderate Earthquake at Close Distance")
        results1 = predictor.predict(
            earthquake_magnitude=6.5,
            ztor_km=15.0,
            finite_fault_model=0,
            rjb_km=10.0,
            vs30_m_s=500.0
        )
        predictor.plot_response_spectrum(results1, 'example1_spectrum.png', show_plot=False)
        
        # Example 2: Large earthquake, moderate distance
        print("\n📍 Example 2: Large Earthquake at Moderate Distance")
        results2 = predictor.predict(
            earthquake_magnitude=7.5,
            ztor_km=20.0,
            finite_fault_model=0,
            rjb_km=50.0,
            vs30_m_s=450.0
        )
        predictor.plot_response_spectrum(results2, 'example2_spectrum.png', show_plot=False)
        
        # Example 3: Small earthquake, far distance
        print("\n📍 Example 3: Smaller Earthquake at Far Distance")
        results3 = predictor.predict(
            earthquake_magnitude=6.0,
            ztor_km=25.0,
            finite_fault_model=0,
            rjb_km=100.0,
            vs30_m_s=600.0
        )
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
        
        predictor.compare_scenarios(scenarios, 'scenario_comparison.png')
        
        print("\n" + "="*80)
        print("✓ Example predictions completed!")
        print("✓ Check the generated plot files in the current directory")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please ensure the model has been trained first by running 'seismic_model_training.py'")

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("SEISMIC PREDICTION SYSTEM")
    print("="*80)
    print("\nOptions:")
    print("  1. Run example predictions")
    print("  2. Interactive mode (enter your own parameters)")
    print("  3. Exit")
    print("="*80)
    
    choice = input("\nSelect option (1-3): ")
    
    if choice == '1':
        example_predictions()
    elif choice == '2':
        interactive_mode()
    elif choice == '3':
        print("\n✓ Exiting...")
    else:
        print("\n❌ Invalid choice. Running example predictions by default...")
        example_predictions()
