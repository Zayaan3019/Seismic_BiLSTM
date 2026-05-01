"""
Seismic Ground Motion Prediction Interface
===========================================
Interactive interface for making predictions with the trained model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path

class SeismicPredictor:
    """Prediction interface for seismic ground motion model."""
    
    def __init__(self, model_dir='sda_results'):
        print("="*80)
        print("SEISMIC GROUND MOTION PREDICTION SYSTEM")
        print("="*80)
        print("\nLoading model...")
        
        self.model_dir = Path(model_dir)
        
        # Load model and scalers
        self.model = joblib.load(self.model_dir / 'seismic_model_robust.pkl')
        self.scaler_X = joblib.load(self.model_dir / 'scaler_X.pkl')
        self.scaler_y = joblib.load(self.model_dir / 'scaler_y.pkl')
        
        # Load configuration
        with open(self.model_dir / 'model_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.input_features = self.config['input_features']
        self.output_features = self.config['output_features']
        
        print(f"✓ Model loaded successfully")
        print(f"\nModel Performance (Test Set):")
        print(f"  • R² Score: {self.config['test_metrics']['r2']:.4f}")
        print(f"  • RMSE: {self.config['test_metrics']['rmse']:.4f}")
        print(f"  • MAE: {self.config['test_metrics']['mae']:.4f}")
        print("="*80)
    
    def predict(self, magnitude, ztor_km, finite_fault_model, rjb_km, vs30_m_s, 
                verbose=True):
        """
        Predict spectral accelerations for given seismic parameters.
        
        Parameters:
        -----------
        magnitude : float
            Earthquake magnitude (Mw)
        ztor_km : float
            Depth to top of rupture (km)
        finite_fault_model : int
            Finite fault model flag (0 or 1)
        rjb_km : float
            Joyner-Boore distance (km)
        vs30_m_s : float
            Shear wave velocity (m/s)
        verbose : bool
            Print prediction details
            
        Returns:
        --------
        dict : Predicted spectral accelerations
        """
        
        # Create input array
        epsilon = 1e-10
        log_vs30 = np.log(vs30_m_s + epsilon)
        log_rjb = np.log(rjb_km + epsilon)
        
        X_input = np.array([[
            magnitude,
            ztor_km,
            finite_fault_model,
            rjb_km,
            log_vs30,
            log_rjb
        ]])
        
        # Scale input
        X_scaled = self.scaler_X.transform(X_input)
        
        # Make prediction (in log space)
        y_scaled = self.model.predict(X_scaled)
        
        # Inverse transform to get actual values
        y_pred = self.scaler_y.inverse_transform(y_scaled.reshape(1, -1))
        
        # Convert from log space to linear space
        y_pred_linear = np.exp(y_pred[0])
        
        # Create results dictionary
        results = {}
        for i, feature_name in enumerate(self.output_features):
            results[feature_name] = float(y_pred_linear[i])
        
        if verbose:
            print("\n" + "="*80)
            print("PREDICTION RESULTS")
            print("="*80)
            print(f"\nInput Parameters:")
            print(f"  • Earthquake Magnitude (Mw): {magnitude}")
            print(f"  • Depth to Top of Rupture: {ztor_km} km")
            print(f"  • Finite Fault Model: {finite_fault_model}")
            print(f"  • Joyner-Boore Distance: {rjb_km} km")
            print(f"  • Shear Wave Velocity (Vs30): {vs30_m_s} m/s")
            print(f"\nPredicted Spectral Accelerations (g):")
            print(f"  {'-'*70}")
            print(f"  {'Period':<25} {'PSA (g)':<20}")
            print(f"  {'-'*70}")
            
            for feature_name, value in results.items():
                # Parse period from feature name
                if 'PGA' in feature_name:
                    period_str = "PGA"
                else:
                    period_str = feature_name.replace('PSA', 'T=').replace('pt', '.').replace('S', 's')
                print(f"  {period_str:<25} {value:.6f}")
            
            print(f"  {'-'*70}")
            print("="*80)
        
        return results
    
    def plot_response_spectrum(self, predictions, save_path=None, show_plot=True):
        """Plot response spectrum."""
        
        # Extract periods and accelerations
        periods = []
        accelerations = []
        
        for feature_name, value in predictions.items():
            if 'PGA' in feature_name:
                periods.insert(0, 0.01)  # Approximate for PGA
                accelerations.insert(0, value)
            else:
                # Parse period from feature name
                period_str = feature_name.replace('PSA', '').replace('pt', '.').replace('S', '')
                try:
                    period = float(period_str)
                    periods.append(period)
                    accelerations.append(value)
                except:
                    pass
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.loglog(periods, accelerations, 'b-o', linewidth=2, markersize=6, 
                  label='Predicted Spectrum')
        ax.scatter([periods[0]], [accelerations[0]], s=200, c='red', marker='*', 
                   label='PGA', zorder=5)
        
        ax.set_xlabel('Period (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Spectral Acceleration (g)', fontsize=14, fontweight='bold')
        ax.set_title('Predicted Response Spectrum', fontsize=16, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig, ax


def run_examples():
    """Run example predictions."""
    
    print("\n" + "="*80)
    print("RUNNING EXAMPLE PREDICTIONS")
    print("="*80)
    
    # Initialize predictor
    predictor = SeismicPredictor()
    
    # Example 1: Moderate earthquake at close distance
    print("\n📍 Example 1: Moderate Earthquake at Close Distance")
    print("-" * 80)
    results1 = predictor.predict(
        magnitude=6.5,
        ztor_km=15.0,
        finite_fault_model=0,
        rjb_km=10.0,
        vs30_m_s=500.0
    )
    predictor.plot_response_spectrum(
        results1, 
        save_path='sda_results/example1_response_spectrum.png',
        show_plot=False
    )
    
    # Example 2: Large earthquake at moderate distance
    print("\n📍 Example 2: Large Earthquake at Moderate Distance")
    print("-" * 80)
    results2 = predictor.predict(
        magnitude=7.5,
        ztor_km=20.0,
        finite_fault_model=0,
        rjb_km=50.0,
        vs30_m_s=450.0
    )
    predictor.plot_response_spectrum(
        results2,
        save_path='sda_results/example2_response_spectrum.png',
        show_plot=False
    )
    
    # Example 3: Smaller earthquake at far distance
    print("\n📍 Example 3: Smaller Earthquake at Far Distance")
    print("-" * 80)
    results3 = predictor.predict(
        magnitude=6.0,
        ztor_km=25.0,
        finite_fault_model=0,
        rjb_km=100.0,
        vs30_m_s=600.0
    )
    predictor.plot_response_spectrum(
        results3,
        save_path='sda_results/example3_response_spectrum.png',
        show_plot=False
    )
    
    print("\n" + "="*80)
    print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated response spectrum plots:")
    print("  • sda_results/example1_response_spectrum.png")
    print("  • sda_results/example2_response_spectrum.png")
    print("  • sda_results/example3_response_spectrum.png")
    print("="*80)


if __name__ == "__main__":
    run_examples()
