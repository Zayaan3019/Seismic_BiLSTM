"""
Master script - complete seismic model pipeline.

Runs:
1. Train robust model and core figures
2. Generate advanced SDA plots/tables
3. Run example predictions
"""

import subprocess
import sys
from datetime import datetime


def run_script(script_name, description):
    """Run a Python script and report success/failure."""
    print("\n" + "=" * 80)
    print(f"Running: {description}")
    print("=" * 80)

    try:
        subprocess.run([sys.executable, script_name], capture_output=False, text=True, check=True)
        print(f"\nOK: {description} completed successfully")
        return True
    except subprocess.CalledProcessError as err:
        print(f"\nERROR: {description} failed")
        print(f"Error: {err}")
        return False
    except Exception as err:
        print(f"\nERROR: Unexpected failure in {description}")
        print(f"Error: {err}")
        return False


def main():
    print("=" * 80)
    print("COMPLETE SEISMIC MODEL PIPELINE")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis run will:")
    print("  1. Train the robust neural network model")
    print("  2. Generate all SDA-compliant plots/tables")
    print("  3. Run example predictions")
    print("\nEstimated time: 10-20 minutes")
    print("=" * 80)

    response = input("\nProceed? (y/n): ").strip().lower()
    if response != "y":
        print("\nCancelled by user")
        return

    success = run_script("seismic_bilstm_model.py", "Step 1/3: Training BiLSTM model")
    if not success:
        print("\nPipeline failed at Step 1")
        return

    success = run_script("seismic_advanced_plots.py", "Step 2/3: Generating advanced plots and tables")
    if not success:
        print("\nPipeline failed at Step 2")
        return

    print("\n" + "=" * 80)
    print("Step 3/3: Running example predictions")
    print("=" * 80)
    try:
        from seismic_predictor_robust import run_examples

        run_examples()
    except Exception as err:
        print(f"\nERROR: Example prediction stage failed: {err}")
        return

    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll results saved in: sda_results/")
    print("\nGenerated artifacts include:")
    print("  - seismic_model_robust.pkl")
    print("  - model_config.json")
    print("  - fig01_data_distribution.png")
    print("  - fig02_data_used.png")
    print("  - fig03_frequency_plots.png")
    print("  - fig04_architecture.png")
    print("  - fig05_regression_plots.png")
    print("  - fig06_residual_plots.png")
    print("  - fig07_sensitivity_plots.png")
    print("  - fig08_shap_analysis.png")
    print("  - fig09_feature_importance.png")
    print("  - table01_statistics.png")
    print("  - table02_residuals_std.png")
    print("  - table03_performance.png")
    print("  - example1_spectrum.png")
    print("  - example2_spectrum.png")
    print("  - scenario_comparison.png")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
    except Exception as err:
        print(f"\n\nUnexpected error: {err}")
        import traceback

        traceback.print_exc()
