"""Seismic ground motion predictor interface for the SDA BiLSTM pipeline."""

import json
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sda_pipeline_utils import OUTPUT_COLUMNS, output_to_label, MIN_RJB_KM


class SeismicPredictorRobust:
    """Production-ready seismic ground motion predictor using the BiLSTM GMM."""

    def __init__(self, model_dir="sda_results"):
        print("=" * 80)
        print("ROBUST SEISMIC GROUND MOTION PREDICTION SYSTEM")
        print("=" * 80)
        print("[Loading model]")

        self.model    = joblib.load(f"{model_dir}/seismic_model_robust.pkl")
        self.scaler_X = joblib.load(f"{model_dir}/scaler_X.pkl")
        self.scaler_y = joblib.load(f"{model_dir}/scaler_y.pkl")

        with open(f"{model_dir}/model_config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.input_features  = self.config["input_features"]
        self.output_features = self.config["output_features"]

        print("   Model loaded successfully")
        print("\n   Test performance:")
        print(f"   - R2 score: {self.config['test_metrics']['r2']:.6f}")
        print(f"   - RMSE    : {self.config['test_metrics']['rmse']:.6f}")
        print(f"   - MAE     : {self.config['test_metrics']['mae']:.6f}")
        print(f"\n   Architecture: {self.config['architecture']}")
        print(f"   Training iterations: {self.config['iterations']}")
        print("=" * 80)

    def predict(
        self,
        earthquake_magnitude,
        ztor_km,
        finite_fault_model,
        rjb_km,
        vs30_m_s,
        verbose=True,
    ):
        """Predict PGA and PSA values for one scenario.

        Parameters
        ----------
        earthquake_magnitude : float
            Moment magnitude Mw.
        ztor_km : float
            Depth to top of rupture (km).
        finite_fault_model : int
            Binary flag: 1 = finite fault model available, 0 = point source.
        rjb_km : float
            Joyner-Boore distance (km). Values of 0 are allowed (floored to
            MIN_RJB_KM = 0.1 km before log-transformation to avoid -inf).
        vs30_m_s : float
            Time-averaged shear-wave velocity in top 30 m (m/s). Must be > 0.
        verbose : bool
            Print formatted prediction table if True.

        Returns
        -------
        dict[str, float]
            Predicted intensity measures in linear (g) space, keyed by column name.
        """
        ffm = int(finite_fault_model)
        if ffm not in (0, 1):
            raise ValueError("finite_fault_model must be 0 or 1")
        if rjb_km < 0:
            raise ValueError(f"rjb_km must be >= 0 (got {rjb_km})")
        if vs30_m_s <= 0:
            raise ValueError(f"vs30_m_s must be > 0 (got {vs30_m_s})")

        rjb_floor = max(float(rjb_km), MIN_RJB_KM)
        mw        = float(earthquake_magnitude)
        log_rjb   = float(np.log(rjb_floor))

        feature_values = {
            "Earthquake_Magnitude": mw,
            "Ztor_km":             float(ztor_km),
            "Finite_Fault_Model":  ffm,
            "Rjb_km":             float(rjb_km),
            "log_Vs30":           float(np.log(float(vs30_m_s))),
            "log_Rjb":            log_rjb,
            "Mw_sq":              mw ** 2,
            "Mw_logR":            mw * log_rjb,
        }

        x_input  = np.array([[feature_values[f] for f in self.input_features]], dtype=float)
        x_scaled = self.scaler_X.transform(x_input)
        y_scaled = self.model.predict(x_scaled)
        y_log    = self.scaler_y.inverse_transform(y_scaled.reshape(1, -1))[0]
        y_actual = np.exp(y_log)

        results = {name: float(y_actual[i]) for i, name in enumerate(self.output_features)}

        if verbose:
            print("\n" + "=" * 80)
            print("PREDICTION RESULTS")
            print("=" * 80)
            print("Input parameters:")
            print(f"   - Earthquake magnitude (Mw): {earthquake_magnitude}")
            print(f"   - Depth to top of rupture  : {ztor_km} km")
            print(f"   - Finite Fault Model (FFM) : {ffm}")
            print(f"   - Joyner-Boore distance    : {rjb_km} km")
            print(f"   - Vs30                     : {vs30_m_s} m/s")
            print("\nPredicted values:")
            print(f"   {'Parameter':<16}{'Value (g)':>16}{'ln(value)':>16}")
            print("   " + "-" * 48)
            for name, value in results.items():
                print(f"   {output_to_label(name):<16}{value:>16.6e}{np.log(value):>16.6f}")
            print("=" * 80 + "\n")

        return results

    def plot_response_spectrum(
        self,
        predictions,
        save_path=None,
        show_plot=True,
        compare_scenarios=None,
    ):
        """Plot PSA response spectrum over the full 0.01–10 s period range.

        PGA is shown as a star at T=0.01 s.  Additional scenario curves can be
        overlaid via the ``compare_scenarios`` list of prediction dicts.
        """
        spectral_features = [
            f for f in self.output_features if f.startswith("T") and f.endswith("S")
        ]

        periods       = []
        accelerations = []
        for feature in spectral_features:
            period = float(feature.replace("T", "").replace("S", "").replace("pt", "."))
            periods.append(period)
            accelerations.append(predictions[feature])

        plt.figure(figsize=(12, 8))
        plt.plot(periods, accelerations, "b-o", linewidth=2.5, markersize=6,
                 label="Predicted PSA")

        if "PGA_g" in predictions:
            plt.scatter(
                [0.01],
                [predictions["PGA_g"]],
                s=180,
                c="red",
                marker="*",
                label="PGA",
                edgecolors="black",
                linewidths=1.0,
                zorder=4,
            )

        if compare_scenarios:
            colors = ["green", "orange", "purple", "brown"]
            for i, scenario in enumerate(compare_scenarios):
                p, a = [], []
                for feature in spectral_features:
                    if feature in scenario:
                        p.append(float(feature.replace("T", "").replace("S", "").replace("pt", ".")))
                        a.append(scenario[feature])
                if p:
                    plt.plot(p, a, "--s", color=colors[i % len(colors)],
                             linewidth=2, markersize=5, label=f"Scenario {i + 1}")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Period (seconds)", fontsize=14, fontweight="bold")
        plt.ylabel("Spectral acceleration (g)", fontsize=14, fontweight="bold")
        plt.title("Predicted response spectrum", fontsize=16, fontweight="bold")
        plt.grid(True, which="both", alpha=0.3, linestyle="--")
        plt.legend(fontsize=11, loc="best")
        plt.xlim([0.01, 10.0])
        plt.ylim([max(min(accelerations) / 2, 1e-6), max(accelerations) * 2])

        info_text = (
            f"Model: BiLSTM GMM\n"
            f"R² = {self.config['test_metrics']['r2']:.3f}"
        )
        plt.text(
            0.02, 0.98, info_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"   Response spectrum saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def compare_scenarios(
        self,
        scenarios_list,
        labels=None,
        save_path="scenario_comparison.png",
    ):
        """Overlay response spectra for multiple seismic scenarios on one plot.

        Each entry in ``scenarios_list`` must contain keys:
            magnitude, ztor, finite_fault_model (or ffm), rjb, vs30
        """
        print("\nComparing multiple scenarios...")

        predictions_list = []
        for scenario in scenarios_list:
            pred = self.predict(
                earthquake_magnitude=scenario["magnitude"],
                ztor_km=scenario["ztor"],
                finite_fault_model=scenario.get("finite_fault_model",
                                                scenario.get("ffm", 1)),
                rjb_km=scenario["rjb"],
                vs30_m_s=scenario["vs30"],
                verbose=False,
            )
            predictions_list.append(pred)

        spectral_features = [
            f for f in self.output_features if f.startswith("T") and f.endswith("S")
        ]
        periods = [
            float(f.replace("T", "").replace("S", "").replace("pt", "."))
            for f in spectral_features
        ]

        plt.figure(figsize=(14, 9))
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios_list)))

        for i, (pred, scenario) in enumerate(zip(predictions_list, scenarios_list)):
            acc = [pred[f] for f in spectral_features]
            if labels and i < len(labels):
                label = labels[i]
            else:
                ffm_val = scenario.get("finite_fault_model", scenario.get("ffm", 1))
                label = (
                    f"Mw={scenario['magnitude']}, Rjb={scenario['rjb']:.0f}km, "
                    f"Vs30={scenario['vs30']:.0f}m/s, FFM={ffm_val}"
                )
            plt.plot(periods, acc, "-o", linewidth=2.3, color=colors[i],
                     markersize=5.5, label=label)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Period (seconds)", fontsize=14, fontweight="bold")
        plt.ylabel("Spectral acceleration (g)", fontsize=14, fontweight="bold")
        plt.title("Scenario comparison — response spectra", fontsize=16, fontweight="bold")
        plt.grid(True, which="both", alpha=0.3, linestyle="--")
        plt.legend(fontsize=10, loc="best")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   Comparison saved to {save_path}")
        plt.close()


# ---------------------------------------------------------------------------
# Canned examples
# ---------------------------------------------------------------------------

def run_examples():
    """Run representative prediction examples and save result plots."""
    print("\n" + "=" * 80)
    print("RUNNING EXAMPLE PREDICTIONS")
    print("=" * 80)

    predictor = SeismicPredictorRobust()

    print("\nExample 1: Moderate subduction earthquake at close distance")
    results1 = predictor.predict(
        earthquake_magnitude=6.5,
        ztor_km=15.0,
        finite_fault_model=1,
        rjb_km=10.0,
        vs30_m_s=500.0,
    )
    predictor.plot_response_spectrum(
        results1, "sda_results/example1_spectrum.png", show_plot=False
    )

    print("\nExample 2: Large interface earthquake at moderate distance")
    results2 = predictor.predict(
        earthquake_magnitude=7.5,
        ztor_km=20.0,
        finite_fault_model=1,
        rjb_km=50.0,
        vs30_m_s=450.0,
    )
    predictor.plot_response_spectrum(
        results2, "sda_results/example2_spectrum.png", show_plot=False
    )

    print("\nExample 3: Deep intraslab earthquake (no finite fault model)")
    results3 = predictor.predict(
        earthquake_magnitude=6.8,
        ztor_km=80.0,
        finite_fault_model=0,
        rjb_km=30.0,
        vs30_m_s=760.0,
    )
    predictor.plot_response_spectrum(
        results3, "sda_results/example3_spectrum.png", show_plot=False
    )

    print("\nExample 4: Scenario comparison — distance effect (Mw=7.0)")
    scenarios = [
        {"magnitude": 7.0, "ztor": 20, "finite_fault_model": 1, "rjb":  10, "vs30": 760},
        {"magnitude": 7.0, "ztor": 20, "finite_fault_model": 1, "rjb":  30, "vs30": 760},
        {"magnitude": 7.0, "ztor": 20, "finite_fault_model": 1, "rjb": 100, "vs30": 760},
        {"magnitude": 7.0, "ztor": 20, "finite_fault_model": 1, "rjb": 300, "vs30": 760},
    ]
    scenario_labels = [
        "Mw=7.0, Rjb=10 km",
        "Mw=7.0, Rjb=30 km",
        "Mw=7.0, Rjb=100 km",
        "Mw=7.0, Rjb=300 km",
    ]
    predictor.compare_scenarios(
        scenarios,
        labels=scenario_labels,
        save_path="sda_results/scenario_comparison.png",
    )

    print("\n" + "=" * 80)
    print("All examples completed successfully")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode():
    """Interactive prediction mode — prompts user for scenario parameters."""
    print("\n" + "=" * 80)
    print("INTERACTIVE PREDICTION MODE")
    print("=" * 80)
    print("Finite Fault Model (FFM): 1 = finite fault model available, 0 = point source")

    predictor = SeismicPredictorRobust()

    while True:
        print("\n" + "-" * 80)
        print("Enter earthquake parameters (or 'q' to quit):")
        print("-" * 80)

        try:
            mag = input("Earthquake magnitude (Mw): ")
            if mag.lower() == "q":
                break
            mag  = float(mag)
            ztor = float(input("Depth to top of rupture (km): "))
            ffm  = int(input("Finite Fault Model (0 = point source, 1 = finite fault): "))
            rjb  = float(input("Joyner-Boore distance (km): "))
            vs30 = float(input("Vs30 (m/s): "))

            results = predictor.predict(
                earthquake_magnitude=mag,
                ztor_km=ztor,
                finite_fault_model=ffm,
                rjb_km=rjb,
                vs30_m_s=vs30,
            )

            plot_choice = input("\nPlot response spectrum? (y/n): ")
            if plot_choice.lower() == "y":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                predictor.plot_response_spectrum(
                    results,
                    save_path=f"sda_results/spectrum_{timestamp}.png",
                )

        except ValueError as e:
            print(f"\nInvalid input: {e}")
            continue
        except Exception as e:
            print(f"\nError: {e}")
            continue

    print("\nExiting...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SEISMIC PREDICTION SYSTEM — PRODUCTION INTERFACE")
    print("=" * 80)
    print("\nOptions:")
    print("  1. Run example predictions")
    print("  2. Interactive mode")
    print("  3. Exit")
    print("=" * 80)

    try:
        choice = input("\nSelect option (1-3): ")
        if choice == "1":
            run_examples()
        elif choice == "2":
            interactive_mode()
        else:
            print("\nExiting...")
    except KeyboardInterrupt:
        print("\n\nExiting...")
