import csv
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import scenario_setup
import dictionary_builder
import TDOAGen
import cs_solver

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
# Global parameters used throughout the Monte Carlo study.
C = scenario_setup.C
SENSORS = scenario_setup.sensors
TRUE_SOURCES_SINGLE = scenario_setup.sources[:1]
TRUE_SOURCES_MULTI = scenario_setup.sources

GRID_X_MAX = 10
GRID_Y_MAX = 10
GRID_STEP = 1.0
REFERENCE_SENSOR_INDEX = 0

N_TRIALS = 200
SIGMA_VALUES = np.linspace(1e-5, 5e-3, 12)

# Solver and evaluation settings
LASSO_ALPHA = 1e-3
RMSE_FAILURE_THRESHOLD = 1.0
SUCCESS_RATE_THRESHOLD = 0.5
PER_SOURCE_SUCCESS_TOLERANCE = 0.5
OUTPUT_CSV = Path("montecarlo_results.csv")


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
# Compute the noiseless TOA matrix for a given set of sensor and source positions.
def compute_true_toa(sensor_positions, source_positions, c=343.0):
    """Compute the true time-of-arrival matrix for all sensor-source pairs."""
    sensor_count = sensor_positions.shape[0]
    source_count = source_positions.shape[0]
    toa_matrix = np.zeros((sensor_count, source_count))

    for sensor_index in range(sensor_count):
        # Distance from this sensor to every source location.
        diffs = source_positions - sensor_positions[sensor_index]
        distances = np.linalg.norm(diffs, axis=1)
        toa_matrix[sensor_index, :] = distances / c

    return toa_matrix


# Estimate a single source position from a TDOA vector using direct nonlinear optimization over the 2D location space.
def classical_tdoa_estimate_single(tdoa_vector, sensors, ref_idx=0, c=343.0):

    def cost(position):
        position = np.asarray(position, dtype=float)
        distances = np.linalg.norm(sensors - position, axis=1)
        predicted_tdoa = (distances - distances[ref_idx]) / c
        predicted_tdoa_without_ref = np.delete(predicted_tdoa, ref_idx)
        return np.sum((predicted_tdoa_without_ref - tdoa_vector) ** 2)

    initial_guess = np.mean(sensors, axis=0)
    result = minimize(cost, initial_guess, method="L-BFGS-B")

    if not result.success:
        return initial_guess
    return result.x


# Match estimated source positions to the true positions while preventing reuse of the same estimate for multiple true sources.
def match_estimates_to_truth(estimated_positions, true_positions):

    if estimated_positions is None or np.asarray(estimated_positions).size == 0:
        return float("inf"), [float("inf")] * true_positions.shape[0]

    estimates = np.asarray(estimated_positions, dtype=float)
    truths = np.asarray(true_positions, dtype=float)
    used_indices = set()
    per_source_errors = []

    for true_index in range(truths.shape[0]):
        distances = np.linalg.norm(estimates - truths[true_index], axis=1)

        # Reject already assigned estimates so each true source matches at most one estimated position.
        for used_index in used_indices:
            distances[used_index] = np.inf

        best_index = int(np.argmin(distances))
        used_indices.add(best_index)
        per_source_errors.append(float(distances[best_index]))

    per_source_errors = np.asarray(per_source_errors, dtype=float)
    rmse = float(np.sqrt(np.mean(per_source_errors ** 2)))
    return rmse, per_source_errors.tolist()


# ==========================================
# 3. MONTE CARLO EXPERIMENT DRIVER
# ==========================================
# Run repeated noisy localization trials for either one or multiple sources and collect summary statistics across noise levels.
def run_experiment(true_sources, description="scenario"):
    source_count = true_sources.shape[0]

    # Build the dictionary of all possible TDOA signatures on the search grid.
    dictionary, grid_points = dictionary_builder.build_dictionary_matrix(
        SENSORS,
        grid_x_max=GRID_X_MAX,
        grid_y_max=GRID_Y_MAX,
        step=GRID_STEP,
        c=C,
        ref_sensor_idx=REFERENCE_SENSOR_INDEX,
    )

    all_rows = []
    start_all = time.time()

    for sigma in SIGMA_VALUES:
        start_single_sigma = time.time()
        classical_errors = []
        cs_errors = []
        cs_successes = 0

        for _ in range(N_TRIALS):
            # Simulate noisy TOA measurements for this trial and convert them to TDOA.
            _, noisy_timestamps = scenario_setup.simulate_toa(SENSORS, true_sources, C, sigma)
            tdoa_matrix = TDOAGen.generate_tdoa(noisy_timestamps, ref_sensor_idx=REFERENCE_SENSOR_INDEX)

            # For the single-source case, also estimate the result using a direct classical optimization approach for comparison.
            if source_count == 1:
                tdoa_vector = tdoa_matrix[:, 0]
                classical_estimate = classical_tdoa_estimate_single(
                    tdoa_vector,
                    SENSORS,
                    ref_idx=REFERENCE_SENSOR_INDEX,
                    c=C,
                )
                classical_rmse, _ = match_estimates_to_truth(np.atleast_2d(classical_estimate), true_sources)
                classical_errors.append(classical_rmse)

            # Sparse-combinatorial localization solver for the multi-source setup.
            cs_estimate, _ = cs_solver.solve_multi_source_localization(dictionary, tdoa_matrix, grid_points)
            cs_rmse, per_source_errors = match_estimates_to_truth(cs_estimate, true_sources)
            cs_errors.append(cs_rmse)

            # Count a successful localization only if all sources are within tolerance.
            if np.all(np.asarray(per_source_errors) <= PER_SOURCE_SUCCESS_TOLERANCE):
                cs_successes += 1

        cs_mean_rmse = float(np.mean(cs_errors))
        cs_std_rmse = float(np.std(cs_errors))
        success_rate = float(cs_successes) / N_TRIALS

        if source_count == 1:
            classical_mean_rmse = float(np.mean(classical_errors))
            classical_std_rmse = float(np.std(classical_errors))
        else:
            classical_mean_rmse = None
            classical_std_rmse = None

        row = {
            "description": description,
            "sigma_s": sigma,
            "cs_rmse_mean_m": cs_mean_rmse,
            "cs_rmse_std_m": cs_std_rmse,
            "cs_success_rate": success_rate,
            "classical_rmse_mean_m": classical_mean_rmse,
            "classical_rmse_std_m": classical_std_rmse,
        }
        all_rows.append(row)

        elapsed = time.time() - start_single_sigma
        print(
            f"{description} | sigma={sigma:.6e} s | "
            f"CS RMSE={cs_mean_rmse:.3f} m | "
            f"CS succ={success_rate:.3f} | "
            f"time={elapsed:.1f}s"
        )

    total_elapsed = time.time() - start_all
    print(f"Finished {description} in {total_elapsed:.1f}s")
    return all_rows


# ==========================================
# 4. SAVE AND VISUALIZE RESULTS
# ==========================================
# Write the Monte Carlo experiment results to CSV and optionally plot the error curves.
def save_results_csv(rows, out_file=OUTPUT_CSV):
    if not rows:
        raise ValueError("No rows were produced. Nothing to save.")

    fieldnames = list(rows[0].keys())
    with open(out_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved results to {out_file}")


def plot_summary(all_rows):
    if plt is None:
        print("Plotting is disabled because matplotlib is not available.")
        return

    def plot_rows(rows, label):
        sigma_values = [row["sigma_s"] for row in rows]
        rmse_values = [row["cs_rmse_mean_m"] for row in rows]
        plt.plot(sigma_values, rmse_values, marker="o", label=label)

    single_rows = [row for row in all_rows if row["description"] == "single_source"]
    multi_rows = [row for row in all_rows if row["description"] == "multi_source"]

    plt.figure(figsize=(8, 5))
    plot_rows(single_rows, "CS (single)")
    plot_rows(multi_rows, "CS (multi)")
    plt.xscale("log")
    plt.xlabel("sigma_tau (s)")
    plt.ylabel("CS RMSE (m)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.title("Monte Carlo localization error vs. noise level")
    plt.tight_layout()
    plt.show()


# ==========================================
# 5. MAIN ENTRY POINT
# ==========================================
def main():
    all_rows = []

    print("Running single-source experiments...")
    single_rows = run_experiment(TRUE_SOURCES_SINGLE, description="single_source")
    all_rows.extend(single_rows)

    print("\nRunning multi-source experiments...")
    multi_rows = run_experiment(TRUE_SOURCES_MULTI, description="multi_source")
    all_rows.extend(multi_rows)

    save_results_csv(all_rows, OUTPUT_CSV)

    print("\nPlotting results...")
    plot_summary(all_rows)

    print("Done")


if __name__ == "__main__":
    main()
