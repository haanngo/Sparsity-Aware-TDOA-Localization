import importlib.util
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso


def solve_single_source_localization(Psi, y, grid_points):
    # Return the grid point whose TDOA fingerprint best matches y.
    Psi = np.asarray(Psi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    grid_points = np.asarray(grid_points)

    if Psi.ndim != 2 or Psi.shape[0] != y.size:
        raise ValueError("Psi must have one row per TDOA measurement in y.")
    if Psi.shape[1] != grid_points.shape[0]:
        raise ValueError("Psi columns must correspond to grid_points.")

    residuals = np.linalg.norm(Psi - y[:, None], axis=0)
    best_index = int(np.argmin(residuals))
    scores = -residuals
    return grid_points[[best_index]], scores


def solve_multi_source_localization(Psi, y_matrix, grid_points):
    # Localize independently observed sources from a (TDOA, source) matrix.
    y_matrix = np.asarray(y_matrix, dtype=float)
    if y_matrix.ndim != 2:
        raise ValueError("y_matrix must have shape (TDOA measurements, sources).")

    estimates = []
    scores = []
    for source_index in range(y_matrix.shape[1]):
        estimate, score = solve_single_source_localization(
            Psi, y_matrix[:, source_index], grid_points
        )
        estimates.append(estimate[0])
        scores.append(score)

    return np.asarray(estimates), np.vstack(scores).T


def solve_cs_localization(Psi, y, grid_points, num_sources=1, alpha=1e-4, c=343.0):
   
    # Solves the sparse TDOA localization problem using Lasso (l1-norm minimization).

    # 1. Scale the data to meters   
    Psi_m = Psi * c
    y_m = y * c

    # 2. Normalize the columns of the dictionary matrix (L2-norm = 1)
    # This is required so every grid point is weighted fairly
    norms = np.linalg.norm(Psi_m, axis=0)
    norms[norms == 0] = 1e-10 
    Psi_norm = Psi_m / norms

    # 3. Initialize Lasso
    # Reduce alpha to a very small value to preserve the signal
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=50000)
    lasso.fit(Psi_norm, y_m)

    # 4. Take the absolute values of the weights
    theta_abs = np.abs(lasso.coef_)

    # 5. FALLBACK MECHANISM
    # If noise causes Lasso to zero out the entire vector, use the dot product instead
    if np.all(theta_abs == 0):
        print(f"[Fallback] Lasso was zeroed with alpha={alpha}; using correlation matching.")
        theta_abs = np.abs(Psi_norm.T @ y_m)

    # 6. Extract the K highest peaks
    peak_indices = np.argsort(theta_abs)[-num_sources:]
    estimated_coords = grid_points[peak_indices]

    return estimated_coords, theta_abs


def load_module(name, filename):
    module_path = Path(__file__).with_name(filename)
    if not module_path.exists():
        raise FileNotFoundError(f"Missing module: {module_path}")

    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    scenario = load_module("scenario_setup", "scenario_setup.py")
    dictionary_builder = load_module("dictionary_builder", "dictionary_builder.py")
    tdoa_gen = load_module("tdoa_gen", "TDOAGen.py")

    sensors = scenario.sensors
    c_sound = scenario.C
    Psi, grid_points = dictionary_builder.build_dictionary_matrix(sensors, step=1.0)

    # 1. Generate noisy TOA measurements
    _, tau_hat_1 = scenario.simulate_toa(sensors, scenario.sources[:1], c_sound, scenario.SIGMA_TAU)
    _, tau_hat_3 = scenario.simulate_toa(sensors, scenario.sources, c_sound, scenario.SIGMA_TAU)

    # 2. Create the TDOA measurement vector y
    tdoa_single = tdoa_gen.generate_tdoa(tau_hat_1, ref_sensor_idx=0)
    y_single = tdoa_single[:, 0]

    tdoa_multi = tdoa_gen.generate_tdoa(tau_hat_3, ref_sensor_idx=0)

    print("--- 1. CS Localization: Single Source (Sec 3.1) ---")
    est_single, theta_single = solve_single_source_localization(Psi, y_single, grid_points)
    print(f"True source position:\n{scenario.sources[:1]}")
    print(f"Estimated position (Single Source):\n{est_single}\n")

    print("--- 2. CS Localization: Multiple Sources (Sec 3.2) ---")
    est_multi, theta_multi = solve_multi_source_localization(Psi, tdoa_multi, grid_points)
    print(f"True source positions:\n{scenario.sources}")
    print(f"Estimated position (Multi-Source):\n{est_multi}")


if __name__ == "__main__":
    main()