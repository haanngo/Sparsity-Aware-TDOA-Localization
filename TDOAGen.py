import numpy as np
from scenario_setup import simulate_toa, sensors, sources, C, SIGMA_TAU


# Convert TOA measurements into TDOA values.
def generate_tdoa(toa_matrix, ref_sensor_idx=0):
    # Extract the reference sensor's arrival times across all source positions.
    ref_toa = toa_matrix[ref_sensor_idx, :]

    # Subtract the reference times from all sensors to obtain relative delays.
    tdoa_matrix_full = toa_matrix - ref_toa

    # Remove the reference sensor row itself, since TDOA is defined relative to it.
    tdoa_matrix = np.delete(tdoa_matrix_full, ref_sensor_idx, axis=0)
    return tdoa_matrix

def main():
    # Simulate TOA data for one source and for the full source set.
    _, tau_hat_1 = simulate_toa(sensors, sources[:1], C, SIGMA_TAU)
    _, tau_hat_3 = simulate_toa(sensors, sources, C, SIGMA_TAU)

    # Section 3.1: single-source case.
    print("--- TDOA for Section 3.1 (Single Source) ---")
    tdoa_single = generate_tdoa(tau_hat_1, ref_sensor_idx=0)

    # Keep the first source realization as the observation vector y.
    y_single = tdoa_single[:, 0]
    print(f"y_single shape: {y_single.shape}")
    print(y_single)

    # Section 3.2: multi-source case.
    print("\n--- TDOA for Section 3.2 (Multiple Sources) ---")
    tdoa_multi = generate_tdoa(tau_hat_3, ref_sensor_idx=0)

    # Aggregate across source realizations to produce one scalar summary per sensor.
    y_multi = np.sum(tdoa_multi, axis=1)
    print(f"y_multi shape: {y_multi.shape}")
    print(y_multi)


if __name__ == "__main__":
    main()