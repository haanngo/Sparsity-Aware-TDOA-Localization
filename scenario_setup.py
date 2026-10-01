import numpy as np

# 1. SCENARIO PARAMETERS

C = 343.0  
SIGMA_TAU = 0.00029  # Noise standard deviation 
NUM_SENSORS = 4
NUM_SOURCES = 3

# Define sensor coordinates in meters
sensors = np.array([
    [1.0, 1.0],
    [1.0, 9.0],
    [9.0, 1.0],
    [9.0, 9.0]
])

# Define source coordinates in meters
sources = np.array([
    [3.0, 4.0],  # Source 1
    [7.0, 8.0],  # Source 2
    [5.0, 5.0]   # Source 3
])

# 2. TOA SIMULATION ENGINE

def simulate_toa(sensor_coords, source_coords, c, sigma_tau):
    # Simulates theoretical and noisy time-of-arrival (TOA) measurements.
    # Returns two matrices of shape (num_sensors, num_sources).

    num_sen = sensor_coords.shape[0]
    num_src = source_coords.shape[0]

    # Initialize matrices for tau (theoretical TOA) and tau_hat (noisy TOA)
    tau = np.zeros((num_sen, num_src))
    tau_hat = np.zeros((num_sen, num_src))

    for k in range(num_src):
        for i in range(num_sen):
            # Compute the physical distance between source k and sensor i:
            # ||p_source - p_sensor||_2
            distance = np.linalg.norm(source_coords[k] - sensor_coords[i])

            # 1. Theoretical TOA (tau_{i,k})
            tau[i, k] = distance / c

            # 2. Add Gaussian noise e_i with mean 0 and standard deviation sigma_tau
            noise = np.random.normal(0, sigma_tau)

            # 3. Noisy TOA measurement (\hat{tau}_{i,k})
            tau_hat[i, k] = tau[i, k] + noise

    return tau, tau_hat

if __name__ == "__main__":
    print("--- 1. Single Source TOA ---")
    single_source = sources[:1]  # Use only the first source in the array
    tau_1, tau_hat_1 = simulate_toa(sensors, single_source, C, SIGMA_TAU)
    print(f"Noisy TOA matrix shape: {tau_hat_1.shape} (Sensors x Sources)")
    print(tau_hat_1)

    print("\n--- 2. Multi-Source TOA (3 Sources) ---")
    tau_3, tau_hat_3 = simulate_toa(sensors, sources, C, SIGMA_TAU)
    print(f"Noisy TOA matrix shape: {tau_hat_3.shape} (Sensors x Sources)")
    print(tau_hat_3)