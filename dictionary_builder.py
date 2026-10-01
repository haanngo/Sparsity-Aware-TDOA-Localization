import numpy as np


# Default sensor layout used when the scenario configuration is not available.
DEFAULT_SENSORS = np.array([
    [1.0, 1.0],
    [1.0, 9.0],
    [9.0, 1.0],
    [9.0, 9.0],
])


# Build a dictionary matrix whose columns correspond to candidate grid positions.
def build_dictionary_matrix(sensors, grid_x_max=10, grid_y_max=10, step=1.0, c=343.0, ref_sensor_idx=0):

    # Create a 2D search grid over the x- and y-axes.
    x_coords = np.arange(0, grid_x_max, step)
    y_coords = np.arange(0, grid_y_max, step)
    xv, yv = np.meshgrid(x_coords, y_coords)

    # Flatten the grid to a list of candidate source positions.
    grid_points = np.vstack([xv.ravel(), yv.ravel()]).T
    N = grid_points.shape[0]
    M = sensors.shape[0]

    # Each column of Psi stores the TDOA vector for one grid point.
    Psi = np.zeros((M - 1, N))

    for n in range(N):
        # Distances from all sensors to the current grid point.
        distances = np.linalg.norm(sensors - grid_points[n], axis=1)

        # Reference distance from the selected sensor to this source location.
        ref_distance = distances[ref_sensor_idx]

        # Compute relative TDOA values with respect to the reference sensor.
        tdoa_relative_full = (distances - ref_distance) / c

        # Remove the reference sensor entry, because it is always zero.
        tdoa_relative = np.delete(tdoa_relative_full, ref_sensor_idx)

        # Store the TDOA vector in the dictionary column for this grid point.
        Psi[:, n] = tdoa_relative

    return Psi, grid_points


# Import the sensor configuration from the scenario setup file.
def load_sensors_from_scenario():
    try:
        from scenario_setup import sensors
        return np.array(sensors, copy=True)
    except ImportError:
        return DEFAULT_SENSORS.copy()


# Example script to generate and inspect the dictionary matrix.
def main():
    sensors = load_sensors_from_scenario()
    Psi, grid_points = build_dictionary_matrix(sensors, step=1.0)

    print(f"Dictionary Matrix Psi shape: {Psi.shape} (Sensors-1 x Grid Points)")
    print(f"Grid Points array shape: {grid_points.shape}")


if __name__ == "__main__":
    main()