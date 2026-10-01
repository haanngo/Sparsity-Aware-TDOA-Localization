import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')


def plot_localization_results(sensors, true_sources, estimated_sources, title="TDOA Localization", output_path=None):

    # Create a square figure for the localization map.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define the grid used for the sparse dictionary.
    x_coords = np.arange(0, 10, 1.0)
    y_coords = np.arange(0, 10, 1.0)
    xv, yv = np.meshgrid(x_coords, y_coords)

    # Show the candidate grid points in the background.
    ax.scatter(xv, yv, color='lightgray', marker='.', s=10, label='Grid Points (GPs)')

    # Draw the sensors as red triangular markers.
    ax.scatter(sensors[:, 0], sensors[:, 1],
               color='red', marker='^', s=150, label='Sensors (APs)')

    # Draw the true source positions as blue circles.
    ax.scatter(true_sources[:, 0], true_sources[:, 1],
               color='blue', marker='o', s=80, label='True Sources')

    # Draw the estimated locations as hollow green circles.
    ax.scatter(estimated_sources[:, 0], estimated_sources[:, 1],
               edgecolors='green', facecolors='none', marker='o', s=300,
               linewidths=2, label='Estimated Locations')

    # Set the plotting area and axis ticks.
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_xlabel('x-axis (meters)', fontsize=12)
    ax.set_ylabel('y-axis (meters)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add a legend in the upper-right corner.
    ax.legend(loc='upper right', frameon=True, shadow=True)

    # Save the figure as a PNG if an output path was given, then display it.
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    plt.tight_layout()
    plt.show()


def load_module(name, filename):

    # Build the full path to the requested file in the same directory.
    module_path = Path(__file__).with_name(filename)
    if not module_path.exists():
        raise FileNotFoundError(f"Missing module: {module_path}")

    # Create a module specification and load it from its file.
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():

    # Import the scenario and solver modules.
    scenario = load_module("scenario_engine", "scenario_setup.py")
    cs_solver = load_module("cs_solver", "cs_solver.py")

    # Extract sensor and source definitions from the scenario.
    sensors = scenario.sensors
    sources = scenario.sources

    # Build the dictionary matrix and the corresponding grid points.
    dictionary_builder = load_module("dictionary_builder", "dictionary_builder.py")
    Psi, grid_points = dictionary_builder.build_dictionary_matrix(sensors, step=1.0)

    # Generate time-of-arrival measurements for one source and for the full source set.
    _, tau_hat_1 = scenario.simulate_toa(sensors, sources[:1], scenario.C, scenario.SIGMA_TAU)
    _, tau_hat_3 = scenario.simulate_toa(sensors, sources, scenario.C, scenario.SIGMA_TAU)

    # Convert TOA data to TDOA data using the first sensor as the reference.
    tdoa_gen = load_module("tdoa_gen", "TDOAGen.py")
    tdoa_single = tdoa_gen.generate_tdoa(tau_hat_1, ref_sensor_idx=0)
    y_single = tdoa_single[:, 0]
    tdoa_multi = tdoa_gen.generate_tdoa(tau_hat_3, ref_sensor_idx=0)

    # Solve the localization problem for the single-source and multi-source cases.
    est_single, _ = cs_solver.solve_single_source_localization(Psi, y_single, grid_points)
    est_multi, _ = cs_solver.solve_multi_source_localization(Psi, tdoa_multi, grid_points)

    # Visualize the single-source localization result.
    plot_localization_results(
        sensors=sensors,
        true_sources=sources[:1],
        estimated_sources=est_single,
        title="Sparsity-Aware TDOA Localization (Single Source)",
        output_path=Path(__file__).with_name("tdoa_localization_single_source.png")
    )

    # Visualize the multi-source localization result.
    plot_localization_results(
        sensors=sensors,
        true_sources=sources,
        estimated_sources=est_multi,
        title="Sparsity-Aware TDOA Localization (3 Sources)",
        output_path=Path(__file__).with_name("tdoa_localization_multi_source.png")
    )


if __name__ == "__main__":
    # Run the main workflow only when the file is executed directly.
    main()