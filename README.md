# TDOA Localization Project

This project studies time-difference-of-arrival (TDOA)-based source localization using a small sensor array. The scripts simulate sensor measurements, build a TDOA dictionary over a 2D search grid, and solve localization with sparse reconstruction / matching-based methods.

## Project goal

The code models a simple acoustic localization scenario in which multiple sensors record noisy time-of-arrival (TOA) data and the relative delays between sensors are used to estimate the source position. The project includes:

- TOA/TDOA simulation for a fixed sensor layout
- dictionary generation for a grid-based search space
- single-source and multi-source localization
- Monte Carlo robustness analysis under varying noise levels
# Key files

- `scenario_setup.py` – sensor/source geometry and TOA simulation parameters
- `TDOAGen.py` – conversion from TOA measurements into TDOA observations
- `dictionary_builder.py` – construction of the TDOA dictionary over a spatial grid
- `cs_solver.py` – localization solvers for single-source and multi-source cases
- `montecarlo_noise_test.py` – repeated-noise experiments and evaluation statistics


## Setup

This project is implemented in Python and relies on NumPy, SciPy, scikit-learn, and optionally Matplotlib for plotting.

Install dependencies:

```bash
pip install numpy scipy scikit-learn matplotlib
```

## Usage

Run the main scripts directly from the project directory:

```bash
python scenario_setup.py
python TDOAGen.py
python dictionary_builder.py
python cs_solver.py
python montecarlo_noise_test.py
```

The main localization flow is centered around `cs_solver.py`, which loads the simulated scenario and dictionary and estimates the source coordinates.

## Simulation model

The sensor configuration is defined in `scenario_setup.py` as a 2D rectangular array with four sensors and several candidate source positions. The code computes the theoretical propagation delay from each source to each sensor as:

```python
distance / c
```

where `c` is the speed of sound (set to 343.0 m/s in the project). Gaussian noise is added to produce noisy arrival times, and the TDOA is computed relative to a selected reference sensor.

## Localization approach

The dictionary-based formulation stores the TDOA signature for each grid point. The solver matches the measured TDOA vector against the dictionary and identifies the grid locations that best explain the observations. For sparse localization, the implementation uses a Lasso-based regression model with a fallback correlation-based approximation if the regression solution collapses to zeros.


