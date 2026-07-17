# Molecular Dynamics & Monte Carlo Simulation of Lennard-Jones Fluids

A C++ simulation framework for studying the thermodynamic and structural behavior of an Argon-like Lennard-Jones fluid, combining a velocity-Verlet molecular dynamics (MD) integrator with an experimental Metropolis Monte Carlo (MC) sampler and pair distribution function (PDF/RDF) analysis tools.

## Overview

This project simulates a system of particles interacting via the Lennard-Jones potential under periodic boundary conditions, using a cell-list (linked-list) neighbor search for efficient force evaluation. It supports two independent simulation modes:

- **Molecular Dynamics** — deterministic time integration via the velocity-Verlet scheme, used to extract thermodynamic properties (potential/kinetic energy, virial, temperature) from the system's real dynamics.
- **Monte Carlo** — stochastic Metropolis sampling of particle configurations at fixed temperature, used to explore equilibrium statistics and phase behavior independently of dynamics.

Both modes share the same underlying particle system, cell-list infrastructure, and Lennard-Jones potential implementation.

## Features

- Reduced-unit Lennard-Jones potential and force (with cutoff and potential shift) calibrated for Argon (`Sigma`, `Epsilon`, `Mass` constants).
- Cell-list spatial decomposition with periodic boundary conditions for O(N) neighbor search instead of O(N²) pairwise evaluation.
- Velocity-Verlet time integration with configurable timestep, number of steps, and output resolution.
- Metropolis Monte Carlo sweep with adaptive step-size (`dr`) control targeting a configurable acceptance rate.
- Pair distribution function (radial distribution function, g(r)) analysis of output trajectories/configurations to characterize structural order (solid/liquid/gas phases).
- JSON-based configuration support via `nlohmann/json`.

## Project Structure

```
.
├── verlet.hpp          # Core simulation engine: Atom, Cell/linked-list, System classes,
│                       # Lennard-Jones potential, Verlet integrator
├── verlet_main.cpp     # MD simulation entry point (velocity-Verlet)
├── mc_main.cpp         # Monte Carlo simulation entry point (Metropolis sampling)
├── pdf_analysis/       # Pair distribution function computation and plotting scripts
├── CMakeLists.txt      # Build configuration
└── README.md
```

*(Adjust file/folder names above to match your actual repository layout.)*

## Building

This project uses CMake and requires a C++17-capable compiler.

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Dependencies

- [nlohmann/json](https://github.com/nlohmann/json) for configuration parsing
- A C++17 (or later) compiler (tested with Clang on macOS)
- CMake ≥ 3.15

## Usage

### Molecular Dynamics Simulation

```bash
./verlet_sim
```

Runs velocity-Verlet integration for `SWEEPS` steps at initial temperature `T_INIT`, writing atom positions, potential/kinetic energy, and virial data to a trajectory file at the configured `resolution`. Key parameters (system size, atom count, timestep, cell-grid resolution) are set as compile-time constants at the top of the source file and can be adjusted before building.

### Monte Carlo Simulation (experimental)

```bash
./mc_sim
```

Performs `RUNUP` equilibration sweeps followed by `SWEEPS` production sweeps of single-particle Metropolis moves, with automatic step-size (`dr`) tuning to maintain a target acceptance rate (~20%). Potential energy per sweep is written to `MCdata.txt`.

> **Note:** The single-particle Metropolis scheme suffers from severe critical slowing down near phase transitions. A cluster-based algorithm (e.g. the geometric cluster algorithm, the off-lattice analogue of the Wolff algorithm) is a planned/possible extension to improve sampling efficiency near coexistence points.

### Pair Distribution Function Analysis

Given a trajectory or configuration snapshot, the PDF analysis tools compute g(r) by binning pairwise distances (with periodic boundary corrections) and normalizing against the ideal-gas pair density, allowing identification of crystalline, liquid, and gas-like structural signatures.

## Physical Model

- **Potential**: Lennard-Jones 12-6 potential with cutoff at r² = 6.25 (in reduced units) and a constant energy shift to ensure continuity at the cutoff.
- **Units**: Reduced (Lennard-Jones) units based on Argon parameters — length in units of σ = 0.339 nm, energy in units of ε/k_B = 137.9 K, mass in Dalton.
- **Boundary conditions**: Fully periodic in all three dimensions, with minimum-image convention applied via `PeriodicDifference`.
- **Neighbor search**: Linked-cell method (head/next index arrays) for O(N) force evaluation, rebuilt each step after position updates.

## Known Limitations / Roadmap

- Monte Carlo sampler currently uses single-atom Metropolis moves only; cluster moves for improved performance near phase transitions are not yet implemented.
- No parallelization (OpenMP/MPI) yet — single-threaded execution.
- Cell-list grid resolution (`BOX_N`) is fixed at compile time via macros; runtime-configurable grids are a possible improvement.

## License

*(Add your chosen license here, e.g. MIT.)*

## Acknowledgments

- Lennard-Jones parameters for Argon based on standard literature values.
- JSON parsing via [nlohmann/json](https://github.com/nlohmann/json).
