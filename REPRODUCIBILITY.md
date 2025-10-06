# Reproducibility Guide

## Detailed Overview: `src/` Directory

The `src/` directory contains the core Python implementation of the Prime Annealer framework, including:

- **Spectral partitioning logic** (`heat_kernel_partition.py`)
- **CNF utilities and partitioners** (`cnf_partition.py`)
- **Experiment drivers** (`run_experiments.py`)
- **Benchmarking and validation scripts** (`benchmarks/`)
- **Verification of mathematical conjectures** (`verify_conjectures.py`)
- **Real-world demos** (`demos/`)

### Core Modules

- `heat_kernel_partition.py`: Implements the main spectral partitioning model using heat kernel objectives. Includes:
  - `sieve_primes`: Generates prime weights for graph construction.
  - `HeatKernelPartitionModel`: Models a circular graph with cut-dependent Laplacian.
  - Energy functions: `balance_entropy_penalty`, `multiplicative_weights_functional`, `composite_energy`.

- `cnf_partition.py`: Utilities for partitioning CNF formulas using spectral energies. Includes:
  - `CNFFormula`, `ClauseGraph`, `CNFPartitioner`, `PartitionSolution` classes.
  - Functions for loading DIMACS files, building conflict graphs, and writing partitioned CNFs.

- `run_experiments.py`: Batch driver for running partitioning experiments on example and random CNF formulas. Automates loading, partitioning, and saving results for multiple cases.

- `verify_conjectures.py`: Comprehensive script to empirically test and verify the main mathematical conjectures (e.g., spectral action equivalence). Runs multiple parameter sweeps and prints statistical summaries.

### Benchmarking & Validation

- `benchmarks/`: Contains scripts for benchmarking the spectral framework against Google OR-Tools and validating robustness.
  - `benchmark_vs_ortools.py`: Head-to-head comparison on classic combinatorial problems (partitioning, TSP, coloring).
  - `validate_robustness.py`: Tests robustness, constraint satisfaction, generalization, and speed.
  - `README.md`: Summarizes benchmark results and validation methodology.

### Demos & Applications

- `demos/`: Real-world application examples.
  - `cluster_ml_datasets.py`: Demonstrates spectral clustering on standard ML datasets (Iris, Wine, Digits, synthetic blobs).
  - `README.md`: Instructions and explanation of demo scripts.

### Additional Utilities

- `gap_bounds_demo.py`, `hpc_partition_demo.py`, `solve_spectral_partition.py`: Specialized scripts for demonstrating gap bounds, high-performance partitioning, and toy solvers.
- `test_heat_kernel_partition.py`: Unit and regression tests for the heat kernel partitioning logic.
- `train_energy_model.py`: (If present) Trains a neural surrogate model for energy prediction.

---

**How to Use:**

- For theoretical verification: Run `verify_conjectures.py`.
- For benchmarking and validation: Use scripts in `benchmarks/`.
- For practical partitioning: Use `run_experiments.py` or import `heat_kernel_partition.py` and `cnf_partition.py` in your own scripts.
- For real-world demos: See `demos/cluster_ml_datasets.py`.

---

This guide explains how to reproduce the main theoretical and empirical results from the Prime Annealer repository.

## 1. Environment Setup

- Clone the repository:
  ```bash
  git clone https://github.com/sethuiyer/prime-annealer.git
  cd prime-annealer
  ```
- Install Python dependencies:
  ```bash
  pip install numpy scikit-learn ortools
  ```
  (Crystal dependencies are only needed for the `multiplicative_constraint/` engine.)

## 2. Mathematical Verification

- To verify the Bost-Connes truncation and spectral equivalence claims:
  ```bash
  python src/bc_truncation_check.py
  python src/verify_conjectures.py
  ```
  - These scripts print numerical and statistical evidence for the main conjectures.

## 3. Benchmarking vs OR-Tools

- To reproduce the performance and quality benchmarks:
  ```bash
  cd src/benchmarks
  python benchmark_vs_ortools.py
  python validate_robustness.py
  ```
  - Results will match the tables in the main README and paper.

## 4. Real-World Application Demos

- To run ML clustering demos:
  ```bash
  cd src/demos
  python cluster_ml_datasets.py
  ```
  - This compares the spectral annealer to sklearn's clustering on standard datasets.

## 5. Crystal Engine (Optional)

- To run the Crystal implementation:
  ```bash
  cd multiplicative_constraint/examples
  crystal run demo.cr
  crystal run prime_necklace.cr
  ```
  - Requires [Crystal](https://crystal-lang.org/) and dependencies in `shard.yml`.

## 6. Experiment Data

- Example and random partition data is in `experiment_partitions/` and `samples/`.
- You can use these files as input to the Python or Crystal scripts for further experiments.

## 7. Troubleshooting

- If you encounter missing dependencies, ensure you have the latest versions of Python, pip, and (optionally) Crystal.
- For any issues, consult the `README.md` or open an issue on GitHub.

---

**All main results in the paper and documentation can be reproduced using the above steps.**
