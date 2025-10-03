"""Empirical tests for the heat-kernel composite energy functionals."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from heat_kernel_partition import (
    HeatKernelPartitionModel,
    sieve_primes,
    composite_energy,
    multiplicative_weights_functional,
)


@dataclass
class EmpiricalStatistics:
    correlation: float
    composite_min: np.ndarray
    multiplicative_min: np.ndarray
    sup_difference: float
    rmse: float
    minimizer_distance: float
    shared_minimizer: bool

    def to_summary(self) -> str:
        lag = ", ".join(f"{x:.4f}" for x in self.composite_min)
        multi = ", ".join(f"{x:.4f}" for x in self.multiplicative_min)
        return (
            "Empirical Equivalence Summary\n"
            f"  Pearson correlation (E vs F_multi): {self.correlation:.6f}\n"
            f"  Minimizer alpha (composite): [{lag}]\n"
            f"  Minimizer alpha (multiplicative): [{multi}]\n"
            f"  Minimizer distance ||alpha_comp - alpha_mult||: {self.minimizer_distance:.6f}\n"
            f"  Sup |E - F_multi|: {self.sup_difference:.6f}\n"
            f"  RMSE(E, F_multi): {self.rmse:.6f}\n"
            f"  Shared global minimizer: {self.shared_minimizer}"
        )


def sample_alphas(rng: np.random.Generator, cuts: int, samples: int) -> np.ndarray:
    return rng.uniform(0.0, 2.0 * math.pi, size=(samples, cuts))


def empirical_test(
    *,
    n_weights: int = 32,
    cuts: int = 3,
    samples: int = 500,
    seed: int = 314159,
) -> EmpiricalStatistics:
    weights = sieve_primes(8 * n_weights)[:n_weights]
    system = HeatKernelPartitionModel(weights=weights, cuts=cuts)
    rng = np.random.default_rng(seed)
    alphas = sample_alphas(rng, cuts, samples)
    composite_vals = []
    multiplicative_vals = []

    for alpha in alphas:
        composite_vals.append(composite_energy(system, alpha))
        multiplicative_vals.append(multiplicative_weights_functional(system, alpha))

    composite_array = np.array(composite_vals)
    multiplicative_array = np.array(multiplicative_vals)

    correlation = float(np.corrcoef(composite_array, multiplicative_array)[0, 1])
    composite_min_idx = int(np.argmin(composite_array))
    multiplicative_min_idx = int(np.argmin(multiplicative_array))
    composite_min = alphas[composite_min_idx]
    multiplicative_min = alphas[multiplicative_min_idx]

    sup_difference = float(np.max(np.abs(composite_array - multiplicative_array)))
    rmse = float(np.sqrt(np.mean((composite_array - multiplicative_array) ** 2)))

    minimizer_distance = float(np.linalg.norm(composite_min - multiplicative_min))
    shared_minimizer = composite_min_idx == multiplicative_min_idx

    return EmpiricalStatistics(
        correlation=correlation,
        composite_min=composite_min,
        multiplicative_min=multiplicative_min,
        sup_difference=sup_difference,
        rmse=rmse,
        minimizer_distance=minimizer_distance,
        shared_minimizer=shared_minimizer,
    )


def main() -> None:
    stats = empirical_test()
    print(stats.to_summary())


if __name__ == "__main__":
    main()
