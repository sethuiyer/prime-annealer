"""Solve a combinatorial partition problem using standard heat-kernel energies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from heat_kernel_partition import (
    HeatKernelPartitionModel,
    sieve_primes,
    composite_energy,
    multiplicative_weights_functional,
)


@dataclass
class PartitionResult:
    alpha: np.ndarray
    multiplicative_energy: float
    spectral_energy: float
    segment_sizes: List[int]
    imbalance: float
    variance: float


def _segment_indices(system: HeatKernelPartitionModel, alpha: np.ndarray) -> List[np.ndarray]:
    cuts = np.sort(system._cuts_from_alpha(alpha))
    segments: List[np.ndarray] = []
    for idx in range(cuts.size):
        start = cuts[idx]
        end = cuts[(idx + 1) % cuts.size]
        if start == end:
            segment = np.array([], dtype=int)
        elif start < end:
            segment = np.arange(start, end)
        else:
            segment = np.concatenate((np.arange(start, system.size), np.arange(0, end)))
        segments.append(segment)
    return segments


def solve_partition(
    *,
    n_weights: int = 48,
    cuts: int = 4,
    restarts: int = 8,
    iterations: int = 4000,
    initial_step: float = 0.5,
    seed: int = 20240202,
    system: HeatKernelPartitionModel | None = None,
) -> PartitionResult:
    if system is None:
        weights = sieve_primes(10 * n_weights)[:n_weights]
        system = HeatKernelPartitionModel(weights=weights, cuts=cuts)
    else:
        if system.size != n_weights or system.cuts != cuts:
            raise ValueError("provided system does not match parameters")
    rng = np.random.default_rng(seed)

    best_alpha: np.ndarray | None = None
    best_energy = math.inf
    best_spectral = math.inf

    for restart in range(restarts):
        alpha = rng.uniform(0.0, 2.0 * math.pi, size=cuts)
        current = multiplicative_weights_functional(system, alpha)
        spectral = composite_energy(system, alpha)
        step = initial_step

        if current < best_energy:
            best_energy = current
            best_alpha = alpha.copy()
            best_spectral = spectral

        for iteration in range(iterations):
            temperature = max(0.05, 1.0 - iteration / iterations)
            perturb = rng.normal(scale=step * temperature, size=cuts)
            candidate = np.mod(alpha + perturb, 2.0 * math.pi)
            candidate_value = multiplicative_weights_functional(system, candidate)

            if candidate_value <= current or rng.random() < math.exp(-(candidate_value - current) / (temperature + 1e-6)):
                alpha = candidate
                current = candidate_value
                spectral = composite_energy(system, alpha)

                if current < best_energy:
                    best_energy = current
                    best_alpha = alpha.copy()
                    best_spectral = spectral

            step = max(0.02, step * 0.999)

    if best_alpha is None:
        raise RuntimeError("solver failed to find any configuration")

    segments = _segment_indices(system, best_alpha)
    sizes = [seg.size for seg in segments]
    target = system.size / cuts
    imbalance = float(max(abs(size - target) for size in sizes))
    variance = float(np.var(sizes))

    return PartitionResult(
        alpha=best_alpha,
        multiplicative_energy=best_energy,
        spectral_energy=best_spectral,
        segment_sizes=sizes,
        imbalance=imbalance,
        variance=variance,
    )


def describe_result(system: HeatKernelPartitionModel, result: PartitionResult) -> str:
    cut_indices = [int(idx) for idx in system._cuts_from_alpha(result.alpha)]

    lines = [
        "Spectral Partition Solution",
        f"  Cuts (alpha / 2π): {[round(float(angle / (2*math.pi)), 4) for angle in result.alpha]}",
        f"  Cut indices: {cut_indices}",
        f"  Multiplicative energy: {result.multiplicative_energy:.6f}",
        f"  Spectral energy: {result.spectral_energy:.6f}",
        f"  Segment sizes: {result.segment_sizes}",
        f"  Max deviation from ideal: {result.imbalance:.3f}",
        f"  Segment size variance: {result.variance:.6f}",
    ]

    for idx, segment in enumerate(_segment_indices(system, result.alpha)):
        weights = [system.weights[i] for i in segment]
        if weights:
            total = sum(weights)
            lines.append(
                f"    Segment {idx}: count={len(weights)}, min={min(weights)}, max={max(weights)}, sum={total}"
            )
        else:
            lines.append(f"    Segment {idx}: count=0 (empty)")
    return "\n".join(lines)


def main() -> None:
    params = dict(n_weights=48, cuts=4, restarts=12, iterations=5000)
    weights = sieve_primes(10 * params["n_weights"])[0 : params["n_weights"]]
    system = HeatKernelPartitionModel(weights=weights, cuts=params["cuts"])
    result = solve_partition(system=system, **params)
    print(describe_result(system, result))


if __name__ == "__main__":
    main()
