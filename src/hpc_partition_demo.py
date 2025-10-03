"""Demonstrate heat-kernel spectral partitioning on an HPC workload graph."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from cnf_partition import (
    CNFPartitioner,
    ClauseGraph,
    PartitionSolution,
    solve_partition,
)
from heat_kernel_partition import sieve_primes


@dataclass
class Task:
    name: str
    flops_g: float  # billions of floating point ops per second requirement
    memory_gb: float
    datasets: List[str]


TASKS: List[Task] = [
    Task("MeshRefine", 950.0, 24.0, ["mesh", "geom"]),
    Task("ParticlePush", 1120.0, 18.0, ["particles", "fields"]),
    Task("FFT_Solver", 780.0, 12.0, ["fields", "freq"]),
    Task("BoundaryUpdate", 420.0, 8.0, ["mesh", "boundary"]),
    Task("IO_Checkpoint", 260.0, 32.0, ["checkpoint"]),
    Task("Diagnostics", 180.0, 4.0, ["particles", "analytics"]),
    Task("RadiationModel", 860.0, 20.0, ["fields", "radiation"]),
    Task("Turbulence", 690.0, 16.0, ["mesh", "tke"]),
    Task("LoadBalance", 300.0, 6.0, ["analytics", "geom"]),
    Task("Visualization", 380.0, 14.0, ["analytics", "checkpoint"]),
    Task("InletOutflow", 520.0, 10.0, ["boundary", "freq"]),
    Task("AMR_Builder", 640.0, 22.0, ["mesh", "geom", "particles"]),
]


def build_weights(tasks: List[Task]) -> np.ndarray:
    seeds = sieve_primes(20 * len(tasks))[: len(tasks)]
    norm_flops = np.array([task.flops_g for task in tasks], dtype=float)
    norm_flops /= norm_flops.max()
    norm_mem = np.array([task.memory_gb for task in tasks], dtype=float)
    norm_mem /= norm_mem.max()
    weights = []
    for seed, f_weight, m_weight in zip(seeds, norm_flops, norm_mem):
        combined = 1.0 + 0.7 * f_weight + 0.3 * m_weight
        weights.append(seed * combined)
    return np.array(weights, dtype=float)


def build_adjacency(tasks: List[Task]) -> np.ndarray:
    n = len(tasks)
    adjacency = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = tasks[i], tasks[j]
            shared = set(ti.datasets) & set(tj.datasets)
            if not shared:
                continue
            conflict = len(shared)
            load_delta = abs(ti.flops_g - tj.flops_g) / max(ti.flops_g, tj.flops_g)
            mem_delta = abs(ti.memory_gb - tj.memory_gb) / max(ti.memory_gb, tj.memory_gb)
            weight = conflict * (1.0 + 0.5 * (load_delta + mem_delta))
            adjacency[i, j] = adjacency[j, i] = weight
    return adjacency


def summarize(solution: PartitionSolution, tasks: List[Task]) -> None:
    print("Unified energy:", round(solution.energy, 6))
    print("Spectral action:", round(solution.spectral, 6))
    print("Cross-conflict weight:", round(solution.cross_conflict, 6))
    print("Fairness energy:", round(solution.fairness, 6))
    print("Weight fairness:", round(solution.weight_fairness, 6))
    for idx, indices in enumerate(solution.segments):
        if len(indices) == 0:
            print(f"  Segment {idx + 1}: (empty)")
            continue
        segment_tasks = [tasks[i] for i in indices]
        total_flops = sum(task.flops_g for task in segment_tasks)
        total_mem = sum(task.memory_gb for task in segment_tasks)
        names = ", ".join(task.name for task in segment_tasks)
        print(f"  Segment {idx + 1}: {names}")
        print(f"    FLOPs: {total_flops:.1f} GFLOP/s, Memory: {total_mem:.1f} GB")


def baseline_metrics(partitioner: CNFPartitioner, samples: int = 256, seed: int = 2026) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    energies = []
    cross_conflicts = []
    fairness_vals = []
    best_idx = -1
    for idx in range(samples):
        alpha = rng.uniform(0.0, 2.0 * math.pi, size=partitioner.num_segments)
        energy = partitioner.unified_energy(alpha)
        cross = partitioner.cross_conflict_weight(alpha)
        fairness = partitioner.fairness_energy(alpha)
        energies.append(energy)
        cross_conflicts.append(cross)
        fairness_vals.append(fairness)
        if best_idx == -1 or energy < energies[best_idx]:
            best_idx = idx
    return {
        "energy_mean": float(np.mean(energies)),
        "energy_min": float(np.min(energies)),
        "cross_mean": float(np.mean(cross_conflicts)),
        "cross_min": float(np.min(cross_conflicts)),
        "fairness_mean": float(np.mean(fairness_vals)),
        "energy_min_cross": float(cross_conflicts[best_idx]),
        "energy_min_fairness": float(fairness_vals[best_idx]),
    }


def main() -> None:
    weights = build_weights(TASKS)
    adjacency = build_adjacency(TASKS)
    graph = ClauseGraph(weights=weights, adjacency=adjacency)
    partitioner = CNFPartitioner(graph, num_segments=3)

    baseline = baseline_metrics(partitioner, samples=512, seed=7)
    print("Random baseline (512 samples):")
    print(
        f"  energy mean={baseline['energy_mean']:.3f}, min={baseline['energy_min']:.3f}; "
        f"cross mean={baseline['cross_mean']:.3f}, min={baseline['cross_min']:.3f}; "
        f"fairness mean={baseline['fairness_mean']:.3f}; "
        f"cross@energy_min={baseline['energy_min_cross']:.3f}, fairness@energy_min={baseline['energy_min_fairness']:.3f}"
    )

    solution = solve_partition(
        partitioner,
        restarts=8,
        iterations=2500,
        initial_step=0.35,
        seed=2026,
    )

    summarize(solution, TASKS)


if __name__ == "__main__":
    main()
