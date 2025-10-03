"""Batch experiments for heat-kernel spectral CNF partitioning."""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from cnf_partition import (
    CNFFormula,
    CNFPartitioner,
    PartitionSolution,
    build_conflict_graph,
    load_dimacs,
    solve_partition,
    write_cnf_partitions,
)


@dataclass
class ExperimentCase:
    name: str
    formula: CNFFormula
    blocks: Sequence[int]


def generate_random_formula(
    *,
    num_vars: int,
    num_clauses: int,
    clause_size: int,
    seed: int,
) -> CNFFormula:
    rng = np.random.default_rng(seed)
    clauses: List[List[int]] = []
    for _ in range(num_clauses):
        vars_selected = rng.choice(num_vars, size=min(clause_size, num_vars), replace=False)
        clause: List[int] = []
        for var_index in vars_selected:
            literal = var_index + 1
            if rng.random() < 0.5:
                literal = -literal
            clause.append(int(literal))
        clauses.append(clause)
    return CNFFormula(num_vars=num_vars, clauses=clauses)


def sample_correlations(
    partitioner: CNFPartitioner,
    *,
    samples: int = 128,
    seed: int = 1234,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(0.0, 2.0 * math.pi, size=(samples, partitioner.num_segments))
    spectral_vals = []
    penalty_vals = []
    energy_vals = []
    fairness_vals = []

    for alpha in alphas:
        spectral_vals.append(partitioner.spectral_action(alpha))
        penalty_vals.append(partitioner.multiplicative_weights_penalty(alpha))
        energy_vals.append(partitioner.unified_energy(alpha))
        fairness_vals.append(partitioner.fairness_energy(alpha))

    spectral_arr = np.array(spectral_vals)
    penalty_arr = np.array(penalty_vals)
    energy_arr = np.array(energy_vals)
    fairness_arr = np.array(fairness_vals)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "corr_spectral_penalty": corr(spectral_arr, penalty_arr),
        "corr_energy_penalty": corr(energy_arr, penalty_arr),
        "corr_energy_spectral": corr(energy_arr, spectral_arr),
        "corr_fairness_penalty": corr(fairness_arr, penalty_arr),
    }


def describe_solution(solution: PartitionSolution) -> str:
    clauses_per_segment = [len(segment) for segment in solution.segments]
    return (
        f"energy={solution.energy:.4f}, spectral={solution.spectral:.4f}, penalty={solution.multiplicative_penalty:.4f}, "
        f"cross={solution.cross_conflict:.4f}, fairness={solution.fairness:.4f}, "
        f"weight={solution.weight_fairness:.4f}, entropy={solution.entropy:.4f}, "
        f"segment_sizes={clauses_per_segment}"
    )


def run_case(
    case: ExperimentCase,
    *,
    restarts: int = 12,
    iterations: int = 4000,
    step: float = 0.4,
    seed: int = 2025,
    output_root: pathlib.Path,
) -> None:
    print(f"\n=== Case: {case.name} (clauses={len(case.formula.clauses)}, vars={case.formula.num_vars}) ===")
    graph = build_conflict_graph(case.formula)
    for blocks in case.blocks:
        partitioner = CNFPartitioner(graph, num_segments=blocks)
        solution = solve_partition(
            partitioner,
            restarts=restarts,
            iterations=iterations,
            initial_step=step,
            seed=seed,
        )
        correlations = sample_correlations(partitioner, samples=96, seed=seed)
        print(f"Blocks={blocks} -> {describe_solution(solution)}")
        print(
            "  correlations: "
            + ", ".join(f"{key}={value:.4f}" for key, value in correlations.items())
        )

        output_dir = output_root / case.name
        base = f"{case.name}_b{blocks}"
        paths = write_cnf_partitions(case.formula, solution.segments, output_dir, base_name=base)
        for path in paths:
            print(f"    wrote {path}")


def main() -> None:
    base_dir = pathlib.Path("experiment_partitions").resolve()
    base_dir.mkdir(exist_ok=True)

    example_path = pathlib.Path("samples/example.cnf").resolve()
    example_formula = load_dimacs(example_path)
    random_formula_1 = generate_random_formula(num_vars=18, num_clauses=45, clause_size=3, seed=42)
    random_formula_2 = generate_random_formula(num_vars=24, num_clauses=60, clause_size=4, seed=123)

    cases = [
        ExperimentCase(name="example", formula=example_formula, blocks=[2, 3]),
        ExperimentCase(name="rand18x45", formula=random_formula_1, blocks=[2, 4]),
        ExperimentCase(name="rand24x60", formula=random_formula_2, blocks=[3, 5]),
    ]

    for case in cases:
        run_case(
            case,
            restarts=4,
            iterations=1200,
            step=0.25,
            seed=2025,
            output_root=base_dir,
        )


if __name__ == "__main__":
    main()
