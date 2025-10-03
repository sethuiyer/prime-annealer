"""Command-line tool to partition CNF formulas using heat-kernel spectral energies."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, List

import numpy as np

from cnf_partition import (
    CNFPartitioner,
    PartitionSolution,
    build_conflict_graph,
    load_dimacs,
    solve_partition,
    write_cnf_partitions,
)


def format_solution(solution: PartitionSolution) -> Dict[str, object]:
    return {
        "energy": solution.energy,
        "spectral": solution.spectral,
        "fairness": solution.fairness,
        "weight_fairness": solution.weight_fairness,
        "entropy": solution.entropy,
        "multiplicative_penalty": solution.multiplicative_penalty,
        "cross_conflict": solution.cross_conflict,
        "alpha": [float(x) for x in solution.alpha],
        "segments": [list(map(int, segment.tolist())) for segment in solution.segments],
    }


def describe_segments(solution: PartitionSolution) -> List[str]:
    lines: List[str] = []
    for idx, segment in enumerate(solution.segments):
        count = len(segment)
        segment_indices = segment.tolist()
        if count > 0:
            min_idx = min(segment_indices)
            max_idx = max(segment_indices)
        else:
            min_idx = max_idx = "-"
        lines.append(
            f"  Block {idx+1}: clauses={count}, range=[{min_idx}, {max_idx}]"
        )
    return lines


def run(args: argparse.Namespace) -> None:
    input_path = pathlib.Path(args.input).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()

    formula = load_dimacs(input_path)
    graph = build_conflict_graph(formula)
    partitioner = CNFPartitioner(graph, num_segments=args.blocks)
    solution = solve_partition(
        partitioner,
        restarts=args.restarts,
        iterations=args.iterations,
        initial_step=args.step,
        seed=args.seed,
    )

    summary_lines = [
        f"Spectral CNF partition for {input_path.name}",
        f"  Unified energy: {solution.energy:.6f}",
        f"  Spectral action: {solution.spectral:.6f}",
        f"  Fairness energy: {solution.fairness:.6f}",
        f"  Weight energy: {solution.weight_fairness:.6f}",
        f"  Entropy: {solution.entropy:.6f}",
        f"  Multiplicative penalty: {solution.multiplicative_penalty:.6f}",
        f"  Cross-conflict weight: {solution.cross_conflict:.6f}",
    ]
    summary_lines.extend(describe_segments(solution))
    print("\n".join(summary_lines))

    paths = write_cnf_partitions(formula, solution.segments, output_dir, base_name=args.base_name)
    print("\nGenerated partitions:")
    for path in paths:
        print(f"  {path}")

    if args.json:
        json_path = pathlib.Path(args.json).expanduser().resolve()
        json_path.write_text(json.dumps(format_solution(solution), indent=2), encoding="utf-8")
        print(f"Summary written to {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to DIMACS CNF file")
    parser.add_argument("--blocks", type=int, required=True, help="Number of partitions")
    parser.add_argument("--output-dir", required=True, help="Directory for partition CNFs")
    parser.add_argument("--base-name", default="partition", help="Prefix for output CNF files")
    parser.add_argument("--restarts", type=int, default=12, help="Number of annealing restarts")
    parser.add_argument("--iterations", type=int, default=4000, help="Annealing iterations per restart")
    parser.add_argument("--step", type=float, default=0.4, help="Initial proposal step size")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--json", help="Optional path to write JSON summary")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
