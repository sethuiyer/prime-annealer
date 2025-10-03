"""Utilities for partitioning CNF formulas via standard spectral energies."""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from heat_kernel_partition import sieve_primes


@dataclass
class CNFFormula:
    num_vars: int
    clauses: List[List[int]]


def load_dimacs(path: pathlib.Path) -> CNFFormula:
    """Load a DIMACS CNF file."""
    num_vars = num_clauses_declared = None
    clauses: List[List[int]] = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("c"):
                continue
            if stripped.startswith("p"):
                parts = stripped.split()
                if len(parts) != 4 or parts[1] != "cnf":
                    raise ValueError(f"invalid problem line: {stripped}")
                num_vars = int(parts[2])
                num_clauses_declared = int(parts[3])
                continue
            if num_vars is None:
                raise ValueError("DIMACS header missing before clauses")
            literals = [int(x) for x in stripped.split()]
            while literals and literals[-1] == 0:
                literals.pop()
            if not literals:
                continue
            clauses.append(literals)
    if num_vars is None or num_clauses_declared is None:
        raise ValueError("DIMACS file missing header")
    if len(clauses) != num_clauses_declared:
        # Accept mismatched counts but warn via exception for now
        raise ValueError(
            f"clause count mismatch: declared {num_clauses_declared}, found {len(clauses)}"
        )
    return CNFFormula(num_vars=num_vars, clauses=clauses)


@dataclass
class ClauseGraph:
    weights: np.ndarray  # deterministic weights per clause
    adjacency: np.ndarray  # symmetric conflict weights


def assign_clause_weights(num_clauses: int) -> np.ndarray:
    seeds = sieve_primes(20 * num_clauses)
    if len(seeds) < num_clauses:
        raise ValueError("sieve did not generate enough weights")
    return np.array(seeds[:num_clauses], dtype=float)


def build_conflict_graph(formula: CNFFormula) -> ClauseGraph:
    num_clauses = len(formula.clauses)
    weights = assign_clause_weights(num_clauses)

    adjacency = np.zeros((num_clauses, num_clauses), dtype=float)
    pos_map: List[List[int]] = [[] for _ in range(formula.num_vars + 1)]
    neg_map: List[List[int]] = [[] for _ in range(formula.num_vars + 1)]

    for idx, clause in enumerate(formula.clauses):
        seen: set[int] = set()
        for lit in clause:
            var = abs(lit)
            if var in seen:
                continue
            seen.add(var)
            if lit > 0:
                pos_map[var].append(idx)
            else:
                neg_map[var].append(idx)

    for var in range(1, formula.num_vars + 1):
        positives = pos_map[var]
        negatives = neg_map[var]
        if not positives or not negatives:
            continue
        for i in positives:
            for j in negatives:
                if i == j:
                    continue
                a, b = sorted((i, j))
                adjacency[a, b] += 1.0
                adjacency[b, a] += 1.0

    return ClauseGraph(weights=weights, adjacency=adjacency)


class CNFPartitioner:
    """Energy-based partitioner for CNF clauses."""

    def __init__(self, graph: ClauseGraph, num_segments: int) -> None:
        if num_segments <= 0:
            raise ValueError("num_segments must be positive")
        self.graph = graph
        self.num_segments = num_segments
        self.size = graph.weights.shape[0]
        self.total_weight = float(np.sum(graph.weights))

    def _cuts_from_alpha(self, alpha: np.ndarray) -> np.ndarray:
        if alpha.shape != (self.num_segments,):
            raise ValueError("alpha length mismatch")
        normalized = np.mod(alpha, 2.0 * math.pi)
        scaled = np.floor(normalized / (2.0 * math.pi) * self.size).astype(int)
        scaled = np.mod(scaled, self.size)
        scaled.sort()
        if scaled.size == 0:
            return np.array([0], dtype=int)

        adjusted: list[int] = []
        used: set[int] = set()
        for value in scaled:
            candidate = int(value)
            if self.size > 0:
                while candidate in used and len(used) < self.size:
                    candidate = (candidate + 1) % self.size
            used.add(candidate)
            adjusted.append(candidate)
            if len(used) == self.size:
                break

        if not adjusted:
            adjusted = [0]
        return np.array(sorted(set(adjusted)), dtype=int)

    def _segments(self, alpha: np.ndarray) -> List[np.ndarray]:
        cuts = self._cuts_from_alpha(alpha)
        if cuts.size == 0:
            return [np.arange(self.size)]
        segments: List[np.ndarray] = []
        for idx in range(cuts.size):
            start = cuts[idx]
            end = cuts[(idx + 1) % cuts.size]
            if start == end:
                segment = np.array([], dtype=int)
            elif start < end:
                segment = np.arange(start, end)
            else:
                segment = np.concatenate((np.arange(start, self.size), np.arange(0, end)))
            segments.append(segment)
        return segments

    def _segment_labels(self, alpha: np.ndarray) -> np.ndarray:
        labels = np.empty(self.size, dtype=int)
        for seg_idx, segment in enumerate(self._segments(alpha)):
            labels[segment] = seg_idx
        return labels

    def build_laplacian(self, alpha: np.ndarray) -> np.ndarray:
        labels = self._segment_labels(alpha)
        mask = labels[:, None] == labels[None, :]
        adjacency = self.graph.adjacency * mask
        degrees = np.sum(adjacency, axis=1)
        laplacian = np.diag(degrees) - adjacency
        return laplacian

    def spectral_action(self, alpha: np.ndarray) -> float:
        laplacian = self.build_laplacian(alpha)
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
        return float(np.sum(np.exp(-eigenvalues)))

    def segment_sizes(self, alpha: np.ndarray) -> np.ndarray:
        segments = self._segments(alpha)
        return np.array([seg.size for seg in segments], dtype=float)

    def segment_weight_sums(self, alpha: np.ndarray) -> np.ndarray:
        segments = self._segments(alpha)
        weights = self.graph.weights
        return np.array([float(np.sum(weights[segment])) for segment in segments], dtype=float)

    def fairness_energy(self, alpha: np.ndarray) -> float:
        sizes = self.segment_sizes(alpha)
        if sizes.size == 0:
            return 0.0
        target = self.size / self.num_segments
        imbalance = sizes - target
        return 0.5 * float(np.dot(imbalance, imbalance))

    def weight_energy(self, alpha: np.ndarray) -> float:
        sums = self.segment_weight_sums(alpha)
        if sums.size == 0:
            return 0.0
        target = self.total_weight / self.num_segments
        imbalance = sums - target
        return 0.5 * float(np.dot(imbalance, imbalance))

    def entropy(self, alpha: np.ndarray) -> float:
        sizes = self.segment_sizes(alpha)
        if sizes.sum() == 0:
            return 0.0
        probabilities = sizes / sizes.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            return -float(np.sum(probabilities * np.log(probabilities + 1e-12)))

    def multiplicative_weights_penalty(self, alpha: np.ndarray) -> float:
        weights = self.graph.weights
        segments = self._segments(alpha)
        product = 1.0
        for segment in segments:
            if segment.size == 0:
                continue
            factor = 1.0
            for idx in segment:
                w = weights[idx]
                factor *= 1.0 - 1.0 / (w * w)
            product *= factor
        return product

    def unified_energy(self, alpha: np.ndarray) -> float:
        spectral = -self.spectral_action(alpha)
        balance = self.fairness_energy(alpha)
        weight_balance = self.weight_energy(alpha)
        entropy_term = self.entropy(alpha)
        weight_penalty = self.multiplicative_weights_penalty(alpha)
        return spectral + balance + 0.5 * weight_balance - 0.1 * entropy_term - weight_penalty

    def segments(self, alpha: np.ndarray) -> List[np.ndarray]:
        return self._segments(alpha)

    def cross_conflict_weight(self, alpha: np.ndarray) -> float:
        labels = self._segment_labels(alpha)
        mask = labels[:, None] != labels[None, :]
        cross = np.sum(self.graph.adjacency * mask)
        return float(0.5 * cross)


@dataclass
class PartitionSolution:
    alpha: np.ndarray
    energy: float
    spectral: float
    fairness: float
    weight_fairness: float
    entropy: float
    multiplicative_penalty: float
    cross_conflict: float
    segments: List[np.ndarray]


def solve_partition(
    partitioner: CNFPartitioner,
    *,
    restarts: int = 10,
    iterations: int = 3000,
    initial_step: float = 0.4,
    seed: int = 2025,
) -> PartitionSolution:
    rng = np.random.default_rng(seed)
    best_alpha: np.ndarray | None = None
    best_energy = math.inf
    best_components: tuple[float, float, float, float, float] | None = None
    best_segments: List[np.ndarray] | None = None

    for _ in range(restarts):
        alpha = rng.uniform(0.0, 2.0 * math.pi, size=partitioner.num_segments)
        current = partitioner.unified_energy(alpha)
        spectral = partitioner.spectral_action(alpha)
        fairness = partitioner.fairness_energy(alpha)
        weight = partitioner.weight_energy(alpha)
        entropy = partitioner.entropy(alpha)
        penalty = partitioner.multiplicative_weights_penalty(alpha)
        cross = partitioner.cross_conflict_weight(alpha)
        step = initial_step

        if current < best_energy:
            best_energy = current
            best_alpha = alpha.copy()
            best_components = (spectral, fairness, weight, entropy, penalty, cross)
            best_segments = partitioner.segments(alpha)

        for iteration in range(iterations):
            temperature = max(0.02, 1.0 - iteration / iterations)
            candidate = np.mod(alpha + rng.normal(scale=step * temperature, size=alpha.shape), 2.0 * math.pi)
            candidate_energy = partitioner.unified_energy(candidate)
            if candidate_energy <= current or rng.random() < math.exp(-(candidate_energy - current) / (temperature + 1e-8)):
                alpha = candidate
                current = candidate_energy
                spectral = partitioner.spectral_action(alpha)
                fairness = partitioner.fairness_energy(alpha)
                weight = partitioner.weight_energy(alpha)
                entropy = partitioner.entropy(alpha)
                penalty = partitioner.multiplicative_weights_penalty(alpha)
                cross = partitioner.cross_conflict_weight(alpha)
                if current < best_energy:
                    best_energy = current
                    best_alpha = alpha.copy()
                    best_components = (spectral, fairness, weight, entropy, penalty, cross)
                    best_segments = partitioner.segments(alpha)
            step = max(0.01, step * 0.999)

    if best_alpha is None or best_components is None or best_segments is None:
        raise RuntimeError("failed to find partition")

    spectral, fairness, weight, entropy, penalty, cross = best_components
    return PartitionSolution(
        alpha=best_alpha,
        energy=best_energy,
        spectral=spectral,
        fairness=fairness,
        weight_fairness=weight,
        entropy=entropy,
        multiplicative_penalty=penalty,
        cross_conflict=cross,
        segments=best_segments,
    )


def write_cnf_partitions(
    formula: CNFFormula,
    segments: Sequence[Sequence[int]],
    output_dir: pathlib.Path,
    base_name: str = "partition",
) -> List[pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[pathlib.Path] = []
    for idx, clause_indices in enumerate(segments):
        clause_indices = list(clause_indices)
        clauses = [formula.clauses[i] for i in clause_indices]
        path = output_dir / f"{base_name}_{idx+1}.cnf"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"c partition {idx+1} of {len(segments)}\n")
            fh.write(f"p cnf {formula.num_vars} {len(clauses)}\n")
            for clause in clauses:
                fh.write(" ".join(str(lit) for lit in clause) + " 0\n")
        paths.append(path)
    return paths


__all__ = [
    "CNFFormula",
    "CNFPartitioner",
    "ClauseGraph",
    "PartitionSolution",
    "assign_clause_weights",
    "build_conflict_graph",
    "load_dimacs",
    "solve_partition",
    "write_cnf_partitions",
]
