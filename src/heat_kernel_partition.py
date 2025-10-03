"""Heat-kernel based spectral partitioning utilities.

This module provides standard components for experimenting with heat kernel
spectral objectives:
- ``sieve_primes``: simple sieve of Eratosthenes (used for deterministic weights)
- ``HeatKernelPartitionModel``: builds Laplacians conditioned on angular cuts
- ``balance_entropy_penalty`` / ``multiplicative_weights_functional`` /
  ``composite_energy``: energy landscapes used by the tests and demos
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


def sieve_primes(n: int) -> List[int]:
    """Return all primes <= n using a simple sieve.

    Parameters
    ----------
    n: upper bound (inclusive) for generated primes.
    """
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(math.isqrt(n)) + 1):
        if sieve[p]:
            sieve[p * p : n + 1 : p] = False
    return np.flatnonzero(sieve).astype(int).tolist()


@dataclass
class HeatKernelPartitionModel:
    """Model of a circular graph with cut-dependent Laplacian."""

    weights: List[int]
    cuts: int

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weight list must be non-empty")
        if self.cuts <= 0:
            raise ValueError("number of cuts must be positive")
        self.weights = list(self.weights)
        self.size = len(self.weights)
        gaps = np.diff(self.weights, append=self.weights[0] + (self.weights[-1] - self.weights[-2]))
        self.weight_gaps = gaps.astype(float)

    def _cuts_from_alpha(self, alpha: np.ndarray) -> np.ndarray:
        if alpha.shape != (self.cuts,):
            raise ValueError("alpha must have length equal to number of cuts")
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
        return np.array(sorted(adjusted), dtype=int)

    def build_laplacian(self, alpha: np.ndarray) -> np.ndarray:
        """Return the Laplacian matrix associated with the cut configuration."""
        cuts = set(self._cuts_from_alpha(alpha))
        laplacian = np.zeros((self.size, self.size), dtype=float)
        for i in range(self.size):
            j = (i + 1) % self.size
            if i in cuts or j in cuts:
                continue
            gap = self.weight_gaps[i % len(self.weight_gaps)]
            weight = 1.0 / (1.0 + gap)
            laplacian[i, j] = -weight
            laplacian[j, i] = -weight
            laplacian[i, i] += weight
            laplacian[j, j] += weight
        return laplacian

    def spectral_trace_action(self, alpha: np.ndarray, kernel: str = "heat") -> float:
        """Evaluate Tr f(L_alpha) with the selected kernel."""
        laplacian = self.build_laplacian(alpha)
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
        if kernel == "heat":
            return float(np.sum(np.exp(-eigenvalues)))
        if kernel == "inverse_sqrt":
            mask = eigenvalues > 1e-12
            inverse_vals = np.zeros_like(eigenvalues)
            inverse_vals[mask] = eigenvalues[mask] ** -0.5
            return float(np.sum(inverse_vals))
        raise ValueError(f"unsupported kernel {kernel}")


def balance_entropy_penalty(system: HeatKernelPartitionModel, alpha: np.ndarray) -> float:
    """Quadratic balance penalty with entropy regularisation."""
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
    segment_sizes = np.array([seg.size for seg in segments], dtype=float)
    imbalance = segment_sizes - float(system.size) / system.cuts
    kinetic = 0.5 * float(np.dot(imbalance, imbalance))
    if segments:
        probabilities = segment_sizes / segment_sizes.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    else:
        entropy = 0.0
    return kinetic - 0.1 * entropy


def _segment_indices(system: HeatKernelPartitionModel, alpha: np.ndarray) -> List[np.ndarray]:
    cuts = np.sort(system._cuts_from_alpha(alpha))
    if cuts.size == 0:
        return [np.arange(system.size)]
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


def multiplicative_weights_factor(weights: Iterable[int]) -> float:
    omega = 1.0
    for weight in weights:
        omega *= 1.0 - 1.0 / (weight * weight)
    return omega


def multiplicative_weights_functional(system: HeatKernelPartitionModel, alpha: np.ndarray) -> float:
    base = balance_entropy_penalty(system, alpha)
    segments = _segment_indices(system, alpha)
    factor = 1.0
    for segment in segments:
        weights = [system.weights[i] for i in segment]
        factor *= multiplicative_weights_factor(weights)
    return base * factor


def composite_energy(system: HeatKernelPartitionModel, alpha: np.ndarray) -> float:
    spectral_term = -system.spectral_trace_action(alpha, kernel="heat")
    segments = _segment_indices(system, alpha)
    sizes = np.array([seg.size for seg in segments], dtype=float)
    if sizes.size > 0:
        probs = sizes / sizes.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    else:
        entropy = 0.0
    weights_sum = sum(1.0 / (w * w) for w in system.weights)
    penalty_term = multiplicative_weights_functional(system, alpha)
    return spectral_term + 0.1 * entropy + 0.01 * weights_sum + penalty_term


__all__ = [
    "HeatKernelPartitionModel",
    "balance_entropy_penalty",
    "sieve_primes",
    "composite_energy",
    "multiplicative_weights_functional",
]
