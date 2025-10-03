"""Compare spectral-style annealing to optimal DP for LeetCode-hard partition problem."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class PartitionResult:
    segments: List[List[int]]
    cost: float
    alpha: np.ndarray


def segment_indices_from_alpha(alpha: np.ndarray, size: int) -> List[List[int]]:
    normalized = np.mod(alpha, 2.0 * math.pi)
    scaled = np.floor(normalized / (2.0 * math.pi) * size).astype(int)
    unique = np.unique(np.sort(scaled))
    if unique.size == 0:
        unique = np.array([0])
    segments: List[List[int]] = []
    for idx in range(unique.size):
        start = unique[idx]
        end = unique[(idx + 1) % unique.size]
        if start == end:
            segment = []
        elif start < end:
            segment = list(range(start, end))
        else:
            segment = list(range(start, size)) + list(range(0, end))
        segments.append(segment)
    return segments


def linearize_segments(segments: List[List[int]], size: int) -> List[List[int]]:
    if not segments:
        return []
    ordered_segments = sorted((min(seg) if seg else size, seg) for seg in segments)
    result: List[List[int]] = []
    seen = set()
    for _, seg in ordered_segments:
        if not seg:
            continue
        contiguous = sorted(seg)
        chunk: List[int] = []
        prev = None
        for idx in contiguous:
            if idx in seen:
                continue
            if prev is not None and idx != prev + 1 and chunk:
                result.append(chunk)
                chunk = []
            chunk.append(idx)
            seen.add(idx)
            prev = idx
        if chunk:
            result.append(chunk)
    remaining = sorted(set(range(size)) - seen)
    if remaining:
        chunk: List[int] = []
        prev = None
        for idx in remaining:
            if prev is not None and idx != prev + 1 and chunk:
                result.append(chunk)
                chunk = []
            chunk.append(idx)
            prev = idx
        if chunk:
            result.append(chunk)
    return result


def segment_cost(nums: Sequence[int], indices: List[int]) -> float:
    if not indices:
        return 0.0
    values = [nums[i] for i in indices]
    return float(max(values) * min(values))


def total_cost(nums: Sequence[int], segments: List[List[int]]) -> float:
    return sum(segment_cost(nums, segment) for segment in segments if segment)


def anneal(nums: Sequence[int], k: int, *, restarts: int = 6, iterations: int = 4000, step: float = 0.35, seed: int = 2025) -> PartitionResult:
    rng = np.random.default_rng(seed)
    size = len(nums)
    best_segments: List[List[int]] | None = None
    best_cost = float("inf")
    best_alpha: np.ndarray | None = None

    for restart in range(restarts):
        alpha = np.zeros(k)
        if k > 1:
            alpha[1:] = rng.uniform(0.0, 2.0 * math.pi, size=k - 1)
        cost = total_cost(nums, linearize_segments(segment_indices_from_alpha(alpha, size), size))
        if cost < best_cost:
            best_cost = cost
            best_segments = linearize_segments(segment_indices_from_alpha(alpha, size), size)
            best_alpha = alpha.copy()

        step_scale = step
        current_alpha = alpha
        current_cost = cost

        for iteration in range(iterations):
            temperature = max(0.02, 1.0 - iteration / iterations)
            perturb = np.zeros(k)
            if k > 1:
                perturb[1:] = rng.normal(scale=step_scale * temperature, size=k - 1)
            candidate = np.mod(current_alpha + perturb, 2.0 * math.pi)
            candidate_cost = total_cost(nums, linearize_segments(segment_indices_from_alpha(candidate, size), size))
            if candidate_cost <= current_cost or rng.random() < math.exp(-(candidate_cost - current_cost) / (temperature + 1e-8)):
                current_alpha = candidate
                current_cost = candidate_cost
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_segments = linearize_segments(segment_indices_from_alpha(candidate, size), size)
                    best_alpha = candidate.copy()
            step_scale = max(0.05, step_scale * 0.999)

    return PartitionResult(segments=best_segments or [], cost=best_cost, alpha=best_alpha if best_alpha is not None else np.zeros(k))


def optimal_dp(nums: Sequence[int], k: int) -> float:
    n = len(nums)
    dp = [[float("inf")] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        max_so_far = nums[i - 1]
        min_so_far = nums[i - 1]
        for j in range(i, 0, -1):
            max_so_far = max(max_so_far, nums[j - 1])
            min_so_far = min(min_so_far, nums[j - 1])
            cost_segment = max_so_far * min_so_far
            for parts in range(1, k + 1):
                if dp[j - 1][parts - 1] != float("inf"):
                    dp[i][parts] = min(dp[i][parts], dp[j - 1][parts - 1] + cost_segment)
    return dp[n][k]


def main() -> None:
    nums = [5, -2, 4, -1, 3, -3, 2, -4]
    k = 3
    exact = optimal_dp(nums, k)
    approx = anneal(nums, k, restarts=8, iterations=5000, step=0.4, seed=2026)

    print(f"Array: {nums}")
    print(f"k = {k}")
    print(f"Exact minimal cost (DP): {exact}")
    print(f"Annealed cost: {approx.cost}")
    print(f"Relative gap: {(approx.cost - exact) / abs(exact):.2%}")
    print("Suggested segments:")
    for idx, segment in enumerate(approx.segments, start=1):
        values = [nums[i] for i in segment]
        print(f"  Segment {idx}: indices {segment}, values {values}, cost {segment_cost(nums, segment)}")


if __name__ == "__main__":
    main()
