"""Compare heat-kernel partition output against brute-force modularity (Louvain proxy)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Generator, Iterable, List, Sequence


@dataclass
class Profile:
    name: str
    daily_posts: float
    interactions: float
    communities: Sequence[str]


PROFILES: List[Profile] = [
    Profile("CreatorA", 8.5, 420.0, ("gaming", "streaming", "esports")),
    Profile("CreatorB", 6.2, 310.0, ("streaming", "music")),
    Profile("AnalystC", 4.8, 260.0, ("finance", "tech", "startups")),
    Profile("EngineerD", 3.5, 180.0, ("tech", "opensource")),
    Profile("DesignerE", 5.1, 205.0, ("design", "art", "tech")),
    Profile("PhotographerF", 7.0, 275.0, ("art", "travel")),
    Profile("CommunityG", 2.4, 520.0, ("gaming", "tech", "opensource")),
    Profile("FounderH", 3.1, 330.0, ("startups", "finance", "community")),
    Profile("EducatorI", 4.0, 295.0, ("education", "tech", "community")),
    Profile("ResearcherJ", 2.8, 150.0, ("science", "tech", "opensource")),
]


def normalised(values: Iterable[float]) -> List[float]:
    values = list(values)
    maximum = max(values) if values else 0.0
    if maximum <= 0:
        return [0.0] * len(values)
    return [v / maximum for v in values]


def build_adjacency(profiles: Sequence[Profile]) -> List[List[float]]:
    n = len(profiles)
    adjacency = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = profiles[i], profiles[j]
            shared = len(set(pi.communities) & set(pj.communities))
            if shared == 0:
                continue
            post_gap = abs(pi.daily_posts - pj.daily_posts) / (pi.daily_posts + pj.daily_posts)
            interact_gap = abs(pi.interactions - pj.interactions) / (pi.interactions + pj.interactions)
            affinity = shared + 0.5 * (2.0 - post_gap - interact_gap)
            adjacency[i][j] = adjacency[j][i] = affinity
    return adjacency


def generate_partitions(n: int, k: int) -> Generator[List[int], None, None]:
    assignment = [0] * n

    def backtrack(index: int, max_label: int) -> Generator[List[int], None, None]:
        if index == n:
            if max_label + 1 == k:
                yield assignment.copy()
            return
        for label in range(max_label + 1):
            assignment[index] = label
            yield from backtrack(index + 1, max_label)
        if max_label + 1 < k:
            assignment[index] = max_label + 1
            yield from backtrack(index + 1, max_label + 1)

    # enforce first element starts new community to remove permutation duplicates
    assignment[0] = 0
    yield from backtrack(1, 0)


def modularity(adjacency: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    m = 0.0
    n = len(adjacency)
    degrees = [0.0 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            weight = adjacency[i][j]
            degrees[i] += weight
            if i < j:
                m += weight
    if m == 0:
        return 0.0
    two_m = 2.0 * m
    total = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] != labels[j]:
                continue
            expected = degrees[i] * degrees[j] / two_m
            total += adjacency[i][j] - expected
    return total / two_m


def best_modularity(n: int, k: int, adjacency: Sequence[Sequence[float]]) -> tuple[float, List[int]]:
    best_score = -math.inf
    best_partition: List[int] = []
    for labels in generate_partitions(n, k):
        score = modularity(adjacency, labels)
        if score > best_score:
            best_score = score
            best_partition = labels
    return best_score, best_partition


def describe_partition(labels: Sequence[int]) -> List[List[str]]:
    groups: List[List[str]] = [[] for _ in range(max(labels) + 1)]
    for idx, label in enumerate(labels):
        groups[label].append(PROFILES[idx].name)
    return groups


def main() -> None:
    adjacency = build_adjacency(PROFILES)
    start = time.perf_counter()
    score, labels = best_modularity(len(PROFILES), 3, adjacency)
    elapsed = time.perf_counter() - start

    partition = describe_partition(labels)
    print("Brute-force modularity optimum (k=3):")
    print(f"  Modularity score: {score:.6f}")
    print(f"  Elapsed time: {elapsed:.3f}s")
    for idx, members in enumerate(partition, start=1):
        print(f"  Community {idx}: {', '.join(members)}")


if __name__ == "__main__":
    main()
