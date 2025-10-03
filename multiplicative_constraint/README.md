# MultiplicativeConstraint – Heat-Kernel Constraint Partitioning Engine

General-purpose partition engine by [sethuiyer](https://github.com/sethuiyer) for arbitrary segmentable objectives. Combines heat-kernel spectral actions, weight fairness, and annealed optimization to solve:

- Non-contiguous graph partitioning (SAT, EDA, HPC payloads)
- Contiguous segmentation (dynamic programming style problems)
- Multi-objective costs (e.g., non-additive `max × min` terms)

Spectral Optimization Kernel – differentiable, auditable, and future-compatible.

### Usage Snapshot

```crystal
require "multiplicative_constraint"

weights = [12.0, 15.0, 17.0, 10.0, 8.0, 22.0]
adjacency = [
  [0.0, 2.0, 1.0, 0.0, 0.0, 4.0],
  [2.0, 0.0, 3.5, 1.0, 0.0, 2.0],
  [1.0, 3.5, 0.0, 0.0, 1.0, 0.5],
  [0.0, 1.0, 0.0, 0.0, 2.3, 0.0],
  [0.0, 0.0, 1.0, 2.3, 0.0, 1.2],
  [4.0, 2.0, 0.5, 0.0, 1.2, 0.0],
]
labels = ["TaskA", "TaskB", "TaskC", "TaskD", "TaskE", "TaskF"]

graph = MultiplicativeConstraint::Graph.new(weights, adjacency)
engine = MultiplicativeConstraint::Engine.new(graph, 3)
result = engine.solve(iterations: 1500, step: 0.35, seed: 2025)
puts MultiplicativeConstraint::Report.generate(result, labels)
```

### Computational Complexity

Let `n` = number of items, `k` = segments, `m` = heat-trace order (default 6), `s` = Hutchinson samples (default 4), and `nnz` = number of non-zero edges:

- The spectral term uses a stochastic trace with masked Laplacian matvecs → `O(s · m · nnz)` per evaluation.
- Fairness, entropy, multiplicative penalties, and cross-conflict terms operate on per-segment aggregates → `O(n + nnz)`.
- Simulated annealing runtime is `O(R · I · (s · m · nnz))` for `R` restarts and `I` iterations; `k` remains small.
- Memory footprint dominated by edge lists and segment aggregates: `O(n + nnz)`.
- Neural surrogate sampling/training (external) remains `O(S · s · m · nnz)`; inference stays near constant time.

### Empirical Latency (M1 Pro, single core)

| Scenario                            | Size / Params                     | Time |
|------------------------------------|-----------------------------------|------|
| Batch experiments (Python driver)  | `n ≤ 60`, `k ≤ 5`, `I=1200`       | ~5.0 s |
| HPC workload demo                  | 12 tasks, 512 random baseline evals | ~2.2 s |
| Neural surrogate training          | 6k samples, 300 epochs            | ~9.6 s |
| Crystal EDA partition demo         | 12 nets, 2k anneal iterations     | ~2.0 s |

For `n ≈ 50–60`, sub-10-second runs are typical with default settings. Larger instances can leverage sparse linear algebra or GPU acceleration to tame the cubic spectral term.

### License

Restricted proprietary license. Redistribution without written consent is prohibited. See `LICENSE`.
