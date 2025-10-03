# Experiments Overview

This file tracks the major experiments run so far—data, settings, and outcomes.
It’s a living document; add new entries as more runs happen.

## 1. Prime Necklace (24 Primes, 5 Segments)
- **Script:** `multiplicative_constraint/examples/prime_necklace.cr`
- **Settings:** 11 seeds (3001–3011), iterations=2600, step=0.31
- **Best Seed:** 3009
- **Energy:**
  - Unified: –2 094 999.383
  - Spectral action: –2 125 746.328
  - Fairness energy: 14.4
  - Weight fairness: 61 466.6
  - Entropy: 1.4577
  - Multiplicative penalty: 0.6091
  - Cross-conflict weight: 32.25
- **Segments:**
  - Segment 1: p₃=5, p₄=7, p₅=11
  - Segment 2: p₆=13
  - Segment 3: p₇=17 … p₁₃=41
  - Segment 4: p₁₄=43 … p₁₉=67
  - Segment 5: p₂₀=71 … p₂₄=89, p₁=2, p₂=3

## 2. Prime Necklace (All Primes < 10 000, 24 Segments)
- **Script:** `multiplicative_constraint/examples/prime_necklace_10000.cr`
- **Settings:** 6 seeds (4101–4106), iterations=1800, step=0.28
- **Best Seed:** 4101
- **Energy:**
  - Unified: –6.986×10¹⁰
  - Spectral action: –1.173×10¹¹
  - Fairness energy: 9 883
  - Weight fairness: 9.495×10¹⁰
  - Entropy: 3.0463
  - Multiplicative penalty: 0.6079
  - Cross-conflict weight: 658.5
- **Segment summary (subset):**
  - Segment 1: count=88, min=307, max=863, sum=50 994
  - Segment 2: count=97, min=877, max=1567, sum=118 107
  - …
  - Segment 24: count=87, min=2, max=9973, sum=254 964

## 3. Social Network Partition (10 Profiles)
- **Script:** `multiplicative_constraint/examples/social_network.cr`
- **Settings:** seeds=2022–2032, iterations=2200
- **Best Seed:** 2022
- **Energy:**
  - Unified: –15 244.423
  - Spectral action: –20 616.631
  - Fairness energy: 16.333
  - Weight fairness: 10 713.603
  - Entropy: 0.6390
  - Multiplicative penalty: 0.8625
  - Cross-conflict weight: 3.513
- **Segments:**
  - Segment 1: CreatorA
  - Segment 2: CreatorB
  - Segment 3: AnalystC, EngineerD, DesignerE, PhotographerF, CommunityG, FounderH, EducatorI, ResearcherJ

## 4. Surrogate Training (Crystal MLP)
- **Script:** `crystal_surrogate/train_surrogate.cr`
- **Dataset:** 1 600 random α samples for a 24-prime necklace, split 80/20
- **Model:** Two-layer MLP, hidden=24, manual SGD, LR=5e-3, 200 epochs
- **Result:**
  - Final validation correlation ≈ 0.5 (varies by seed)
  - Weights saved via `SURROGATE_EXPORT_PATH`
- **Reranking:** `crystal_surrogate/rerank.cr` loads the saved model, scores 200
  candidates, prints top-10 predicted energy vs. true energy.

## 5. Heat-Trace vs. Unified Energy Correlation
- **Scripts:** various notebooks (`run_experiments.py`, exploratory analysis)
- **Observation:** repeated sweeps show Pearson ρ ≈ 0.999 between `Tr(exp(-Lα))`
  and the total energy `E(α)` across random α, supporting the “spectral action”
  narrative.

## 6. CNF Partitioning & HPC Demo
- **Scripts:** `run_experiments.py`, `hpc_partition_demo.py`
- **Summary:**
  - CNF cases: sample DIMACS, random clauses; annealer prints energy breakdowns
    and writes partitioned CNFs.
  - HPC demo: 12 tasks, restarts=8, iterations=2500; reports baseline stats and
    final segment summaries with FLOPs/memory totals.

Add future runs below with enough detail that a reader can reproduce them or
compare their own results. For deep dives (plots, CSVs), drop them in `analysis/`
and link from here.
