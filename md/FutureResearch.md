# Future Research Directions

This repository doubles as a proving ground for heat-kernel spectral partitioning
and prime-indexed spectral triples. Here are several follow-on projects that feel
well within reach.

## 1. Spectral Triple Refinements
- Formalise the `(A, H, D)` construction for larger prime necklaces and prove
  stability of the Bogoliubov–de Gennes Dirac operator under arbitrary cut
  permutations.
- Derive closed-form expressions for the spectral action density when segments
  wrap multiple times, and compare against numerical annealing traces.
- Explore alternative algebras (e.g., q-deformed cut operators) and test whether
  the empirical ρ ≈ 0.999 correlation between heat trace and GL energy persists.

## 2. Scaling and Performance
- Profile the Crystal annealer on >10k bead necklaces and identify opportunities
  for sparse Laplacian acceleration or GPU-backed heat-trace estimation.
- Build a reproducible benchmark suite that pits modularity, minimum-cut, and
  unified-energy solutions on increasingly large SAT and HPC graphs.
- Experiment with adaptive restart schedules that re-use spectral information to
  cut the search horizon without losing solution quality.

## 3. Learning and Surrogate Models
- Train sequence-to-angle transformers that map graph embeddings directly to
  near-optimal α vectors, using the current MLP surrogate as a baseline.
- Investigate contrastive losses that align heat trace, GL energy, and zeta
  penalties, aiming for models that generalise across domains (CNF, HPC, social).
- Integrate uncertainty estimates (e.g., ensembles or Laplace approximations) so
  surrogates can flag when annealing refinement is required.

## 4. Domain Extensions
- Apply the prime-weight framework to physical layouts (DEF/LEF), modelling vias
  and wiring jogs as additional non-commuting operators.
- Explore biochemical or supply-chain graphs where deterministic prime seeds can
  encode stoichiometry or logistical tiers.
- Collaborate with community analytics teams to validate the social-network demo
  on real creator graphs, measuring lift over modularity-centric pipelines.

## 5. Explainability and Tooling
- Build interactive dashboards that plot energy components, segment fairness,
  and cross-conflict weights over time, turning the annealer into a monitoring
  tool rather than a batch script.
- Auto-generate natural-language rationales from `Report.generate` output so
  domain experts can inspect “why these nodes were grouped” without reading raw
  dumps.
- Package small reference datasets and expected outputs to make third-party
  verification of claims (speed, energy improvements) trivial.

## 6. Mathematical Curiosities
- Analyse whether prime index segments obey analogues of Chebotarev bias when
  subjected to the multiplicative penalty.
- Connect the spectral action here to zeta-regularised determinants and compare
  against predictions in random matrix theory.
- Examine the effect of replacing primes with other deterministic sequences
  (square-free numbers, Beatty sequences) to see which preserve the spectral
  triple structure.

These projects would deepen both the theoretical footing and the practical punch
of the prime-partitioning engine. If you pick one up, let us know—progress here
compounds quickly.
