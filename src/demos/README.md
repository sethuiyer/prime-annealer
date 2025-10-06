# Application Demos

This directory contains real-world application examples using the Prime Annealer spectral framework.

## Available Demos

### ML Dataset Clustering

**File**: `cluster_ml_datasets.py` (334 lines)

Demonstrates spectral partitioning on standard machine learning clustering datasets:
- **Iris** (3 classes, 4 features)
- **Wine** (3 classes, 13 features)
- **Digits** (10 classes, 64 features)
- **Synthetic blobs** (configurable)

Compares spectral-fairness annealing vs sklearn's spectral clustering.

**Run it:**
```bash
pip install scikit-learn numpy
cd src/demos
python cluster_ml_datasets.py
```

**What it shows:**
- Spectral methods work on real ML data
- Competitive with sklearn's implementations
- Same framework handles different feature dimensions
- Graph construction from feature vectors

## Purpose

These demos illustrate:
1. **Real-world applicability** beyond toy problems
2. **Integration with standard ML libraries** (sklearn, numpy)
3. **Practical performance** on known datasets
4. **Code examples** for adapting to your domain

## Adding Your Own Demo

To add a new application:

1. Build similarity graph from your data
2. Use `heat_kernel_partition.py` for spectral energy
3. Apply annealing from `cnf_partition.py`
4. Document results and interpretation

See existing demos for template code.

---

**Last Updated**: October 2025
