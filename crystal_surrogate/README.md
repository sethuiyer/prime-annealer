# Crystal Surrogate Demos

This folder contains a minimal Crystal implementation of a neural surrogate that
approximates the unified energy functional on synthetic prime-necklace data.

## 1. Train the Surrogate

```
$ CRYSTAL_CACHE_DIR=./.crystal_cache SURROGATE_EXPORT_PATH=./surrogate_state.json \
    crystal run train_surrogate.cr
```

- Generates random angle configurations, evaluates true energy via the Crystal
  engine, and trains a small two-layer MLP with manual SGD.
- Prints epoch losses, validation correlation, and a handful of target vs.
  predicted energies.
- Saves model weights to `surrogate_state.json` if `SURROGATE_EXPORT_PATH` is
  set.

## 2. Re-rank Search Candidates

```
$ CRYSTAL_CACHE_DIR=./.crystal_cache SURROGATE_MODEL=./surrogate_state.json \
    crystal run rerank.cr
```

- Loads the saved model, samples random cut vectors, scores them, and prints the
  top candidates ranked by predicted energy alongside the true energy.
- Acts as a drop-in pre-filter for annealing or other search routines.

## Notes

- These scripts mirror the Python surrogate and annealer found in the repo but
  stick to Crystal for a consistent runtime stack.
- The MLP is intentionally tiny—feel free to extend it with more layers, better
  optimisation, or dataset sampling strategies.
- For production use, train on actual annealer-labelled data rather than purely
  random samples.
