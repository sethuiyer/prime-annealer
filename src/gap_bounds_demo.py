import math
import numpy as np
from typing import Tuple

from heat_kernel_partition import HeatKernelPartitionModel, sieve_primes


def build_system(n_weights: int, cuts: int) -> HeatKernelPartitionModel:
    weights = sieve_primes(8 * n_weights)[:n_weights]
    return HeatKernelPartitionModel(weights=weights, cuts=cuts)


def smallest_nonzero_eigenvalue(matrix: np.ndarray) -> float:
    evals = np.linalg.eigvalsh(matrix)
    evals = np.clip(evals, a_min=0.0, a_max=None)
    # Return the smallest strictly positive eigenvalue (if any)
    positives = evals[evals > 1e-12]
    return float(positives.min()) if positives.size > 0 else 0.0


def empirical_gap_vs_rho(n_list: Tuple[int, ...], cuts_list: Tuple[int, ...], seed: int = 1234) -> None:
    rng = np.random.default_rng(seed)
    print("rho\tN\tk\tlambda*\tw_min_bound\tpi^2*(rho^2)*w_min")
    for n in n_list:
        for k in cuts_list:
            if k <= 0 or k >= n:
                continue
            system = build_system(n, k)
            alpha = rng.uniform(0.0, 2.0 * math.pi, size=k)
            L = system.build_laplacian(alpha)
            lam_star = smallest_nonzero_eigenvalue(L)
            # Estimate segment sizes from cuts
            cuts = np.sort(system._cuts_from_alpha(alpha))
            seg_sizes = []
            for idx in range(cuts.size):
                start = cuts[idx]
                end = cuts[(idx + 1) % cuts.size]
                size = (end - start) % system.size
                seg_sizes.append(int(size))
            if not seg_sizes:
                seg_sizes = [system.size]
            max_seg = max(seg_sizes)
            # weight lower bound
            w_min = 1.0 / (1.0 + np.max(system.weight_gaps)) if system.weight_gaps.size > 0 else 1.0
            bound_path = w_min * (math.pi ** 2) / (max_seg ** 2 if max_seg > 0 else float('inf'))
            rho = k / n
            bound_density = w_min * (math.pi ** 2) * (rho ** 2)
            print(f"{rho:.3f}\t{n}\t{k}\t{lam_star:.6f}\t{bound_path:.6f}\t{bound_density:.6f}")


if __name__ == "__main__":
    empirical_gap_vs_rho(n_list=(24, 48, 96), cuts_list=(2, 3, 4, 6))
