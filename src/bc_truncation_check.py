import math
import numpy as np
from typing import Iterable

from heat_kernel_partition import HeatKernelPartitionModel, sieve_primes


def truncated_euler_product(primes: Iterable[int], beta: float) -> float:
    prod = 1.0
    for p in primes:
        prod *= 1.0 / (1.0 - p ** (-beta))
    return prod


def zN_proxy(system: HeatKernelPartitionModel, beta: float, seed: int = 2025) -> float:
    # Proxy: use heat-trace of L_alpha as spectral component; average over a few alphas
    rng = np.random.default_rng(seed)
    samples = 8
    vals = []
    for _ in range(samples):
        alpha = rng.uniform(0.0, 2.0 * math.pi, size=system.cuts)
        L = system.build_laplacian(alpha)
        evals = np.linalg.eigvalsh(L)
        evals = np.clip(evals, a_min=1e-12, a_max=None)
        vals.append(float(np.sum(np.exp(-beta * evals))))
    return float(np.mean(vals))


def main() -> None:
    configs = [(24, 3), (48, 3), (96, 3)]
    betas = [1.5, 2.0, 3.0]
    print("N\tk\tbeta\tZ_N(proxy)\tEuler_trunc\trel_error")
    for n, k in configs:
        primes = sieve_primes(8 * n)[:n]
        system = HeatKernelPartitionModel(primes, cuts=k)
        for beta in betas:
            z_proxy = zN_proxy(system, beta)
            euler = truncated_euler_product(primes, beta)
            rel = abs(z_proxy - euler) / euler if euler > 0 else float('inf')
            print(f"{n}\t{k}\t{beta:.1f}\t{z_proxy:.6e}\t{euler:.6e}\t{rel:.3e}")


if __name__ == "__main__":
    main()
