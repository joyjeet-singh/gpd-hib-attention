# MPO-Attention Kernel Benchmarks

This document records the empirical benchmarking results for the MERA-based MPO-attention kernel compared to a standard $O(n^2)$ attention baseline.

## Constraints
- **Hardware**: Intel Mac CPU (i5, 8GB RAM).
- **Environment**: No JAX/XLA compilation.
- **Problem Size**: Batch Size $B=1$, Sequence Length $n=256$, Hidden Dimension $d=64$.

## Benchmark Table

| Architecture | Bond Dimension ($\chi$) | Wall-Clock Time (s) | Memory (MB) | Estimated FLOPs |
| :--- | :--- | :--- | :--- | :--- |
| Standard ($O(n^2)$) | N/A | 0.011764 | 1.05 | 8,388,608 |
| MPO-Attention | 2 | 0.005440 | 1.05 | 32,768 |
| MPO-Attention | 4 | 0.005283 | 1.05 | 65,536 |
| MPO-Attention | 8 | 0.003471 | 1.05 | 131,072 |
| MPO-Attention | 16 | 0.003253 | 1.05 | 262,144 |

## Analysis
The MPO-attention kernel demonstrates significant constant-factor reduction in FLOPs compared to the standard attention baseline, while maintaining comparable memory usage for the given $n=256$ constraint. As $\chi$ increases, FLOP counts scale linearly, as expected for the localized tensor chain structure.
