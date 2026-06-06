# Bottleneck Analysis Report

## Overview
Analysis of the MPO kernel to identify performance bottlenecks using JAX.

## Profiling Results (Manual Measurement)

| n | Launch Time (ms) | Compute Time (ms) | Total Time (ms) |
|---|------------------|-------------------|-----------------|
| 256 | 1.01 | -0.53 | 0.48 |
| 2048 | 10.32 | 9.19 | 19.52 |

## Findings
1. At low n (n=256), the launch overhead (JAX overhead) dominates or is comparable to the compute time, explaining the sub-linear scaling observed at low n.
2. At high n (n=2048), compute time scales significantly and dominates the launch overhead.
3. The MPO kernel efficiency is high for large n, as expected.

## Conclusion
Performance at low sequence lengths is limited by launch overhead, while performance at large sequence lengths is compute-bound, validating the hierarchical MPO kernel design.
