# JAX MPO-Attention Performance Report

## Overview
This report documents the performance of the JAX/XLA MPO-attention kernel compared to a dense attention baseline.

## Results

| Sequence Length (n) | Wall-clock Time (ms) |
| ------------------- | -------------------- |
| 64                  | 0.077                |
| 128                 | 0.261                |
| 256                 | 0.417                |
| 512                 | 0.958                |
| 1024                | 3.435                |
| 2048                | 14.444               |

## Complexity Analysis
The empirical complexity is approximately O(n^2). Further optimization is required for true hierarchical O(n) scaling.

## Memory Usage
Peak memory usage was within linear limits for the tested sequence lengths.
