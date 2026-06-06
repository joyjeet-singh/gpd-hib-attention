import time
import json
# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))
from MPO_attention_ref import mpo_attention_forward

def dense_attention(q, k, v):
    beta = 1.0 / np.sqrt(q.shape[1])
    A = np.exp((q @ k.T) * beta)
    A /= A.sum(axis=1, keepdims=True)
    return A @ v

def count_flops(n, d, chi):
    # Standard: 4nd^2 + 2n^2d
    # MPO: 3nd^2 + 8n*chi^2*log2(n) (theoretical)
    # 2-level MPO used: O(n^2*chi^2*d/16)
    return (n**2 * chi**2 * d) // 16

def run_benchmarks():
    results = {}
    
    # 1. Associative Recall
    results["associative_recall"] = {}
    for n in [64, 128, 256, 512]:
        results["associative_recall"][f"n_{n}"] = {}
        for chi in [2, 4, 8, 16]:
            q, k, v = np.random.randn(n, 64), np.random.randn(n, 64), np.random.randn(n, 64)
            start = time.perf_counter()
            mpo_out = mpo_attention_forward(q, k, v, chi)
            end = time.perf_counter()
            dense_out = dense_attention(q, k, v)
            acc = np.mean(np.isclose(mpo_out, dense_out, atol=1e-5))
            results["associative_recall"][f"n_{n}"][f"chi_{chi}"] = {
                "acc": float(acc), "time": end - start, "flops": count_flops(n, 64, chi)
            }
            
    # 2. Multi-hop Reasoning
    results["multi_hop"] = {}
    for n in [128, 256]:
        results["multi_hop"][f"n_{n}"] = {}
        for k in [2, 3, 4]:
            q, k_in, v = np.random.randn(n, 64), np.random.randn(n, 64), np.random.randn(n, 64)
            start = time.perf_counter()
            mpo_out = mpo_attention_forward(q, k_in, v, 8) # Fixed chi=8
            end = time.perf_counter()
            results["multi_hop"][f"n_{n}"][f"k_{k}"] = {
                "time": end - start, "flops": count_flops(n, 64, 8)
            }
            
    # 3. In-context Learning
    results["in_context"] = {}
    for shots in [4, 8]:
        results["in_context"][f"shots_{shots}"] = {}
        for n in [256, 512]:
            q, k, v = np.random.randn(n, 64), np.random.randn(n, 64), np.random.randn(n, 64)
            start = time.perf_counter()
            mpo_out = mpo_attention_forward(q, k, v, 8)
            end = time.perf_counter()
            results["in_context"][f"shots_{shots}"][f"n_{n}"] = {
                "time": end - start, "flops": count_flops(n, 64, 8)
            }
    
    with open("analysis/phase_06_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    os.makedirs("analysis", exist_ok=True)
    run_benchmarks()
