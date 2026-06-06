# ASSERT_CONVENTION: natural_units=natural, metric_signature=euclidean, fourier_convention=physics, coupling_convention=N/A, renormalization_scheme=N/A, gauge_choice=N/A
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../code')))

import numpy as np
import json
from MPO_attention_ref import mpo_attention_forward

def dense_attention(q, k, v):
    beta = 1.0 / np.sqrt(q.shape[1])
    A = np.exp((q @ k.T) * beta)
    A /= A.sum(axis=1, keepdims=True)
    return A @ v

def evaluate_tasks():
    n, d, chi = 64, 16, 4
    
    # Baseline comparison
    q = np.random.randn(n, d)
    k = np.random.randn(n, d)
    v = np.random.randn(n, d)
    
    mpo_out = mpo_attention_forward(q, k, v, chi)
    dense_out = dense_attention(q, k, v)
    
    mpo_acc = np.mean(np.isclose(mpo_out, v, atol=1e-2))
    dense_acc = np.mean(np.isclose(dense_out, v, atol=1e-2))
    
    results = {
        "mpo_accuracy": float(mpo_acc),
        "dense_accuracy": float(dense_acc),
        "error_diff": float(np.linalg.norm(mpo_out - dense_out))
    }
    
    with open("analysis/synthetic_task_results.json", "w") as f:
        json.dump(results, f)
    
    print("Comparative evaluation completed.")

if __name__ == "__main__":
    evaluate_tasks()
